"""Loan-month panel construction and the definition-of-default engine, in DuckDB SQL.

Why SQL rather than pandas: the panel is tens of millions of rows even on a sampled
slice of Freddie Mac, and the DoD logic is naturally a window-function problem -
finding each loan's first 90 DPD month, tracking a probation streak, deciding
whether a re-default is a new event. Looping in Python is both slower and harder
to read than the SQL below.

THE STATE MACHINE, expressed without a recursive CTE
----------------------------------------------------
A loan is in default from the first trigger until it has served probation:

    clean_streak(t)  = age(t) - (age of last trigger at or before t)
    exit_point(t)    = NOT raw_flag(t) AND clean_streak(t) >= probation_months
    in_default(t)    = last_trigger_age(t) IS NOT NULL
                       AND (last_exit_age(t) IS NULL OR last_trigger_age > last_exit_age)

Both `last_trigger_age` and `last_exit_age` are running MAX window functions, so the
whole thing is one pass with no recursion.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import duckdb

from hcr.config import REPO_ROOT, load

_DD = load("default_definition")
_DATA = load("data")

__all__ = ["connect", "load_synthetic", "build_panel", "observation_dataset",
           "default_rate_by_period", "compare_variants"]


def connect(database: str | Path | None = None,
            temp_directory: str | Path | None = None) -> duckdb.DuckDBPyConnection:
    """Open the analytical database.

    `temp_directory` controls where DuckDB spills large intermediate results. It
    defaults to the system temp dir, because the repo may sit on a synced or
    permission-restricted volume where spilling fails mid-query.
    """
    if database is None:
        database = REPO_ROOT / _DATA["paths"]["database"]
    if str(database) != ":memory:":
        Path(database).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(database))
    tmp = Path(temp_directory) if temp_directory else Path(tempfile.gettempdir()) / "duckdb_hcr"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{tmp}'")
    con.execute("SET preserve_insertion_order = false")
    return con


def _dod(variant: str) -> dict:
    if variant == "baseline":
        return _DD["primary"]
    for v in _DD["variants"]:
        if v["name"] == variant:
            return {**_DD["primary"], **v}
    known = ["baseline"] + [v["name"] for v in _DD["variants"]]
    raise KeyError(f"Unknown DoD variant '{variant}'. Known: {known}")


def load_synthetic(con: duckdb.DuckDBPyConnection, directory: str | Path | None = None) -> None:
    """Register the synthetic parquet files as base tables."""
    d = Path(directory) if directory else REPO_ROOT / _DATA["paths"]["raw"] / "synthetic"
    for name in ("origination", "performance", "macro"):
        path = d / f"{name}.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run: python -m hcr.data.synthetic"
            )
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM read_parquet('{path}')")
    ensure_optional_columns(con)


def ensure_optional_columns(con: duckdb.DuckDBPyConnection) -> None:
    """Columns the DoD engine reads that synthetic data does not generate."""
    for col in ("borrower_assistance_plan", "modification_flag"):
        con.execute(f"ALTER TABLE performance ADD COLUMN IF NOT EXISTS {col} VARCHAR")


def build_panel(con: duckdb.DuckDBPyConnection, variant: str = "baseline",
                table_name: str | None = None, data_end_period: str | None = None) -> str:
    """Build the loan-month panel with default flags for one DoD variant.

    `data_end_period` ('YYYY-MM') is the last month treated as observed. It defaults
    to the latest month in the data. Set it explicitly when building in batches (so
    every batch shares the same cut-off) and for out-of-time validation (to build a
    development sample that ends before the test period).
    """
    cfg = _dod(variant)
    table = table_name or f"panel_{variant}"

    utp = cfg.get("utp_zero_balance_codes") or []
    # COALESCE is load-bearing. `NULL IN (...)` returns NULL, not FALSE, so without
    # it raw_flag is NULL on every row with no zero_balance_code, `NOT raw_flag` is
    # NULL, exit_point never fires and no loan ever cures out of default - silently
    # turning the probation rule into a no-op. Caught by
    # tests/test_default_definition.py::test_longer_probation_keeps_loan_in_default_longer
    utp_pred = (
        "COALESCE(p.zero_balance_code IN (" + ", ".join(f"'{c}'" for c in utp) + "), FALSE)"
        if utp else "FALSE"
    )
    probation = int(cfg["probation_months"])
    # EBA/GL/2016/07: after a distressed restructuring the minimum probation before
    # return to non-default is ONE YEAR, not three months. Every Freddie modification
    # is a loss-mitigation modification, so a loan modified at or before the cure
    # month must serve the longer period.
    probation_restructured = int(cfg.get("probation_months_distressed_restructuring",
                                         probation))
    dpd_threshold = int(cfg["dpd_bucket_threshold"])
    horizon = int(_DD["observation"]["horizon_months"])

    # Forbearance. In Freddie data arrears keep accruing under a forbearance plan, so
    # counting them turns COVID forbearance into mass "default" (85% of 2020 default
    # events on the three starter vintages). "suspend_dpd" switches off the DPD limb
    # while a plan is active - mirroring the treatment of qualifying payment moratoria
    # under EBA/GL/2020/02 - and leaves the UTP limb running.
    suspend_dpd_in_forbearance = cfg.get("forbearance_treatment", "count_dpd") == "suspend_dpd"

    # Re-default handling. "continuation" merges a re-default occurring within
    # `redefault_window_months` of the previous episode's exit into that episode.
    continuation = cfg["redefault_treatment"] == "continuation"
    window = int(cfg.get("redefault_window_months") or 0)

    # The last reporting month in the data. Only windows that END on or before it are
    # usable - see window_complete below.
    ensure_optional_columns(con)
    last_period = data_end_period or con.execute("SELECT MAX(period) FROM performance").fetchone()[0]
    if not (isinstance(last_period, str) and len(last_period) == 7 and last_period[4] == "-"):
        raise ValueError(f"data_end_period must be 'YYYY-MM', got {last_period!r}")
    data_end_idx = int(last_period[:4]) * 12 + int(last_period[5:7])

    # TIME KEY: everything is ordered by CALENDAR MONTH (month_idx), never loan_age.
    # Freddie resets loan_age when a loan is modified (e.g. age 48 -> 6 on a HAMP-style
    # rate cut and term extension). Ordering by loan_age interleaves pre- and
    # post-modification months and scrambles the default history of exactly the
    # distressed loans that matter. Caught on real data by the duplicate-key check.
    # `months_on_book` is the non-resetting seasoning variable; use it, not loan_age.
    month_idx = ("CAST(SUBSTR({c},1,4) AS INTEGER) * 12 "
                 "+ CAST(SUBSTR({c},6,2) AS INTEGER)")

    con.execute(f"""
    CREATE OR REPLACE TABLE {table} AS
    WITH base AS (
        SELECT
            p.loan_id,
            p.period,
            {month_idx.format(c='p.period')}                 AS month_idx,
            {month_idx.format(c='o.orig_period')}            AS orig_idx,
            p.loan_age,
            p.current_upb,
            p.delinq_bucket,
            p.estimated_ltv,
            p.zero_balance_code,
            o.orig_period, o.credit_score, o.orig_ltv, o.orig_cltv, o.dti,
            o.orig_upb, o.orig_interest_rate, o.orig_term_months,
            o.loan_purpose, o.occupancy_status, o.property_state,
            o.first_time_buyer, o.num_borrowers,
            m.z AS macro_z,
            m.unemployment_rate,
            m.hpi_growth_yoy,
            CAST(p.borrower_assistance_plan AS VARCHAR) AS borrower_assistance_plan,
            CAST(p.modification_flag AS VARCHAR)        AS modification_flag,
            COALESCE(CAST(p.borrower_assistance_plan AS VARCHAR) = 'F', FALSE) AS in_forbearance,
            (COALESCE(p.delinq_bucket >= {dpd_threshold}, FALSE)
               AND NOT ({str(suspend_dpd_in_forbearance).upper()}
                        AND COALESCE(CAST(p.borrower_assistance_plan AS VARCHAR) = 'F', FALSE)))
              OR ({utp_pred}) AS raw_flag
        FROM performance p
        JOIN origination o USING (loan_id)
        LEFT JOIN macro m ON m.period = p.period
    ),
    streaks AS (
        SELECT *,
            MAX(CASE WHEN raw_flag THEN month_idx END) OVER w  AS last_trigger_idx,
            BOOL_OR(COALESCE(modification_flag IN ('Y', 'P'), FALSE)) OVER w
                                                               AS restructured_to_date,
            MAX(month_idx) OVER (PARTITION BY loan_id)         AS loan_last_idx,
            BOOL_OR(zero_balance_code IS NOT NULL)
                OVER (PARTITION BY loan_id)                    AS loan_terminated
        FROM base
        WINDOW w AS (PARTITION BY loan_id ORDER BY month_idx
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ),
    exits AS (
        SELECT *,
            month_idx - last_trigger_idx AS clean_streak,     -- NULL if never triggered
            (NOT raw_flag
             AND last_trigger_idx IS NOT NULL
             AND month_idx - last_trigger_idx >=
                 CASE WHEN restructured_to_date THEN {probation_restructured}
                      ELSE {probation} END)                   AS exit_point
        FROM streaks
    ),
    state AS (
        SELECT *,
            MAX(CASE WHEN exit_point THEN month_idx END) OVER w AS last_exit_idx
        FROM exits
        WINDOW w AS (PARTITION BY loan_id ORDER BY month_idx
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
    ),
    flagged AS (
        SELECT *,
            (last_trigger_idx IS NOT NULL
             AND (last_exit_idx IS NULL OR last_trigger_idx > last_exit_idx)) AS in_default
        FROM state
    ),
    events AS (
        SELECT *,
            in_default AND NOT COALESCE(LAG(in_default) OVER wo, FALSE)  AS new_default_raw,
            -- the month the loan RETURNED to performing (end of a default episode)
            NOT in_default AND COALESCE(LAG(in_default) OVER wo, FALSE)  AS cure_month
        FROM flagged
        WINDOW wo AS (PARTITION BY loan_id ORDER BY month_idx)
    ),
    cures AS (
        SELECT *,
            MAX(CASE WHEN cure_month THEN month_idx END) OVER
                (PARTITION BY loan_id ORDER BY month_idx
                 ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS last_cure_idx
        FROM events
    ),
    redefault AS (
        SELECT *,
            CASE
              WHEN NOT new_default_raw THEN FALSE
              -- "continuation": a re-default within the window of the previous
              -- episode's END (the cure month) is the same default, not a new one.
              -- Measured from the cure, not from the latest post-probation month -
              -- the latter is always ~1 month earlier and would merge everything.
              WHEN {str(continuation).upper()}
                   AND last_cure_idx IS NOT NULL
                   AND month_idx - last_cure_idx <= {window} THEN FALSE
              ELSE TRUE
            END AS new_default
        FROM cures
    )
    SELECT * EXCLUDE (new_default_raw),
        month_idx - orig_idx AS months_on_book,
        SUM(CASE WHEN new_default THEN 1 ELSE 0 END) OVER
            (PARTITION BY loan_id ORDER BY month_idx
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS default_episode,
        -- forward window in CALENDAR months (RANGE), robust to missing months
        COALESCE(MAX(CASE WHEN new_default THEN 1 ELSE 0 END) OVER
            (PARTITION BY loan_id ORDER BY month_idx
             RANGE BETWEEN 1 FOLLOWING AND {horizon} FOLLOWING), 0) AS default_next_{horizon}m,
        -- Cohort rule. A window is usable when it ends on or before the last month
        -- of data, and the loan is either observed through it or exits inside it
        -- (prepayment, workout) - both give a KNOWN outcome.
        --  * Requiring {horizon} rows ahead instead drops every month before a
        --    prepayment and every fast default-and-exit - a biased sample. The size
        --    of the bias on real data is reported in docs, generated by code.
        --  * Accepting exits near the data end while rejecting survivors would make
        --    the final year a sample of loans that left - so the data-end test
        --    applies to everyone.
        (month_idx + {horizon} <= {data_end_idx})
          AND (loan_last_idx >= month_idx + {horizon} OR loan_terminated) AS window_complete,
        '{variant}' AS dod_variant
    FROM redefault
    ORDER BY loan_id, month_idx
    """)

    _validate(con, table)
    return table


def _validate(con: duckdb.DuckDBPyConnection, table: str) -> None:
    """Hard assertions. Failures stop the pipeline."""
    v = _DATA["validation"]
    if v["assert_no_duplicate_loan_months"]:
        dupes = con.execute(
            f"SELECT COUNT(*) FROM (SELECT loan_id, month_idx FROM {table} "
            f"GROUP BY 1,2 HAVING COUNT(*) > 1)").fetchone()[0]
        if dupes:
            raise AssertionError(f"{table}: {dupes} duplicate (loan, calendar month) keys")
    unparsed = con.execute(f"SELECT COUNT(*) FROM {table} WHERE month_idx IS NULL").fetchone()[0]
    if unparsed:
        raise AssertionError(f"{table}: {unparsed} rows with an unparseable period")
    if v["assert_balance_non_negative"]:
        neg = con.execute(f"SELECT COUNT(*) FROM {table} WHERE current_upb < 0").fetchone()[0]
        if neg:
            raise AssertionError(f"{table}: {neg} rows with negative balance")


def panel_diagnostics(con: duckdb.DuckDBPyConnection, table: str = "panel_baseline"):
    """Soft diagnostics - legitimate data features to document, not errors."""
    return con.execute(f"""
        WITH x AS (
            SELECT loan_id, month_idx, loan_age,
                   month_idx - LAG(month_idx) OVER w AS gap,
                   loan_age  - LAG(loan_age)  OVER w AS age_step
            FROM {table}
            WINDOW w AS (PARTITION BY loan_id ORDER BY month_idx))
        SELECT COUNT(*) FILTER (WHERE gap > 1)                     AS rows_after_missing_months,
               COUNT(DISTINCT loan_id) FILTER (WHERE gap > 1)      AS loans_with_gaps,
               COUNT(DISTINCT loan_id) FILTER (WHERE age_step < 1) AS loans_with_loan_age_reset
        FROM x
    """).df()


def observation_dataset(con: duckdb.DuckDBPyConnection, panel: str = "panel_baseline",
                        table_name: str = "obs_baseline") -> str:
    """Performing loan-months whose 12-month outcome is known - the PD modelling set.

    Excludes months already in default (you cannot default from default), exit rows
    (a loan that has just prepaid is not at risk), and - when configured - windows
    cut off by the end of the data. Windows ending in a prepayment or a workout are
    KEPT: their outcome is known.
    """
    require_full = _DD["observation"]["require_full_horizon"]
    full_clause = "AND window_complete" if require_full else ""
    con.execute(f"""
    CREATE OR REPLACE TABLE {table_name} AS
    SELECT * FROM {panel}
    WHERE NOT in_default AND zero_balance_code IS NULL {full_clause}
    """)
    return table_name


def default_rate_by_period(con: duckdb.DuckDBPyConnection, obs: str = "obs_baseline"):
    horizon = int(_DD["observation"]["horizon_months"])
    return con.execute(f"""
        SELECT SUBSTR(period, 1, 4) AS year,
               COUNT(*)                              AS n_obs,
               SUM(default_next_{horizon}m)          AS n_default,
               SUM(default_next_{horizon}m) * 1.0 / COUNT(*) AS default_rate,
               AVG(macro_z)                          AS mean_z,
               AVG(unemployment_rate)                AS unemployment
        FROM {obs}
        GROUP BY 1 ORDER BY 1
    """).df()


def compare_variants(con: duckdb.DuckDBPyConnection, variants: list[str] | None = None):
    """Run every DoD variant and tabulate the impact on the observed default rate.

    This table goes straight into the MDD's definition-of-default chapter. Showing
    the sensitivity is what turns a judgement into a documented judgement.
    """
    if variants is None:
        variants = ["baseline"] + [v["name"] for v in _DD["variants"]]
    horizon = int(_DD["observation"]["horizon_months"])
    rows = []
    for name in variants:
        panel = build_panel(con, name, table_name=f"panel_{name}")
        obs = observation_dataset(con, panel, table_name=f"obs_{name}")
        r = con.execute(f"""
            SELECT COUNT(*) AS n_obs,
                   SUM(default_next_{horizon}m) AS n_default,
                   SUM(default_next_{horizon}m) * 1.0 / COUNT(*) AS default_rate,
                   (SELECT COUNT(DISTINCT loan_id) FROM {panel} WHERE new_default)
                        AS loans_ever_default
            FROM {obs}""").fetchone()
        cfg = _dod(name)
        rows.append({
            "variant": name,
            "dpd_bucket": cfg["dpd_bucket_threshold"],
            "probation_months": cfg["probation_months"],
            "redefault": cfg["redefault_treatment"],
            "utp_codes": len(cfg.get("utp_zero_balance_codes") or []),
            "n_obs": r[0], "n_default": r[1],
            "default_rate": r[2], "loans_ever_default": r[3],
        })
    import pandas as pd
    df = pd.DataFrame(rows)
    base = df.loc[df.variant == "baseline", "default_rate"].iloc[0]
    df["vs_baseline_pct"] = df.default_rate / base - 1.0
    return df
