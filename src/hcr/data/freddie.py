"""Freddie Mac Single-Family Loan-Level Dataset loader - Release 47 (July 2026) format.

Source of truth: "SFLLD Disclosure Changes - Effective July 2026" (v1.2) and the
File Layout effective July 2026. Files from before July 2026 have a different
layout; the column-count guard rejects them with an explicit message.

Four features of this release silently corrupt results if handled naively:

1. SIGN CONVENTION. Recoveries and gains are now disclosed as NEGATIVE values and
   expenses and losses as POSITIVE. Nearly every public tutorial predates this, so
   code that adds net sales proceeds to MI recoveries expecting positive numbers
   now produces LGD above 100%. This loader normalises into explicitly named
   columns - recovery_* >= 0, cost_* >= 0 - and raises if the raw signs look like
   the old convention.

2. SENTINELS. Credit score 9999, LTV/CLTV/DTI/MI% 999, units 99 and others mean
   "Not Available". Left in place, a credit score of 9999 lands in the best WOE bin
   and looks like the safest borrower in the book. All are mapped to NULL and counted.

3. DELINQUENCY STATUS is a zero-padded string: '00'..'99', 'RA' (REO acquisition),
   'XX' (not available). RA maps to bucket 99 so it triggers default; XX maps to
   NULL and is counted - never silently treated as current.

4. TOTAL EXPENSES is the total of legal, maintenance and preservation, taxes and
   insurance, and miscellaneous. Adding it AND the components double counts
   costs. The data quality report reconciles the two.
"""

from __future__ import annotations

import re
from pathlib import Path

import duckdb
import pandas as pd

from hcr.config import REPO_ROOT, load

__all__ = ["ORIG_COLUMNS", "PERF_COLUMNS", "discover", "load_freddie", "quality_report"]

_CFG = load("data")["sources"]["freddie_mac"]
_DD = load("default_definition")["primary"]

LOAN_ID_RE = re.compile(r"^[FA]\d{2}Q[1-4]\d{7}$")

# Release 47 origination file - 31 fields, positions 1..31
ORIG_COLUMNS = [
    "credit_score", "first_payment_date", "first_time_buyer", "maturity_date", "msa",
    "mi_pct", "num_units", "occupancy_status", "orig_cltv", "dti",
    "orig_upb", "orig_ltv", "orig_interest_rate", "channel", "prepayment_penalty",
    "amortization_type", "property_state", "property_type", "postal_code", "loan_id",
    "loan_purpose", "orig_term_months", "num_borrowers", "seller_name", "super_conforming",
    "pre_harp_loan_id", "special_eligibility_program", "harp_indicator",
    "property_valuation_method", "interest_only", "vantage_score_4",
]

# Release 47 monthly performance file - 35 fields, positions 1..35
PERF_COLUMNS = [
    "loan_id", "period_raw", "current_upb", "delinq_status_raw", "loan_age",
    "remaining_months", "defect_settlement_date", "modification_flag", "zero_balance_code",
    "zero_balance_date", "current_interest_rate", "non_interest_bearing_upb", "ddlpi",
    "mi_recoveries_raw", "net_sales_proceeds_raw", "non_mi_recoveries_raw",
    "total_expenses_raw", "legal_costs_raw", "maintenance_costs_raw", "taxes_insurance_raw",
    "misc_expenses_raw", "actual_loss_raw", "cumulative_mod_cost", "step_modification",
    "payment_deferral", "estimated_ltv_raw", "zero_balance_removal_upb",
    "delinquent_accrued_interest_raw", "disaster_delinquency", "borrower_assistance_plan",
    "current_period_mod_cost", "interest_bearing_upb", "mi_cancellation", "servicer_name",
    "bankruptcy_cramdown_costs",
]

assert len(ORIG_COLUMNS) == 31 and len(PERF_COLUMNS) == 35

# "Not Available" sentinels per field, from the Release 47 enumerations
ORIG_SENTINELS = {
    "credit_score": "9999", "first_time_buyer": "9", "mi_pct": "999", "num_units": "99",
    "occupancy_status": "9", "orig_cltv": "999", "dti": "999", "orig_ltv": "999",
    "channel": "9", "property_type": "99", "postal_code": "000", "loan_purpose": "9",
    "num_borrowers": "99", "property_valuation_method": "7", "vantage_score_4": "9999",
}

RECOVERY_FIELDS = ["mi_recoveries_raw", "net_sales_proceeds_raw", "non_mi_recoveries_raw"]


# --------------------------------------------------------------------------- #
# File discovery and inspection
# --------------------------------------------------------------------------- #

def discover(raw_dir: str | Path | None = None) -> dict[str, list[Path]]:
    """Find origination and performance files under the raw directory."""
    raw = Path(raw_dir) if raw_dir else REPO_ROOT / _CFG["raw_dir"]
    if not raw.exists():
        raise FileNotFoundError(f"{raw} does not exist - see docs/DATA.md")

    def match(patterns):
        found = set()
        for p in patterns:
            found.update(raw.glob(p))
        return sorted(f for f in found if f.is_file())

    orig = match(_CFG["origination_patterns"])
    perf = match(_CFG["performance_patterns"])
    if not orig or not perf:
        present = sorted(str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file())
        raise FileNotFoundError(
            f"Expected origination and performance files under {raw}.\n"
            f"  origination patterns: {_CFG['origination_patterns']}\n"
            f"  performance patterns: {_CFG['performance_patterns']}\n"
            f"  found {len(orig)} origination, {len(perf)} performance.\n"
            f"  files present: {present[:20]}{' ...' if len(present) > 20 else ''}\n"
            f"If the names differ, add a pattern in config/data.yaml. If you see .zip "
            f"files, unzip them first."
        )
    return {"origination": orig, "performance": perf}


def _inspect(path: Path, expected: int, id_position: int) -> dict:
    """Detect a header row and a trailing delimiter; enforce the column count."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        first = fh.readline().rstrip("\r\n")
        second = fh.readline().rstrip("\r\n")

    fields = first.split("|")
    header = not LOAN_ID_RE.match(fields[id_position].strip()) if len(fields) > id_position else True
    data_line = second if header else first
    n = len(data_line.split("|"))

    trailing = False
    if n == expected + 1 and data_line.endswith("|"):
        trailing, n = True, expected
    if n != expected:
        raise ValueError(
            f"{path.name}: expected {expected} fields (Release 47, July 2026 format), "
            f"found {n}. This is probably a pre-July-2026 file - re-download it from "
            f"the SFLLD Data Download page and use the July 2026 File Layout."
        )
    return {"header": header, "trailing": trailing}


def _sql_list(paths: list[Path]) -> str:
    return "[" + ", ".join("'" + str(p).replace("'", "''") + "'" for p in paths) + "]"


def _load_raw(con, files: list[Path], columns: list[str], table: str, id_position: int):
    con.execute(f"DROP TABLE IF EXISTS {table}")
    for i, f in enumerate(files):
        info = _inspect(f, len(columns), id_position)
        cols = columns + (["_trailing"] if info["trailing"] else [])
        struct = "{" + ", ".join(f"'{c}': 'VARCHAR'" for c in cols) + "}"
        select = (f"SELECT * EXCLUDE (_trailing)" if info["trailing"] else "SELECT *")
        src = (f"read_csv({_sql_list([f])}, delim='|', header={str(info['header']).lower()}, "
               f"all_varchar=true, quote='', escape='', columns={struct}, filename=true)")
        verb = f"CREATE TABLE {table} AS" if i == 0 else f"INSERT INTO {table}"
        con.execute(f"{verb} {select} FROM {src}")


# --------------------------------------------------------------------------- #
# Typing, sentinels, sign normalisation
# --------------------------------------------------------------------------- #

def _v(col: str, sentinel: str | None = None) -> str:
    expr = f"NULLIF(TRIM({col}), '')"
    return f"NULLIF({expr}, '{sentinel}')" if sentinel is not None else expr


def _num(col: str, typ: str = "DOUBLE", sentinel: str | None = None,
         alias: str | None = None, negate: bool = False) -> str:
    expr = f"TRY_CAST({_v(col, sentinel)} AS {typ})"
    return f"{'-' if negate else ''}{expr} AS {alias or col}"


def _yyyymm(col: str, alias: str) -> str:
    v = _v(col)
    return f"CASE WHEN {v} IS NULL THEN NULL ELSE SUBSTR({v},1,4) || '-' || SUBSTR({v},5,2) END AS {alias}"


def _build_origination(con) -> None:
    s = ORIG_SENTINELS
    con.execute(f"""
    CREATE OR REPLACE TABLE origination AS
    SELECT
        TRIM(loan_id) AS loan_id,
        CASE WHEN SUBSTR(TRIM(loan_id),2,2) = '99' THEN '1999'
             ELSE '20' || SUBSTR(TRIM(loan_id),2,2) END
          || SUBSTR(TRIM(loan_id),4,2)                         AS orig_quarter,
        {_yyyymm('first_payment_date', 'orig_period')},
        {_num('credit_score', 'INTEGER', s['credit_score'])},
        {_num('vantage_score_4', 'INTEGER', s['vantage_score_4'])},
        {_num('orig_ltv', 'DOUBLE', s['orig_ltv'])},
        {_num('orig_cltv', 'DOUBLE', s['orig_cltv'])},
        {_num('dti', 'DOUBLE', s['dti'])},
        {_num('mi_pct', 'DOUBLE', s['mi_pct'])},
        {_num('orig_upb')},
        {_num('orig_interest_rate')},
        {_num('orig_term_months', 'INTEGER')},
        {_num('num_units', 'INTEGER', s['num_units'])},
        {_num('num_borrowers', 'INTEGER', s['num_borrowers'])},
        -- 2018Q1 and prior code 2 as "more than one borrower"; later vintages give the
        -- exact count. This flag is consistent across the break.
        TRY_CAST({_v('num_borrowers', s['num_borrowers'])} AS INTEGER) >= 2 AS multiple_borrowers,
        {_v('first_time_buyer', s['first_time_buyer'])}     AS first_time_buyer,
        {_v('occupancy_status', s['occupancy_status'])}     AS occupancy_status,
        {_v('channel', s['channel'])}                       AS channel,
        {_v('property_type', s['property_type'])}           AS property_type,
        {_v('postal_code', s['postal_code'])}               AS postal_code,
        {_v('loan_purpose', s['loan_purpose'])}             AS loan_purpose,
        {_v('property_valuation_method', s['property_valuation_method'])} AS property_valuation_method,
        {_v('property_state')}                              AS property_state,
        {_v('msa')}                                         AS msa,
        {_v('amortization_type')}                           AS amortization_type,
        {_v('prepayment_penalty')}                          AS prepayment_penalty,
        {_v('interest_only')}                               AS interest_only,
        {_v('super_conforming')}                            AS super_conforming,
        {_v('harp_indicator')}                              AS harp_indicator,
        {_v('special_eligibility_program')}                 AS special_eligibility_program,
        {_v('pre_harp_loan_id')}                            AS pre_harp_loan_id,
        {_v('seller_name')}                                 AS seller_name,
        {_yyyymm('maturity_date', 'maturity_period')},
        filename AS source_file
    FROM orig_raw
    """)


def _build_performance(con) -> None:
    con.execute(f"""
    CREATE OR REPLACE TABLE performance AS
    SELECT
        TRIM(loan_id) AS loan_id,
        {_yyyymm('period_raw', 'period')},
        {_num('loan_age', 'INTEGER')},
        {_num('current_upb')},
        {_v('delinq_status_raw')}                           AS delinq_status_raw,
        CASE WHEN TRIM(delinq_status_raw) = 'RA' THEN 99
             WHEN TRIM(delinq_status_raw) IN ('XX', '') THEN NULL
             ELSE TRY_CAST(TRIM(delinq_status_raw) AS INTEGER) END AS delinq_bucket,
        COALESCE(TRIM(delinq_status_raw) = 'RA', FALSE)     AS reo_acquisition,
        {_num('estimated_ltv_raw', 'DOUBLE', '999', alias='estimated_ltv')},
        {_v('zero_balance_code')}                           AS zero_balance_code,
        {_yyyymm('zero_balance_date', 'zero_balance_period')},
        {_num('zero_balance_removal_upb')},
        {_num('current_interest_rate')},
        {_num('remaining_months', 'INTEGER')},
        {_num('interest_bearing_upb')},
        {_num('non_interest_bearing_upb')},
        {_yyyymm('ddlpi', 'ddlpi_period')},
        {_v('modification_flag')}                           AS modification_flag,
        {_v('step_modification')}                           AS step_modification,
        {_v('payment_deferral')}                            AS payment_deferral,
        {_v('borrower_assistance_plan')}                    AS borrower_assistance_plan,
        {_v('disaster_delinquency')}                        AS disaster_delinquency,
        {_yyyymm('defect_settlement_date', 'defect_settlement_period')},
        {_num('cumulative_mod_cost')},
        {_num('current_period_mod_cost')},
        -- Release 47 convention: recoveries NEGATIVE, costs POSITIVE. Normalised here
        -- into explicitly signed columns so downstream code cannot mix conventions.
        {_num('mi_recoveries_raw', negate=True, alias='recovery_mi')},
        {_num('net_sales_proceeds_raw', negate=True, alias='recovery_net_sales_proceeds')},
        {_num('non_mi_recoveries_raw', negate=True, alias='recovery_non_mi')},
        {_num('total_expenses_raw', alias='cost_total_expenses')},
        {_num('legal_costs_raw', alias='cost_legal')},
        {_num('maintenance_costs_raw', alias='cost_maintenance_preservation')},
        {_num('taxes_insurance_raw', alias='cost_taxes_insurance')},
        {_num('misc_expenses_raw', alias='cost_miscellaneous')},
        {_num('delinquent_accrued_interest_raw', alias='cost_delinquent_accrued_interest')},
        {_num('bankruptcy_cramdown_costs', alias='cost_bankruptcy_cramdown')},
        -- Freddie's own loss figure, loss-positive. Used to reconcile our LGD, never
        -- as a substitute for computing it.
        {_num('actual_loss_raw', alias='actual_loss')},
        {_v('mi_cancellation')}                             AS mi_cancellation,
        {_v('servicer_name')}                               AS servicer_name,
        filename AS source_file
    FROM perf_raw
    """)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def _check_loan_ids(con) -> None:
    for table in ("origination", "performance"):
        bad = con.execute(f"""
            SELECT loan_id, source_file FROM {table}
            WHERE NOT regexp_full_match(loan_id, '[FA][0-9]{{2}}Q[1-4][0-9]{{7}}')
            LIMIT 5""").fetchall()
        if bad:
            raise ValueError(
                f"{table}: loan identifiers not matching PYYQnXXXXXXX, e.g. {bad}. "
                f"Either a header row slipped through or the layout differs - check the "
                f"July 2026 File Layout."
            )


def _check_zero_balance_codes(con) -> None:
    known = sorted((_DD.get("zero_balance_codes") or {}).keys())
    placeholders = ", ".join(f"'{c}'" for c in known)
    unknown = con.execute(f"""
        SELECT zero_balance_code, COUNT(*) FROM performance
        WHERE zero_balance_code IS NOT NULL AND zero_balance_code NOT IN ({placeholders})
        GROUP BY 1 ORDER BY 2 DESC""").fetchall()
    if unknown:
        raise ValueError(
            f"Unrecognised zero-balance codes {unknown}. Known: {known}. Check the current "
            f"File Layout and classify each new code in config/default_definition.yaml "
            f"before continuing - an unclassified code is a default you may not be counting."
        )


def _check_recovery_signs(con, min_nonzero: int = 20, max_wrong_share: float = 0.05) -> dict:
    """Detect data in the pre-Release-47 sign convention.

    In Release 47, recovery fields are negative. If a material share of non-zero raw
    recoveries is positive, the file almost certainly uses the old convention.
    """
    parts = " UNION ALL ".join(
        f"SELECT TRY_CAST(NULLIF(TRIM({c}), '') AS DOUBLE) AS v FROM perf_raw"
        for c in RECOVERY_FIELDS)
    nonzero, positive = con.execute(
        f"SELECT COUNT(*) FILTER (WHERE v <> 0), COUNT(*) FILTER (WHERE v > 0) "
        f"FROM ({parts})").fetchone()
    share = positive / nonzero if nonzero else 0.0
    result = {"recovery_values_nonzero": nonzero, "recovery_values_positive": positive,
              "positive_share": share}
    if nonzero >= min_nonzero and share > max_wrong_share:
        raise ValueError(
            f"{share:.1%} of non-zero raw recovery values are POSITIVE. Release 47 "
            f"discloses recoveries as NEGATIVE, so this looks like pre-July-2026 data. "
            f"Loading it would invert every recovery and produce LGD above 100%."
        )
    return result


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def _file_year(path: Path) -> int | None:
    m = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", path.name)
    return int(m.group(1)) if m else None


def vintages(raw_dir: str | Path | None = None) -> list[int]:
    """Origination vintages present in the raw directory."""
    return sorted({_file_year(f) for f in discover(raw_dir)["origination"]} - {None})


def data_end_period(raw_dir: str | Path | None = None) -> str:
    """Last reporting month in the data, as 'YYYY-MM'.

    Read from the most recent vintage's performance files: the newest loans are always
    still active at the data end, so their last period is the dataset's last period.
    Batch builds pass this to every vintage so they share one observation cut-off.
    """
    perf = discover(raw_dir)["performance"]
    latest = max(y for y in map(_file_year, perf) if y is not None)
    files = [f for f in perf if _file_year(f) == latest]
    raw = duckdb.connect().execute(
        f"SELECT MAX(column01) FILTER (WHERE regexp_full_match(column01, '[0-9]{{6}}')) "
        f"FROM read_csv({_sql_list(files)}, delim='|', header=false, all_varchar=true, "
        f"quote='', escape='')").fetchone()[0]
    if raw is None:
        raise ValueError(f"No YYYYMM periods found in {[f.name for f in files]}")
    return f"{raw[:4]}-{raw[4:6]}"


def load_freddie(con: duckdb.DuckDBPyConnection, raw_dir: str | Path | None = None,
                 min_nonzero_for_sign_check: int = 20,
                 years: list[int] | None = None) -> dict:
    """Load Release 47 files into `origination` and `performance` tables.

    The resulting tables have the columns `default_def.panel.build_panel` expects,
    so the panel, DoD engine and everything downstream run unchanged.

    `years` restricts the load to those origination vintages - useful for iterating
    on a laptop, or for processing the full history in memory-sized batches (loans
    are independent, so per-vintage results combine exactly).
    """
    files = discover(raw_dir)
    if years is not None:
        wanted = set(years)
        files = {k: [f for f in v if _file_year(f) in wanted] for k, v in files.items()}
        if not files["origination"] or not files["performance"]:
            raise FileNotFoundError(f"No origination/performance files for years {sorted(wanted)}")
    _load_raw(con, files["origination"], ORIG_COLUMNS, "orig_raw", id_position=19)
    _load_raw(con, files["performance"], PERF_COLUMNS, "perf_raw", id_position=0)

    signs = _check_recovery_signs(con, min_nonzero=min_nonzero_for_sign_check)
    _build_origination(con)
    _build_performance(con)
    _check_loan_ids(con)
    _check_zero_balance_codes(con)

    # Macro placeholder until the FRED/FHFA loader exists; the panel LEFT JOINs it.
    con.execute("""CREATE TABLE IF NOT EXISTS macro
                   (period VARCHAR, z DOUBLE, unemployment_rate DOUBLE, hpi_growth_yoy DOUBLE)""")

    return {"files": files, "sign_check": signs, "quality": quality_report(con)}


def quality_report(con: duckdb.DuckDBPyConnection) -> dict[str, pd.DataFrame]:
    """Data quality summary. Goes straight into the MDD's data chapter."""
    q = lambda sql: con.execute(sql).df()  # noqa: E731
    return {
        "volumes": q("""
            SELECT (SELECT COUNT(*) FROM origination)                AS loans,
                   (SELECT COUNT(*) FROM performance)                AS loan_months,
                   (SELECT MIN(orig_quarter) FROM origination)       AS first_vintage,
                   (SELECT MAX(orig_quarter) FROM origination)       AS last_vintage,
                   (SELECT MIN(period) FROM performance)             AS first_period,
                   (SELECT MAX(period) FROM performance)             AS last_period,
                   (SELECT COUNT(*) FROM origination o WHERE NOT EXISTS
                       (SELECT 1 FROM performance p WHERE p.loan_id = o.loan_id))
                                                                    AS loans_without_performance,
                   (SELECT COUNT(DISTINCT loan_id) FROM performance p WHERE NOT EXISTS
                       (SELECT 1 FROM origination o WHERE o.loan_id = p.loan_id))
                                                                    AS orphan_performance_loans
        """),
        "missing_origination": q("""
            SELECT 'credit_score' AS field, AVG((credit_score IS NULL)::INT) AS null_share FROM origination
            UNION ALL SELECT 'orig_ltv', AVG((orig_ltv IS NULL)::INT) FROM origination
            UNION ALL SELECT 'orig_cltv', AVG((orig_cltv IS NULL)::INT) FROM origination
            UNION ALL SELECT 'dti', AVG((dti IS NULL)::INT) FROM origination
            UNION ALL SELECT 'mi_pct', AVG((mi_pct IS NULL)::INT) FROM origination
            UNION ALL SELECT 'occupancy_status', AVG((occupancy_status IS NULL)::INT) FROM origination
            UNION ALL SELECT 'loan_purpose', AVG((loan_purpose IS NULL)::INT) FROM origination
            UNION ALL SELECT 'vantage_score_4', AVG((vantage_score_4 IS NULL)::INT) FROM origination
        """),
        "delinquency_status": q("""
            SELECT COUNT(*) FILTER (WHERE delinq_status_raw = 'RA')      AS reo_acquisition_rows,
                   COUNT(*) FILTER (WHERE delinq_status_raw = 'XX'
                                     OR delinq_status_raw IS NULL)       AS not_available_rows,
                   COUNT(*) FILTER (WHERE delinq_bucket >= 3)            AS dpd_90_plus_rows,
                   COUNT(*)                                              AS total_rows
            FROM performance
        """),
        "zero_balance_codes": q("""
            SELECT zero_balance_code, COUNT(*) AS loans
            FROM performance WHERE zero_balance_code IS NOT NULL
            GROUP BY 1 ORDER BY 1
        """),
        "expense_reconciliation": q("""
            SELECT COUNT(*) AS rows_with_total_and_components,
                   COUNT(*) FILTER (WHERE ABS(cost_total_expenses
                       - (COALESCE(cost_legal,0) + COALESCE(cost_maintenance_preservation,0)
                          + COALESCE(cost_taxes_insurance,0) + COALESCE(cost_miscellaneous,0)))
                       > 1.0) AS rows_not_reconciling
            FROM performance
            WHERE cost_total_expenses IS NOT NULL
              AND (cost_legal IS NOT NULL OR cost_maintenance_preservation IS NOT NULL
                   OR cost_taxes_insurance IS NOT NULL OR cost_miscellaneous IS NOT NULL)
        """),
    }


if __name__ == "__main__":
    from hcr.default_def import panel as P

    con = P.connect()
    out = load_freddie(con)
    pd.set_option("display.width", 200)
    print("files:")
    for k, v in out["files"].items():
        print(f"  {k:<12} {len(v)}: {[p.name for p in v][:6]}")
    print(f"\nsign check: {out['sign_check']}")
    for name, df in out["quality"].items():
        print(f"\n--- {name} ---\n{df.to_string(index=False)}")

    print("\nBuilding panel and definition of default ...")
    t = P.build_panel(con, "baseline")
    obs = P.observation_dataset(con, t)
    d = P.default_rate_by_period(con, obs)
    print(d[["year", "n_obs", "n_default", "default_rate"]].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))
