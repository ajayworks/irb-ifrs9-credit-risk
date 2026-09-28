#!/usr/bin/env python
"""Loader checks and default history across every downloaded Freddie Mac vintage.

Processes one origination vintage at a time. Loans are independent, so per-vintage
results combine exactly - and memory stays modest on a laptop. Every vintage shares
one observation cut-off: the last reporting month of the most recent vintage.

    python scripts/vintage_summary.py              # every vintage found
    python scripts/vintage_summary.py 2006 2007    # selected vintages

Writes CSVs to outputs/vintage_summary/ and prints the combined tables. The numbers
in docs/FINDINGS_LOG.md come from this script.
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcr.config import REPO_ROOT                       # noqa: E402
from hcr.data import freddie as F                      # noqa: E402
from hcr.default_def import panel as P                 # noqa: E402

OUT = REPO_ROOT / "outputs" / "vintage_summary"
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 200)


def data_end_period(years: list[int]) -> str:
    """Last reporting month of the most recent vintage - always active at data end."""
    latest = max(years)
    perf = [f for f in F.discover()["performance"] if F._file_year(f) == latest][0]
    con = P.connect(":memory:")
    raw = con.execute(
        f"SELECT MAX(column01) FROM read_csv('{perf}', delim='|', header=false, "
        f"all_varchar=true, quote='', escape='')").fetchone()[0]
    return f"{raw[:4]}-{raw[4:6]}"


def summarise_vintage(year: int, cut_off: str) -> dict[str, pd.DataFrame]:
    db = Path(tempfile.gettempdir()) / f"hcr_vintage_{year}.duckdb"
    for suffix in ("", ".wal"):
        Path(str(db) + suffix).unlink(missing_ok=True)
    con = P.connect(db)
    try:
        t = time.time()
        loaded = F.load_freddie(con, years=[year])
        con.execute("DROP TABLE orig_raw; DROP TABLE perf_raw; CHECKPOINT")
        P.build_panel(con, "baseline", data_end_period=cut_off)
        P.observation_dataset(con, "panel_baseline")
        q = loaded["quality"]
        vol, dq = q["volumes"].iloc[0], q["delinquency_status"].iloc[0]
        er = q["expense_reconciliation"].iloc[0]
        miss = q["missing_origination"].set_index("field").null_share
        diag = P.panel_diagnostics(con, "panel_baseline").iloc[0]
        out = {
            "checks": pd.DataFrame([{
                "vintage": year, "loans": vol.loans, "loan_months": vol.loan_months,
                "last_period": vol.last_period,
                "orphans": vol.orphan_performance_loans + vol.loans_without_performance,
                "recovery_nonzero": loaded["sign_check"]["recovery_values_nonzero"],
                "recovery_positive": loaded["sign_check"]["recovery_values_positive"],
                "expense_rows": er.rows_with_total_and_components,
                "expense_not_reconciling": er.rows_not_reconciling,
                "ra_rows": dq.reo_acquisition_rows, "xx_rows": dq.not_available_rows,
                "fico_null": miss["credit_score"], "ltv_null": miss["orig_ltv"],
                "dti_null": miss["dti"], "vantage_null": miss["vantage_score_4"],
                "loans_with_gaps": diag.loans_with_gaps,
                "loans_with_age_reset": diag.loans_with_loan_age_reset,
            }]),
            "zero_balance": con.execute(f"""
                WITH e AS (SELECT loan_id, zero_balance_code FROM performance
                           WHERE zero_balance_code IS NOT NULL),
                     m AS (SELECT loan_id, MAX(delinq_bucket) AS mx FROM performance
                           WHERE loan_id IN (SELECT loan_id FROM e) GROUP BY 1)
                SELECT {year} AS vintage, e.zero_balance_code AS code, COUNT(*) AS loans,
                       SUM((m.mx >= 3)::INT) AS ever_90dpd
                FROM e JOIN m USING (loan_id) GROUP BY 2""").df(),
            "vintage": con.execute(f"""
                WITH d AS (SELECT loan_id, BOOL_OR(new_default) AS ever_def,
                                  BOOL_OR(in_forbearance) AS ever_forb
                           FROM panel_baseline GROUP BY 1)
                SELECT {year} AS vintage, COUNT(*) AS loans, SUM(ever_def::INT) AS defaulted,
                       SUM(ever_forb::INT) AS forbearance_loans,
                       SUM((ever_forb AND ever_def)::INT) AS forbearance_defaulted
                FROM d""").df(),
            "obs_year": con.execute(f"""
                SELECT {year} AS vintage, SUBSTR(period,1,4) AS obs_year,
                       COUNT(*) AS n_obs, SUM(default_next_12m) AS n_default
                FROM obs_baseline GROUP BY 2""").df(),
            "events_year": con.execute(f"""
                SELECT {year} AS vintage, SUBSTR(period,1,4) AS event_year,
                       COUNT(*) FILTER (WHERE default_episode = 1) AS first_defaults,
                       COUNT(*) FILTER (WHERE default_episode > 1) AS re_defaults
                FROM panel_baseline WHERE new_default GROUP BY 2""").df(),
        }
        print(f"  {year}: {vol.loans:,} loans, {vol.loan_months:,} loan-months "
              f"({time.time() - t:.0f}s)", flush=True)
        return out
    finally:
        con.close()
        for suffix in ("", ".wal"):
            Path(str(db) + suffix).unlink(missing_ok=True)


def main(argv: list[str]) -> None:
    found = sorted({F._file_year(f) for f in F.discover()["origination"]} - {None})
    years = [int(a) for a in argv] or found
    missing = set(years) - set(found)
    if missing:
        raise SystemExit(f"No files for vintages {sorted(missing)}")
    cut_off = data_end_period(found)
    print(f"{len(years)} vintages, observation cut-off {cut_off}")

    parts: dict[str, list[pd.DataFrame]] = {}
    for y in years:
        for name, df in summarise_vintage(y, cut_off).items():
            parts.setdefault(name, []).append(df)
    t = {k: pd.concat(v, ignore_index=True) for k, v in parts.items()}
    full_run = sorted(years) == found
    if full_run:
        OUT.mkdir(parents=True, exist_ok=True)
        for name, df in t.items():
            df.to_csv(OUT / f"{name}.csv", index=False)

    c = t["checks"]
    print(f"\nLoader checks: {c.loans.sum():,} loans, {c.loan_months.sum():,} loan-months, "
          f"{c.orphans.sum()} orphans, recoveries positive "
          f"{c.recovery_positive.sum() / max(c.recovery_nonzero.sum(), 1):.2%}, "
          f"expenses not reconciling {c.expense_not_reconciling.sum()}")

    z = t["zero_balance"].groupby("code")[["loans", "ever_90dpd"]].sum()
    z["ever_90dpd_share"] = (z.ever_90dpd / z.loans).round(4)
    print(f"\nZero-balance codes\n{z.to_string()}")

    v = t["vintage"].assign(cum_default=lambda d: (d.defaulted / d.loans).round(4))
    print(f"\nCumulative default rate by vintage\n{v.to_string(index=False)}")

    y = t["obs_year"].groupby("obs_year")[["n_obs", "n_default"]].sum()
    y["one_year_default_rate"] = (y.n_default / y.n_obs).round(4)
    e = t["events_year"].groupby("event_year")[["first_defaults", "re_defaults"]].sum()
    print(f"\nOne-year default rate by observation year\n"
          f"{y.join(e, how='left').fillna(0).to_string()}")
    label = ("Long-run average" if full_run else
             "Average over THESE vintages only - NOT a long-run average (run all vintages)")
    print(f"\n{label}: pooled {y.n_default.sum() / y.n_obs.sum():.4%}, "
          f"equal weight per year {(y.n_default / y.n_obs).mean():.4%}")
    if full_run:
        print(f"\nCSVs written to {OUT.relative_to(REPO_ROOT)}/")
    else:
        print("\nSubset run - CSVs NOT written, so the full-sample results in "
              f"{OUT.relative_to(REPO_ROOT)}/ stay intact.")


if __name__ == "__main__":
    main(sys.argv[1:])
