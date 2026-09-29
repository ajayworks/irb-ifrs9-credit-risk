"""PD development sample: annual snapshots of performing loans with behavioural features.

One observation per loan per year, taken at a fixed snapshot month (December by
default). Consecutive monthly observations of the same loan are almost duplicates;
annual cohorts at a fixed reference date are the standard way to estimate one-year
default rates, and they cut ~75 million loan-months to a few million rows.

Target:   default within the 12 months after the snapshot (the panel's default_next_12m).
Eligible: not in default at the snapshot, not exiting that month, 12-month outcome known.

Every feature is known AT the snapshot date - nothing from the future.

  behavioural  current delinquency bucket, worst delinquency and months delinquent over
               the last 12 months, months on book, Freddie's estimated current LTV,
               share of the original balance still outstanding, forbearance in the last
               12 months, ever modified, prior default (defaulted before, since cured)
  origination  credit score, LTV/CLTV, DTI, rate, loan size, MI cover, term, units,
               borrowers, purpose, occupancy, property type, channel, first-time buyer,
               state, HARP

This is a BEHAVIOURAL model: it re-scores the existing book every period using how
each loan has actually performed. IRB capital and IFRS 9 both apply to the existing
book, so that is the right design - an application scorecard (origination data only)
answers a different question: whether to lend.
"""

from __future__ import annotations

import duckdb

__all__ = ["NUMERIC_FEATURES", "CATEGORICAL_FEATURES", "build_snapshots"]

NUMERIC_FEATURES = [
    # behavioural
    "delinq_bucket", "max_delinq_12m", "months_delinq_12m", "months_on_book", "eltv",
    "balance_ratio", "forbearance_12m", "modified", "prior_default",
    # origination
    "credit_score", "orig_ltv", "orig_cltv", "dti", "orig_interest_rate", "orig_upb",
    "mi_pct", "orig_term_months", "num_units", "multiple_borrowers",
]
CATEGORICAL_FEATURES = [
    "loan_purpose", "occupancy_status", "property_type", "channel", "first_time_buyer",
    "property_state", "harp_indicator",
]


def build_snapshots(con: duckdb.DuckDBPyConnection, panel: str = "panel_baseline",
                    table_name: str = "pd_snapshots", snapshot_month: int = 12) -> str:
    """Create the snapshot table from a built panel plus the `origination` table."""
    if not 1 <= snapshot_month <= 12:
        raise ValueError("snapshot_month must be 1-12")
    con.execute(f"""
    CREATE OR REPLACE TABLE {table_name} AS
    WITH h AS (
        SELECT p.loan_id, p.period, p.month_idx, p.delinq_bucket, p.months_on_book,
               p.estimated_ltv, p.current_upb, p.in_forbearance, p.restructured_to_date,
               p.default_episode, p.in_default, p.zero_balance_code, p.window_complete,
               p.default_next_12m,
               -- trailing 12 calendar months INCLUDING the snapshot month; RANGE on the
               -- month index, so a missing reporting month cannot stretch the window
               MAX(COALESCE(p.delinq_bucket, 0)) OVER w12                  AS max_delinq_12m,
               SUM(CASE WHEN COALESCE(p.delinq_bucket, 0) >= 1 THEN 1 ELSE 0 END)
                   OVER w12                                                AS months_delinq_12m,
               BOOL_OR(p.in_forbearance) OVER w12                          AS forbearance_12m
        FROM {panel} p
        WINDOW w12 AS (PARTITION BY p.loan_id ORDER BY p.month_idx
                       RANGE BETWEEN 11 PRECEDING AND CURRENT ROW)
    )
    SELECT
        h.loan_id,
        h.period                                          AS snapshot_period,
        CAST(SUBSTR(h.period, 1, 4) AS INTEGER)           AS snapshot_year,
        CAST(SUBSTR(o.orig_quarter, 1, 4) AS INTEGER)     AS vintage,
        CAST(h.default_next_12m AS INTEGER)               AS target,
        -- behavioural
        h.delinq_bucket,
        h.max_delinq_12m,
        h.months_delinq_12m,
        h.months_on_book,
        h.estimated_ltv                                   AS eltv,
        h.current_upb / NULLIF(o.orig_upb, 0)             AS balance_ratio,
        CAST(h.forbearance_12m AS INTEGER)                AS forbearance_12m,
        CAST(h.restructured_to_date AS INTEGER)           AS modified,
        CAST(h.default_episode >= 1 AS INTEGER)           AS prior_default,
        -- origination
        o.credit_score, o.orig_ltv, o.orig_cltv, o.dti, o.orig_interest_rate, o.orig_upb,
        o.mi_pct, o.orig_term_months, o.num_units,
        CAST(o.multiple_borrowers AS INTEGER)             AS multiple_borrowers,
        o.loan_purpose, o.occupancy_status, o.property_type, o.channel,
        o.first_time_buyer, o.property_state, o.harp_indicator,
        h.current_upb
    FROM h
    JOIN origination o USING (loan_id)
    WHERE CAST(SUBSTR(h.period, 6, 2) AS INTEGER) = {snapshot_month}
      AND NOT h.in_default
      AND h.zero_balance_code IS NULL
      AND h.window_complete
    ORDER BY h.loan_id, h.period
    """)
    return table_name
