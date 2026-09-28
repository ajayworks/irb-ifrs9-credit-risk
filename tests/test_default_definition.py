"""Definition-of-default state machine, tested against hand-constructed cases.

Each case is a delinquency sequence where the correct answer is known by hand.
This is the only honest way to test the state machine - on real or synthetic data
you cannot tell a subtle probation bug from a genuine data pattern.
"""

from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from hcr.default_def import panel as P


def _period(m: int) -> str:
    """Month m (1-based) counted from 2010-01, as 'YYYY-MM'."""
    return f"{2010 + (m - 1) // 12}-{(m - 1) % 12 + 1:02d}"


def _panel_from_sequence(delinq: list[int], probation: int = 3,
                         zb: list[str | None] | None = None,
                         ages: list[int] | None = None,
                         variant: str = "baseline",
                         forbearance: list[bool] | None = None,
                         modified: list[bool] | None = None,
                         extend_data_to: int | None = None,
                         data_end_period: str | None = None) -> pd.DataFrame:
    """Build a single-loan panel from a delinquency bucket sequence.

    Rows are consecutive calendar months from 2010-01. `ages` overrides loan_age
    (e.g. the reset Freddie applies on modification). `forbearance` / `modified` set
    the assistance and modification flags per month. `extend_data_to` adds a
    performing filler loan so the data's last month lies beyond the test loan's -
    as it does in real data. Returns the test loan only, with a 1-based calendar
    position `seq` independent of loan_age.
    """
    n = len(delinq)
    con = duckdb.connect(":memory:")
    con.execute("CREATE TABLE origination AS SELECT * FROM (VALUES "
                "('L1','2010-01',700,80,82,35,200000.0,4.5,360,'P','O','CA','N',2),"
                "('L0','2010-01',700,80,82,35,200000.0,4.5,360,'P','O','CA','N',2)) "
                "t(loan_id,orig_period,credit_score,orig_ltv,orig_cltv,dti,orig_upb,"
                "orig_interest_rate,orig_term_months,loan_purpose,occupancy_status,"
                "property_state,first_time_buyer,num_borrowers)")
    perf = pd.DataFrame({
        "loan_id": ["L1"] * n,
        "period": [_period(m) for m in range(1, n + 1)],
        "loan_age": ages if ages is not None else list(range(1, n + 1)),
        "current_upb": [200000.0 - 100 * i for i in range(n)],
        "delinq_bucket": delinq,
        "estimated_ltv": [80.0] * n,
        "zero_balance_code": zb if zb else [None] * n,
        "borrower_assistance_plan": ["F" if f else None for f in (forbearance or [False] * n)],
        "modification_flag": ["Y" if m else None for m in (modified or [False] * n)],
    })
    if extend_data_to:
        filler = pd.DataFrame({
            "loan_id": ["L0"] * extend_data_to,
            "period": [_period(m) for m in range(1, extend_data_to + 1)],
            "loan_age": list(range(1, extend_data_to + 1)),
            "current_upb": [150000.0] * extend_data_to, "delinq_bucket": [0] * extend_data_to,
            "estimated_ltv": [70.0] * extend_data_to, "zero_balance_code": [None] * extend_data_to,
            "borrower_assistance_plan": [None] * extend_data_to,
            "modification_flag": [None] * extend_data_to,
        })
        perf = pd.concat([perf, filler], ignore_index=True)
    con.register("perf_df", perf)
    con.execute("CREATE TABLE performance AS SELECT * FROM perf_df")
    con.execute("CREATE TABLE macro AS SELECT * FROM (VALUES ('2010-01', 0.0, 5.0, 0.02)) "
                "t(period, z, unemployment_rate, hpi_growth_yoy)")

    original = P._DD["primary"]["probation_months"]
    P._DD["primary"]["probation_months"] = probation
    try:
        t = P.build_panel(con, variant, table_name="p", data_end_period=data_end_period)
        df = con.execute(f"SELECT * FROM {t} WHERE loan_id = 'L1' ORDER BY month_idx").df()
        df.insert(0, "seq", range(1, len(df) + 1))
        return df
    finally:
        P._DD["primary"]["probation_months"] = original


def test_never_defaults():
    df = _panel_from_sequence([0, 0, 1, 0, 0, 1, 2, 1, 0, 0])
    assert not df.in_default.any()
    assert df.new_default.sum() == 0


def test_enters_default_at_90dpd():
    df = _panel_from_sequence([0, 1, 2, 3, 3, 3])
    assert not df.in_default[:3].any(), "should not be in default before bucket 3"
    assert df.in_default[3:].all(), "should be in default from the first 90 DPD month"
    assert df.new_default.sum() == 1
    assert df.loc[df.new_default, "loan_age"].iloc[0] == 4


def test_probation_must_be_served_before_cure():
    # default at age 4, clean from age 5. With 3-month probation the loan exits
    # at age 8 (age - last_trigger_age = 8 - 4 = 4 >= 3 first satisfied at age 7).
    df = _panel_from_sequence([0, 1, 2, 3, 0, 0, 0, 0, 0, 0], probation=3)
    in_def = df.set_index("loan_age").in_default
    assert in_def[4], "in default at trigger"
    assert in_def[5] and in_def[6], "still in default during probation"
    assert not in_def[7], "exits once probation is served"


def test_longer_probation_keeps_loan_in_default_longer():
    seq = [0, 1, 2, 3] + [0] * 14
    short = _panel_from_sequence(seq, probation=3).in_default.sum()
    long_ = _panel_from_sequence(seq, probation=12).in_default.sum()
    assert long_ > short, (
        f"12-month probation must keep the loan in default longer than 3-month "
        f"(got {long_} vs {short}) - if equal, the probation rule is a silent no-op"
    )
    assert long_ - short == 9


def test_redefault_after_cure_is_a_new_event():
    # default, cure and serve probation, then default again
    df = _panel_from_sequence([0, 3, 0, 0, 0, 0, 3, 3], probation=3)
    assert df.new_default.sum() == 2, "a re-default after a served cure is a second event"
    assert df.default_episode.max() == 2


def test_redefault_during_probation_is_not_a_new_event():
    # goes back to 90+ before probation completes - still the same default
    df = _panel_from_sequence([0, 3, 0, 3, 3, 0, 0, 0], probation=3)
    assert df.new_default.sum() == 1, "re-entry during probation is the same episode"


def test_unlikeliness_to_pay_triggers_without_arrears():
    # current on payments, but the loan terminates in an REO disposition
    df = _panel_from_sequence([0, 0, 0, 0], zb=[None, None, None, "09"])
    assert df.in_default.iloc[-1], "UTP limb must trigger default without 90 DPD"
    assert df.new_default.sum() == 1


def test_prepayment_is_not_default():
    df = _panel_from_sequence([0, 0, 0, 0], zb=[None, None, None, "01"])
    assert not df.in_default.any(), "code 01 is prepaid/matured, not a credit event"


# --------------------------------------------------------------------------- #
# Regressions found on real Freddie Mac data
# --------------------------------------------------------------------------- #

def test_loan_age_reset_on_modification_does_not_scramble_history():
    """Freddie resets loan_age on modification (seen: 48 -> 6). The panel must key
    on calendar month. Ordered by loan_age, this sequence would interleave pre- and
    post-modification months and the duplicate ages would break the key."""
    delinq = [0, 1, 2, 3, 3, 0, 0, 0, 0, 0]
    ages = [44, 45, 46, 47, 48, 6, 7, 8, 9, 10]          # modified at position 6
    df = _panel_from_sequence(delinq, ages=ages, probation=3)
    assert df.new_default.sum() == 1
    assert df.loc[df.new_default, "seq"].iloc[0] == 4
    in_def = df.set_index("seq").in_default
    assert in_def[4] and in_def[5] and in_def[6] and in_def[7]
    assert not in_def[8], "cures once 3 clean calendar months have passed"
    assert df.months_on_book.is_monotonic_increasing, "seasoning must not reset"


def test_prepayment_inside_the_window_is_a_valid_non_default_observation():
    """Requiring 12 rows ahead would drop every month before a prepayment."""
    df = _panel_from_sequence([0, 0, 0, 0, 0], zb=[None, None, None, None, "01"],
                              extend_data_to=20)
    early = df[df.seq <= 4]
    assert early.window_complete.all()
    assert (early.default_next_12m == 0).all()


def test_fast_default_and_exit_is_a_valid_default_observation():
    """A loan that defaults and is worked out inside the window has a known outcome.
    Dropping it would remove the fastest, most severe defaults."""
    df = _panel_from_sequence([0, 0, 1, 2, 3, 3, 3], zb=[None] * 6 + ["09"], extend_data_to=20)
    before = df[df.seq <= 3]
    assert before.window_complete.all()
    assert (before.default_next_12m == 1).all()


def test_explicit_data_end_sets_the_observation_cut_off():
    """Batches must share one cut-off, and out-of-time validation needs an earlier one."""
    default = _panel_from_sequence([0] * 30)
    assert default[default.window_complete].seq.max() == 18
    cut = _panel_from_sequence([0] * 30, data_end_period=_period(20))
    assert cut[cut.window_complete].seq.max() == 8


def test_malformed_data_end_is_rejected():
    with pytest.raises(ValueError, match="YYYY-MM"):
        _panel_from_sequence([0] * 5, data_end_period="202603")


def test_exits_near_the_data_end_are_not_cherry_picked():
    """If survivors near the data end are excluded, exits must be too - otherwise the
    final year is a sample of loans that left."""
    df = _panel_from_sequence([0, 0, 0, 0, 0], zb=[None] * 4 + ["01"], extend_data_to=6)
    assert not df.window_complete.any()


# --------------------------------------------------------------------------- #
# Forbearance and distressed restructuring
# --------------------------------------------------------------------------- #

def test_forbearance_suspends_the_dpd_limb():
    """Arrears accrue under a forbearance plan, then a payment deferral brings the
    loan current. Under suspend_dpd that is not a default; counting DPD makes it one."""
    delinq = [0, 1, 2, 3, 4, 5, 0, 0, 0, 0]
    forb = [False, True, True, True, True, True, False, False, False, False]
    assert _panel_from_sequence(delinq, forbearance=forb).new_default.sum() == 0
    counted = _panel_from_sequence(delinq, forbearance=forb, variant="forbearance_dpd_counts")
    assert counted.new_default.sum() == 1
    assert counted.loc[counted.new_default, "seq"].iloc[0] == 4


def test_arrears_persisting_after_forbearance_ends_trigger_default():
    delinq = [0, 1, 2, 3, 4, 5, 6, 7]
    forb = [False, True, True, True, True, False, False, False]
    df = _panel_from_sequence(delinq, forbearance=forb)
    assert df.new_default.sum() == 1
    assert df.loc[df.new_default, "seq"].iloc[0] == 6, "defaults when the plan ends"


def test_unlikeliness_to_pay_still_applies_during_forbearance():
    df = _panel_from_sequence([0, 1, 2, 3], forbearance=[False, True, True, True],
                              zb=[None, None, None, "03"])
    assert df.new_default.sum() == 1


def test_distressed_restructuring_requires_twelve_months_probation():
    """EBA/GL/2016/07: one year after a distressed restructuring, not three months."""
    delinq = [0, 3, 3] + [0] * 14                         # last trigger at seq 3
    modified = [False, False, False, True] + [False] * 13
    df = _panel_from_sequence(delinq, modified=modified)
    in_def = df.set_index("seq").in_default
    assert in_def[14] and not in_def[15], "exits 12 clean months after the last trigger"
    uniform = _panel_from_sequence(delinq, modified=modified, variant="uniform_3m_probation")
    assert not uniform.set_index("seq").in_default[6]


def test_restructured_redefault_inside_twelve_months_is_the_same_default():
    """The 3-month rule turns one struggling modified loan into two defaults."""
    delinq = [0, 3, 3, 0, 0, 0, 0, 0, 0, 3, 3]
    modified = [False, False, False, True] + [False] * 7
    assert _panel_from_sequence(delinq, modified=modified).new_default.sum() == 1
    assert _panel_from_sequence(delinq, modified=modified,
                                variant="uniform_3m_probation").new_default.sum() == 2


def test_windows_cut_off_by_end_of_data_are_incomplete():
    df = _panel_from_sequence([0] * 20)                  # still active at data end
    assert df[df.seq <= 8].window_complete.all()
    assert not df[df.seq >= 9].window_complete.any()


def test_continuation_merges_only_redefaults_inside_the_window():
    """Measured from the cure month. Measuring from the latest post-probation month
    (the original code) is always ~1 month and merged every re-default."""
    near = [0, 3, 0, 0, 0, 0, 0, 0] + [3, 3]             # cure at seq 5, re-default at 9
    far = [0, 3, 0, 0, 0] + [0] * 15 + [3, 3]            # re-default ~16 months after cure
    assert _panel_from_sequence(near, variant="redefault_continuation").new_default.sum() == 1
    assert _panel_from_sequence(far, variant="redefault_continuation").new_default.sum() == 2
    assert _panel_from_sequence(near).new_default.sum() == 2, "baseline counts both"


@pytest.mark.parametrize("probation", [0, 1, 2])
def test_probation_below_regulatory_minimum_is_rejected(probation):
    from hcr.config import _validate
    cfg = {"primary": {"probation_months": probation, "redefault_treatment": "new"}}
    with pytest.raises(ValueError, match="probation"):
        _validate("default_definition", cfg)
