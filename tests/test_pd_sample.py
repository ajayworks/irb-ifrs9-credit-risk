"""PD development sample, built end to end from Release 47 fixture files.

Three loans over 2006-03 .. 2008-12 (the last data month):
  A  30 and 60 days late in spring 2007, cures, then goes 90+ DPD in August 2008
  B  always current
  C  prepays in December 2007
"""

from __future__ import annotations

import pytest

from hcr.data import freddie as F
from hcr.default_def import panel as P
from hcr.pd import sample as S
from test_freddie_loader import orig_line, perf_line


def _months(start_year: int, start_month: int, n: int) -> list[str]:
    out, y, m = [], start_year, start_month
    for _ in range(n):
        out.append(f"{y}{m:02d}")
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


@pytest.fixture(scope="module")
def snapshots(tmp_path_factory):
    d = tmp_path_factory.mktemp("freddie")
    periods = _months(2006, 3, 34)                            # 2006-03 .. 2008-12
    status_a = {"200703": "01", "200704": "02", "200806": "01", "200807": "02",
                "200808": "03", "200809": "04", "200810": "05", "200811": "06",
                "200812": "07"}
    orig = [orig_line(loan_id=f"F06Q1000000{i}", orig_upb="200000") for i in (1, 2, 3)]
    perf = []
    for age, p in enumerate(periods):
        perf.append(perf_line(loan_id="F06Q10000001", period_raw=p, loan_age=str(age),
                              delinq_status_raw=status_a.get(p, "00")))
        perf.append(perf_line(loan_id="F06Q10000002", period_raw=p, loan_age=str(age),
                              current_upb=str(200000 - 1000 * age),
                              estimated_ltv_raw="999" if p == "200612" else "72"))
        if p <= "200712":
            perf.append(perf_line(
                loan_id="F06Q10000003", period_raw=p, loan_age=str(age),
                current_upb="0" if p == "200712" else "150000",
                zero_balance_code="01" if p == "200712" else "",
                zero_balance_date="200712" if p == "200712" else ""))
    (d / "orig_2006Q1.txt").write_text("\n".join(orig) + "\n")
    (d / "perf_2006Q1.txt").write_text("\n".join(perf) + "\n")

    con = P.connect(":memory:")
    F.load_freddie(con, raw_dir=d)
    P.build_panel(con, "baseline")
    S.build_snapshots(con)
    return con.execute("SELECT * FROM pd_snapshots ORDER BY loan_id, snapshot_period").df()


def _row(df, loan, period):
    r = df[(df.loan_id == loan) & (df.snapshot_period == period)]
    assert len(r) == 1, f"expected one snapshot for {loan} {period}, got {len(r)}"
    return r.iloc[0]


def test_one_december_row_per_loan_per_eligible_year(snapshots):
    got = list(zip(snapshots.loan_id, snapshots.snapshot_period))
    assert got == [("F06Q10000001", "2006-12"), ("F06Q10000001", "2007-12"),
                   ("F06Q10000002", "2006-12"), ("F06Q10000002", "2007-12"),
                   ("F06Q10000003", "2006-12")]


def test_excludes_in_default_exit_and_incomplete_windows(snapshots):
    assert not ((snapshots.loan_id == "F06Q10000001")
                & (snapshots.snapshot_period == "2008-12")).any(), "in default"
    assert not ((snapshots.loan_id == "F06Q10000003")
                & (snapshots.snapshot_period == "2007-12")).any(), "exit month"
    assert not (snapshots.snapshot_period == "2008-12").any(), "window runs past data end"


def test_target_is_default_in_the_next_twelve_months(snapshots):
    assert _row(snapshots, "F06Q10000001", "2006-12").target == 0
    assert _row(snapshots, "F06Q10000001", "2007-12").target == 1, "defaults Aug 2008"
    assert snapshots[snapshots.loan_id != "F06Q10000001"].target.eq(0).all()


def test_delinquency_history_uses_only_the_trailing_twelve_months(snapshots):
    early = _row(snapshots, "F06Q10000001", "2006-12")
    late = _row(snapshots, "F06Q10000001", "2007-12")
    assert early.max_delinq_12m == 0 and early.months_delinq_12m == 0
    assert late.max_delinq_12m == 2, "60 days late in April 2007"
    assert late.months_delinq_12m == 2
    assert late.delinq_bucket == 0, "current at the snapshot itself"


def test_features_are_known_at_the_snapshot_not_later(snapshots):
    """Loan A's 2008 delinquency must not leak into its 2007-12 features."""
    late = _row(snapshots, "F06Q10000001", "2007-12")
    assert late.max_delinq_12m < 3 and late.prior_default == 0


def test_seasoning_sentinels_and_amortisation(snapshots):
    b06, b07 = _row(snapshots, "F06Q10000002", "2006-12"), _row(snapshots, "F06Q10000002", "2007-12")
    assert b06.months_on_book == 9 and b07.months_on_book == 21
    assert b06.eltv != b06.eltv, "ELTV 999 is 'not available' -> missing, not 999"
    assert b07.eltv == 72
    assert b07.balance_ratio == pytest.approx((200000 - 1000 * 21) / 200000)


def test_snapshot_month_is_validated():
    with pytest.raises(ValueError):
        S.build_snapshots(P.connect(":memory:"), snapshot_month=13)
