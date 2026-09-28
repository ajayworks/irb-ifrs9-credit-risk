"""Freddie Mac loader, tested against hand-built files in the Release 47 format.

Every test here encodes something from the July 2026 disclosure change that would
silently corrupt results if the loader got it wrong.
"""

from __future__ import annotations

import pytest

from hcr.data import freddie as F
from hcr.default_def import panel as P

ORIG_DEFAULTS = {
    "credit_score": "720", "first_payment_date": "200603", "first_time_buyer": "N",
    "maturity_date": "203602", "msa": "", "mi_pct": "0", "num_units": "1",
    "occupancy_status": "P", "orig_cltv": "80", "dti": "35", "orig_upb": "200000",
    "orig_ltv": "80", "orig_interest_rate": "6.25", "channel": "R",
    "prepayment_penalty": "N", "amortization_type": "FRM", "property_state": "CA",
    "property_type": "SF", "postal_code": "900", "loan_id": "F06Q10000001",
    "loan_purpose": "P", "orig_term_months": "360", "num_borrowers": "2",
    "seller_name": "OTHER", "super_conforming": "N", "pre_harp_loan_id": "",
    "special_eligibility_program": "", "harp_indicator": "N",
    "property_valuation_method": "2", "interest_only": "N", "vantage_score_4": "9999",
}


def orig_line(**kw) -> str:
    v = {**ORIG_DEFAULTS, **kw}
    return "|".join(v[c] for c in F.ORIG_COLUMNS)


def perf_line(**kw) -> str:
    v = {c: "" for c in F.PERF_COLUMNS}
    v.update({"current_upb": "199000", "delinq_status_raw": "00", "remaining_months": "359"})
    v.update(kw)
    return "|".join(v[c] for c in F.PERF_COLUMNS)


def _loan_a_perf(recovery_sign: str = "-") -> list[str]:
    """Defaults at 90 DPD, REO acquisition, then REO disposition with a loss."""
    rows = []
    for age, (period, status) in enumerate(
            [("200604", "00"), ("200605", "01"), ("200606", "02"),
             ("200607", "03"), ("200608", "RA")], start=1):
        rows.append(perf_line(loan_id="F06Q10000001", period_raw=period,
                              delinq_status_raw=status, loan_age=str(age)))
    rows.append(perf_line(
        loan_id="F06Q10000001", period_raw="200609", delinq_status_raw="RA", loan_age="6",
        current_upb="0", zero_balance_code="09", zero_balance_date="200609",
        net_sales_proceeds_raw=f"{recovery_sign}150000",
        mi_recoveries_raw=f"{recovery_sign}12000", non_mi_recoveries_raw="0",
        total_expenses_raw="9500", legal_costs_raw="4500", maintenance_costs_raw="3200",
        taxes_insurance_raw="1800", misc_expenses_raw="0",
        delinquent_accrued_interest_raw="8000", actual_loss_raw="54500"))
    return rows


def _write(tmp_path, orig: list[str], perf: list[str], header: bool = False,
           trailing: bool = False):
    d = tmp_path / "freddie"
    d.mkdir(exist_ok=True)
    suffix = "|" if trailing else ""
    o = [x + suffix for x in orig]
    p = [x + suffix for x in perf]
    if header:
        o.insert(0, "|".join(f"Header {i}" for i in range(31)))
        p.insert(0, "Loan Identifier|" + "|".join(f"H{i}" for i in range(34)))
    (d / "orig_2006Q1.txt").write_text("\n".join(o) + "\n")
    (d / "perf_2006Q1.txt").write_text("\n".join(p) + "\n")
    return d


@pytest.fixture
def standard(tmp_path):
    orig = [
        orig_line(credit_score="9999", dti="999"),                          # sentinels
        orig_line(loan_id="F06Q10000002", credit_score="700", orig_ltv="95"),
        orig_line(loan_id="F07Q30000003", first_payment_date="200710"),
    ]
    perf = _loan_a_perf() + [
        perf_line(loan_id="F06Q10000002", period_raw="200604", loan_age="1"),
        perf_line(loan_id="F06Q10000002", period_raw="200605", loan_age="2",
                  delinq_status_raw="XX"),
        perf_line(loan_id="F06Q10000002", period_raw="200606", loan_age="3",
                  current_upb="0", zero_balance_code="01", zero_balance_date="200606"),
    ] + [
        perf_line(loan_id="F07Q30000003", period_raw=f"20071{m}", loan_age=str(m + 1),
                  zero_balance_code="96" if m == 2 else "",
                  zero_balance_date="200712" if m == 2 else "")
        for m in range(3)
    ]
    d = _write(tmp_path, orig, perf)
    con = P.connect(":memory:")
    out = F.load_freddie(con, raw_dir=d, min_nonzero_for_sign_check=1)
    return con, out


# ---------------------------------------------------------------- typing

def test_volumes(standard):
    con, out = standard
    v = out["quality"]["volumes"].iloc[0]
    assert v.loans == 3
    assert v.loan_months == 12
    assert v.orphan_performance_loans == 0


def test_sentinels_become_null_not_extreme_values(standard):
    """A credit score of 9999 must not land in the best WOE bin."""
    con, _ = standard
    cs, dti, vs = con.execute("SELECT credit_score, dti, vantage_score_4 FROM origination "
                              "WHERE loan_id = 'F06Q10000001'").fetchone()
    assert cs is None and dti is None and vs is None
    assert con.execute("SELECT credit_score FROM origination "
                       "WHERE loan_id = 'F06Q10000002'").fetchone()[0] == 700


def test_orig_quarter_from_loan_id(standard):
    con, _ = standard
    got = dict(con.execute("SELECT loan_id, orig_quarter FROM origination").fetchall())
    assert got["F06Q10000001"] == "2006Q1"
    assert got["F07Q30000003"] == "2007Q3"


def test_period_format_matches_panel(standard):
    con, _ = standard
    assert con.execute("SELECT MIN(period) FROM performance").fetchone()[0] == "2006-04"


# ---------------------------------------------------------------- delinquency

def test_ra_maps_to_bucket_99(standard):
    con, _ = standard
    rows = con.execute("SELECT delinq_bucket, reo_acquisition FROM performance "
                       "WHERE delinq_status_raw = 'RA'").fetchall()
    assert rows and all(b == 99 and reo for b, reo in rows)


def test_xx_is_null_and_counted_not_treated_as_current(standard):
    con, out = standard
    b = con.execute("SELECT delinq_bucket FROM performance WHERE loan_id='F06Q10000002' "
                    "AND loan_age = 2").fetchone()[0]
    assert b is None
    assert out["quality"]["delinquency_status"].iloc[0].not_available_rows == 1


# ---------------------------------------------------------------- signs

def test_recoveries_normalised_to_positive(standard):
    con, _ = standard
    nsp, mi, legal, total = con.execute(
        "SELECT recovery_net_sales_proceeds, recovery_mi, cost_legal, cost_total_expenses "
        "FROM performance WHERE zero_balance_code = '09'").fetchone()
    assert nsp == 150000 and mi == 12000
    assert legal == 4500 and total == 9500


def test_old_sign_convention_is_rejected(tmp_path):
    """Positive raw recoveries = pre-Release-47 data. Loading would invert LGD."""
    d = _write(tmp_path, [orig_line()], _loan_a_perf(recovery_sign=""))
    with pytest.raises(ValueError, match="POSITIVE"):
        F.load_freddie(P.connect(":memory:"), raw_dir=d, min_nonzero_for_sign_check=1)


def test_expenses_reconcile_so_total_is_not_double_counted(standard):
    _, out = standard
    r = out["quality"]["expense_reconciliation"].iloc[0]
    assert r.rows_with_total_and_components == 1
    assert r.rows_not_reconciling == 0


# ---------------------------------------------------------------- format guards

def test_pre_release_47_layout_rejected(tmp_path):
    d = tmp_path / "freddie"
    d.mkdir()
    (d / "orig_2006Q1.txt").write_text(orig_line() + "\n")
    (d / "perf_2006Q1.txt").write_text("|".join(["F06Q10000001"] + ["x"] * 31) + "\n")
    with pytest.raises(ValueError, match="pre-July-2026"):
        F.load_freddie(P.connect(":memory:"), raw_dir=d)


def test_header_rows_are_detected(tmp_path):
    d = _write(tmp_path, [orig_line()], _loan_a_perf(), header=True)
    out = F.load_freddie(P.connect(":memory:"), raw_dir=d, min_nonzero_for_sign_check=1)
    assert out["quality"]["volumes"].iloc[0].loans == 1


def test_trailing_delimiters_are_handled(tmp_path):
    d = _write(tmp_path, [orig_line()], _loan_a_perf(), trailing=True)
    out = F.load_freddie(P.connect(":memory:"), raw_dir=d, min_nonzero_for_sign_check=1)
    assert out["quality"]["volumes"].iloc[0].loan_months == 6


def test_unknown_zero_balance_code_rejected(tmp_path):
    perf = [perf_line(loan_id="F06Q10000001", period_raw="200604", loan_age="1",
                      zero_balance_code="97")]
    d = _write(tmp_path, [orig_line()], perf)
    with pytest.raises(ValueError, match="Unrecognised zero-balance codes"):
        F.load_freddie(P.connect(":memory:"), raw_dir=d)


def test_years_filter_loads_only_requested_vintages(tmp_path):
    d = tmp_path / "freddie"
    for year, lid in [(2006, "F06Q10000001"), (2007, "F07Q10000002")]:
        sub = d / f"sample_{year}"
        sub.mkdir(parents=True)
        (sub / f"sample_orig_{year}.txt").write_text(
            orig_line(loan_id=lid, first_payment_date=f"{year}03") + "\n")
        (sub / f"sample_perf_{year}.txt").write_text(
            perf_line(loan_id=lid, period_raw=f"{year}04", loan_age="1") + "\n")
    con = P.connect(":memory:")
    F.load_freddie(con, raw_dir=d, years=[2007])
    assert con.execute("SELECT loan_id FROM origination").fetchall() == [("F07Q10000002",)]
    with pytest.raises(FileNotFoundError, match="2031"):
        F.load_freddie(P.connect(":memory:"), raw_dir=d, years=[2031])


def test_missing_files_lists_what_is_there(tmp_path):
    d = tmp_path / "freddie"
    d.mkdir()
    (d / "something_else.zip").write_text("x")
    with pytest.raises(FileNotFoundError, match="something_else.zip"):
        F.discover(d)


# ---------------------------------------------------------------- end to end

def test_defaults_through_the_panel(standard):
    """The regression test for the original config error: code 96 is a defect
    repurchase, not a credit event, and must not create a default."""
    con, _ = standard
    P.build_panel(con, "baseline", table_name="p")
    events = dict(con.execute(
        "SELECT loan_id, SUM(new_default::INT) FROM p GROUP BY 1").fetchall())
    assert events["F06Q10000001"] == 1, "90 DPD then REO - one default"
    assert events["F06Q10000002"] == 0, "XX and prepayment - no default"
    assert events["F07Q30000003"] == 0, "code 96 defect repurchase - NOT a default"


def test_default_enters_at_first_90dpd_month(standard):
    con, _ = standard
    P.build_panel(con, "baseline", table_name="p")
    age = con.execute("SELECT loan_age FROM p WHERE loan_id='F06Q10000001' "
                      "AND new_default").fetchone()[0]
    assert age == 4
