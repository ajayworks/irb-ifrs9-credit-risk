"""PD calibration and the TTC -> PIT bridge, tested against known answers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcr.config import load
from hcr.pd import calibration as C
from hcr.pd import vasicek

R = 0.15


def _annual(rows):
    return pd.DataFrame(rows, columns=["year", "grade", "n", "defaults"]).assign(
        dr=lambda d: d.defaults / d.n)


# ------------------------------------------------------------------ long-run average

def test_equal_weight_versus_pooled():
    """Equal weight counts each year once; pooled lets the busy year dominate."""
    a = _annual([(2008, 1, 100, 1), (2009, 1, 900, 27)])           # 1% and 3%
    lra = C.long_run_average(a, "equal").set_index("grade")
    assert lra.loc[1, "lra_equal"] == pytest.approx(0.02)
    assert lra.loc[1, "lra_pooled"] == pytest.approx(28 / 1000)
    assert lra.loc[1, "lra"] == pytest.approx(0.02)


def test_annual_rates_from_loan_level_rows():
    t = C.annual_grade_rates(grade=[1, 1, 2, 2, 2], target=[0, 1, 0, 0, 1],
                             year=[2010] * 5).set_index("grade")
    assert t.loc[1, "dr"] == pytest.approx(0.5) and t.loc[2, "n"] == 3


def test_monotonic_enforcement_pools_violators_by_weight():
    np.testing.assert_allclose(C.enforce_monotonic([0.01, 0.03, 0.02, 0.05], [1, 1, 1, 1]),
                               [0.01, 0.025, 0.025, 0.05])
    np.testing.assert_allclose(C.enforce_monotonic([0.01, 0.03, 0.02], [1, 1, 3]),
                               [0.01, 0.0225, 0.0225])
    ok = [0.001, 0.004, 0.02]
    np.testing.assert_allclose(C.enforce_monotonic(ok, [5, 5, 5]), ok)


# ------------------------------------------------------------------ margin of conservatism

def test_bootstrap_moc_is_zero_without_dispersion_and_grows_with_it():
    flat = _annual([(y, 1, 1000, 10) for y in range(2000, 2020)])
    assert C.bootstrap_moc(flat, 500, 0.75, 1).moc_c.iloc[0] == pytest.approx(0.0)
    rng = np.random.default_rng(3)
    calm = _annual([(y, 1, 1000, int(d)) for y, d in zip(range(2000, 2020), 10 + rng.integers(-1, 2, 20))])
    wild = _annual([(y, 1, 1000, int(d)) for y, d in zip(range(2000, 2020), 10 + rng.integers(-8, 9, 20))])
    m_calm = C.bootstrap_moc(calm, 1000, 0.75, 1).moc_c.iloc[0]
    m_wild = C.bootstrap_moc(wild, 1000, 0.75, 1).moc_c.iloc[0]
    assert 0 <= m_calm < m_wild


def test_bootstrap_is_reproducible():
    a = _annual([(y, g, 1000, y % 7 + g) for y in range(2000, 2015) for g in (1, 2)])
    pd.testing.assert_frame_equal(C.bootstrap_moc(a, 300, 0.75, 9), C.bootstrap_moc(a, 300, 0.75, 9))


def test_moc_config_lists_pending_items_rather_than_hiding_them():
    m = C.moc_add_ons(load("moc"))
    assert set(m["pending"]) == {"A1.1", "A2.1", "B.2"}
    assert m["A"] == 0.0 and m["B"] == 0.0


def test_regulatory_pd_adds_moc_then_floors():
    t = C.regulatory_pd(lra=[0.0002, 0.004], moc_a=0.0, moc_b=0.0, moc_c=[0.0001, 0.0005],
                        floor=0.001)
    assert t.regulatory_pd.tolist() == pytest.approx([0.001, 0.0045])
    assert t.floor_uplift.tolist() == pytest.approx([0.0007, 0.0])
    np.testing.assert_allclose(t.regulatory_pd - t.floor_uplift - t.moc_a - t.moc_b - t.moc_c,
                               t.lra)


# ------------------------------------------------------------------ point in time

@pytest.fixture(scope="module")
def simulated():
    """Twenty years of defaults generated from the Vasicek model with a known Z path."""
    rng = np.random.default_rng(42)
    lra = pd.Series([0.002, 0.01, 0.05], index=[1, 2, 3])
    n = {1: 400_000, 2: 200_000, 3: 50_000}
    z_true = pd.Series(rng.standard_normal(20), index=range(2000, 2020))
    rows = []
    for yr, z in z_true.items():
        for g, p in lra.items():
            pit = float(vasicek.ttc_to_pit(p, z, R))
            rows.append((yr, g, n[g], int(rng.binomial(n[g], pit))))
    return _annual(rows), lra, z_true


def test_implied_z_recovers_the_true_economic_factor(simulated):
    annual, lra, z_true = simulated
    z = C.implied_z(annual, lra, R).set_index("year").z
    assert np.max(np.abs(z - z_true)) < 0.05
    assert np.corrcoef(z, z_true)[0, 1] > 0.999


def test_pit_reproduces_each_years_observed_default_rate(simulated):
    annual, lra, _ = simulated
    z = C.implied_z(annual, lra, R).set_index("year").z
    b = C.bridge_by_year(annual, reg=lra * 1.5, lra=lra, z=z, correlation=R)
    np.testing.assert_allclose(b.pit_pd, b.observed_dr, rtol=1e-9)


def test_stress_years_reverse_the_relationship(simulated):
    """Regulatory PD exceeds PIT in good years and falls below it in bad ones."""
    annual, lra, z_true = simulated
    z = C.implied_z(annual, lra, R).set_index("year").z
    b = C.bridge_by_year(annual, reg=lra * 1.2, lra=lra, z=z, correlation=R).set_index("year")
    assert (b.loc[z_true[z_true > 1].index, "regulatory_over_pit"] > 1).all()
    assert (b.loc[z_true[z_true < -1].index, "regulatory_over_pit"] < 1).all()


def test_pit_correlation_estimate_recovers_the_true_value():
    """Unit-variance Z identifies R; the grade-year likelihood prefers it to a wrong R."""
    rng = np.random.default_rng(8)
    lra = pd.Series([0.003, 0.01, 0.04], index=[1, 2, 3])
    true_r, rows = 0.05, []
    z = rng.standard_normal(300)
    z = (z - z.mean()) / z.std(ddof=1)                    # exactly standard, isolates the estimator
    for yr, zt in zip(range(300), z):
        for g, p in lra.items():
            rows.append((yr, g, 500_000, int(rng.binomial(500_000, float(vasicek.ttc_to_pit(p, zt, true_r))))))
    annual = _annual(rows)
    r_hat = C.estimate_pit_correlation(annual, lra)
    assert r_hat == pytest.approx(true_r, abs=0.004)
    assert C.pit_log_likelihood(annual, lra, r_hat) > C.pit_log_likelihood(annual, lra, 0.15)


def test_correlation_estimator_recovers_r():
    rng = np.random.default_rng(0)
    dr = vasicek.ttc_to_pit(0.02, rng.standard_normal(20_000), 0.15)
    assert C.correlation_from_default_rates(dr) == pytest.approx(0.15, abs=0.005)


# ------------------------------------------------------------------ bridge

def test_waterfall_steps_add_up():
    reg = C.regulatory_pd(lra=[0.0003, 0.004, 0.03], moc_a=0.0, moc_b=0.0,
                          moc_c=[0.0001, 0.0004, 0.002], floor=0.001).assign(grade=[1, 2, 3])
    w = C.waterfall(reg, z=0.8, correlation=R, weights=[50, 30, 20])
    for _, r in w.iterrows():
        assert r.regulatory_pd - r.floor_uplift - r.moc_c - r.moc_b - r.moc_a == pytest.approx(r.lra)
        assert r.lra + r.z_adjustment == pytest.approx(r.pit_pd)
    port = w[w.grade == "portfolio"].iloc[0]
    assert port.regulatory_pd == pytest.approx(np.average(reg.regulatory_pd, weights=[50, 30, 20]))


def test_ifrs9_side_never_sees_conservatism():
    """PIT PD at Z = 0 depends only on the unbiased LRA, whatever the MoC or floor."""
    base = dict(lra=[0.004], moc_a=0.0, moc_b=0.0, floor=0.001)
    lo = C.waterfall(C.regulatory_pd(moc_c=[0.0], **base).assign(grade=[1]), 0.0, R)
    hi = C.waterfall(C.regulatory_pd(moc_c=[0.01], **base).assign(grade=[1]), 0.0, R)
    assert lo.pit_pd.iloc[0] == pytest.approx(hi.pit_pd.iloc[0])
    assert hi.regulatory_pd.iloc[0] > lo.regulatory_pd.iloc[0]
