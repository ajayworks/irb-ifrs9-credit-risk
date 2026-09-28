"""IRB capital engine, checked against Basel benchmarks and analytic identities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from hcr.engines import irb
from hcr.pd import vasicek


class TestCorrelations:
    def test_other_retail_limits(self):
        assert irb.correlation_other_retail(1e-12) == pytest.approx(0.16, abs=1e-6)
        assert irb.correlation_other_retail(1.0) == pytest.approx(0.03, abs=1e-6)

    def test_other_retail_is_decreasing_in_pd(self):
        pds = np.linspace(1e-6, 0.99, 200)
        assert np.all(np.diff(irb.correlation_other_retail(pds)) < 0)

    def test_sme_size_adjustment_spans_exactly_004(self):
        big = irb.correlation_corporate_sme(0.01, 50)
        small = irb.correlation_corporate_sme(0.01, 5)
        assert big - small == pytest.approx(0.04, abs=1e-12)

    def test_sme_turnover_is_clamped(self):
        assert irb.correlation_corporate_sme(0.01, 1) == pytest.approx(
            irb.correlation_corporate_sme(0.01, 5))
        assert irb.correlation_corporate_sme(0.01, 500) == pytest.approx(
            irb.correlation_corporate_sme(0.01, 50))


class TestMaturityAdjustment:
    def test_normalised_to_one_at_m_equals_one(self):
        """A common implementation bug normalises at M = 2.5 instead of M = 1."""
        for pd_ in [0.001, 0.01, 0.05, 0.2]:
            assert irb.maturity_adjustment(pd_, 1.0) == pytest.approx(1.0, abs=1e-12)

    def test_increases_with_maturity(self):
        assert irb.maturity_adjustment(0.01, 5) > irb.maturity_adjustment(0.01, 2.5)


class TestCapital:
    def test_matches_basel_benchmark_risk_weight(self):
        rw = irb.risk_weight(0.01, 0.25, irb.correlation_residential_mortgage())
        assert rw == pytest.approx(0.3133, abs=5e-4)

    def test_conditional_pd_exceeds_unconditional(self):
        cp = irb.conditional_pd(0.01, 0.15)
        assert cp == pytest.approx(0.110265, abs=1e-5)
        assert cp / 0.01 == pytest.approx(11.03, abs=0.01)

    def test_capital_is_linear_in_lgd(self):
        assert irb.capital_requirement(0.01, 0.50, 0.15) == pytest.approx(
            2 * irb.capital_requirement(0.01, 0.25, 0.15))

    def test_expected_loss_is_subtracted(self):
        """K = stressed loss - EL. Capital covers unexpected loss only."""
        pd_, lgd, r = 0.02, 0.30, 0.15
        stressed = lgd * irb.conditional_pd(pd_, r)
        assert irb.capital_requirement(pd_, lgd, r) == pytest.approx(stressed - pd_ * lgd)

    def test_risk_weight_monotonic_in_pd(self):
        pds = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
        assert np.all(np.diff(irb.risk_weight(pds, 0.25, 0.15)) > 0)

    def test_higher_correlation_means_higher_capital(self):
        assert (irb.capital_requirement(0.01, 0.25, 0.15)
                > irb.capital_requirement(0.01, 0.25, 0.04))

    def test_no_106_scaling_factor(self):
        """Basel 3.1 removed the 1.06 factor. Reinstating it inflates every RWA."""
        manual = 0.25 * irb.conditional_pd(0.01, 0.15) - 0.01 * 0.25
        assert irb.capital_requirement(0.01, 0.25, 0.15) == pytest.approx(manual)

    def test_rw_is_k_over_eight_percent(self):
        k = irb.capital_requirement(0.02, 0.30, 0.15)
        assert irb.risk_weight(0.02, 0.30, 0.15) == pytest.approx(k * 12.5)


class TestFloors:
    def test_pd_floors_match_uk_rules(self):
        assert irb.apply_pd_floor(0.0001, "residential_mortgage") == pytest.approx(0.0010)
        assert irb.apply_pd_floor(0.0001, "qrre_transactor") == pytest.approx(0.0010)
        assert irb.apply_pd_floor(0.0001, "qrre_revolver") == pytest.approx(0.0005)
        assert irb.apply_pd_floor(0.0001, "other_retail") == pytest.approx(0.0005)

    def test_pd_floor_does_not_lower_estimates(self):
        assert irb.apply_pd_floor(0.05, "residential_mortgage") == pytest.approx(0.05)

    def test_unknown_exposure_class_raises(self):
        with pytest.raises(KeyError):
            irb.apply_pd_floor(0.01, "not_a_class")

    def test_lgd_floors(self):
        assert irb.apply_lgd_floor(0.01, "residential_mortgage") == pytest.approx(0.05)
        assert irb.apply_lgd_floor(0.10, "qrre_revolver") == pytest.approx(0.50)
        assert irb.apply_lgd_floor(0.10, "other_retail") == pytest.approx(0.30)

    def test_portfolio_lgd_floor_scales_to_ten_percent(self):
        lgd = np.array([0.05, 0.06, 0.07, 0.08])
        ead = np.array([100.0, 100.0, 100.0, 100.0])
        out = irb.apply_portfolio_lgd_floor(lgd, ead, "residential_mortgage")
        assert out["portfolio_floor_binding"]
        assert out["weighted_lgd_after"] == pytest.approx(0.10)
        assert np.all(out["lgd"] >= lgd)

    def test_portfolio_floor_inactive_when_average_already_above(self):
        lgd = np.array([0.12, 0.18])
        out = irb.apply_portfolio_lgd_floor(lgd, np.array([1.0, 1.0]), "residential_mortgage")
        assert not out["portfolio_floor_binding"]


class TestOutputFloor:
    def test_phase_in_schedule(self):
        expected = {2027: 0.550, 2028: 0.600, 2029: 0.650, 2030: 0.700, 2031: 0.725}
        for year, pct in expected.items():
            assert irb.apply_output_floor(100.0, 250.0, year)["floor_pct"] == pytest.approx(pct)

    def test_floor_binds_for_low_risk_weight_book(self):
        r = irb.apply_output_floor(100.0, 250.0, 2027)
        assert r["floor_binding"]
        assert r["rwa_final"] == pytest.approx(137.5)
        assert r["uplift"] == pytest.approx(37.5)

    def test_irb_governs_when_above_the_floor(self):
        r = irb.apply_output_floor(180.0, 250.0, 2027)
        assert not r["floor_binding"]
        assert r["rwa_final"] == pytest.approx(180.0)

    def test_beyond_schedule_uses_full_floor(self):
        assert irb.apply_output_floor(100.0, 250.0, 2040)["floor_pct"] == pytest.approx(0.725)


class TestVasicekConsistency:
    def test_expectation_over_z_equals_ttc(self):
        """The identity that makes the TTC estimate a genuine long-run average."""
        z = np.random.default_rng(0).standard_normal(2_000_000)
        assert vasicek.ttc_to_pit(0.02, z, 0.15).mean() == pytest.approx(0.02, abs=6e-4)

    def test_capital_formula_is_pit_at_one_in_a_thousand(self):
        """Basel conditional PD == PIT PD evaluated at Z = Phi^-1(0.001)."""
        for pd_, r in [(0.01, 0.15), (0.005, 0.04), (0.03, 0.12)]:
            assert vasicek.ttc_to_pit(pd_, norm.ppf(0.001), r) == pytest.approx(
                irb.conditional_pd(pd_, r), rel=1e-12)

    def test_negative_z_is_stress(self):
        assert vasicek.ttc_to_pit(0.02, -1.0, 0.15) > vasicek.ttc_to_pit(0.02, 1.0, 0.15)

    def test_pit_at_median_economy_is_below_ttc(self):
        """Jensen: the transform is convex, so the mean sits above the median."""
        assert vasicek.ttc_to_pit(0.02, 0.0, 0.15) < 0.02

    def test_implied_z_inverts(self):
        for z in [-2.0, -0.5, 0.0, 1.3]:
            pit = vasicek.ttc_to_pit(0.02, z, 0.15)
            assert vasicek.implied_z(pit, 0.02, 0.15) == pytest.approx(z, abs=1e-9)

    def test_mean_reversion_decays_towards_zero(self):
        path = vasicek.mean_revert(-2.0, 10, speed=0.35)
        assert np.all(np.diff(path) > 0)
        assert abs(path[-1]) < abs(path[0])
        assert path[-1] == pytest.approx(-2.0 * 0.65 ** 10)

    def test_invalid_reversion_speed_raises(self):
        with pytest.raises(ValueError):
            vasicek.mean_revert(-1.0, 5, speed=1.5)
