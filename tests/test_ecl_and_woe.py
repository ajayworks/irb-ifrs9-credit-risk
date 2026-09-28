"""IFRS 9 ECL engine and WOE/scorecard module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcr.engines import ecl as E
from hcr.features import woe as W


class TestSurvivalIdentities:
    def test_marginal_pds_sum_to_cumulative_default_probability(self):
        h = np.array([0.008, 0.012, 0.010, 0.007, 0.005])
        s = E.survival_from_hazard(h)
        m = E.marginal_pd_from_survival(s)
        assert m.sum() == pytest.approx(1.0 - s[-1])

    def test_marginal_pd_is_not_the_hazard_rate(self):
        """PD_marginal(t) = h(t) * S(t-1). Using h(t) directly overstates ECL."""
        h = np.array([0.008, 0.012, 0.010])
        s = E.survival_from_hazard(h)
        m = E.marginal_pd_from_survival(s)
        assert m[0] == pytest.approx(h[0])
        assert m[1] == pytest.approx(h[1] * s[0])
        assert m[1] < h[1]

    def test_marginal_pds_are_non_increasing_in_error_versus_hazard(self):
        h = np.full(10, 0.02)
        m = E.marginal_pd_from_survival(E.survival_from_hazard(h))
        assert np.all(np.diff(m) < 0), "constant hazard gives declining marginal PD"


class TestECL:
    def _fixture(self):
        h = np.array([0.008, 0.012, 0.010, 0.007, 0.005])
        m = E.marginal_pd_from_survival(E.survival_from_hazard(h))
        lgd = np.array([0.22, 0.23, 0.24, 0.24, 0.25])
        ead = np.array([98_000.0, 95_500, 92_800, 89_900, 86_700])
        return m, lgd, ead

    def test_twelve_month_and_lifetime(self):
        m, lgd, ead = self._fixture()
        twelve = E.ecl(m[:1], lgd[:1], ead[:1], eir=0.045, periods_per_year=1)
        lifetime = E.ecl(m, lgd, ead, eir=0.045, periods_per_year=1)
        assert twelve == pytest.approx(165.05, abs=0.01)
        assert lifetime == pytest.approx(802.45, abs=0.01)
        assert lifetime / twelve == pytest.approx(4.86, abs=0.01)

    def test_discounting_reduces_ecl(self):
        m, lgd, ead = self._fixture()
        assert (E.ecl(m, lgd, ead, eir=0.10, periods_per_year=1)
                < E.ecl(m, lgd, ead, eir=0.0, periods_per_year=1))

    def test_negative_marginal_pd_rejected(self):
        with pytest.raises(ValueError, match="negative marginal PD"):
            E.ecl(np.array([0.01, -0.002]), 0.2, 1000.0)

    def test_marginal_pds_above_one_rejected(self):
        with pytest.raises(ValueError, match="sum"):
            E.ecl(np.array([0.7, 0.5]), 0.2, 1000.0)


class TestScenarios:
    SC = {"upside": 142.0, "base": 436.0, "downside": 1180.0, "severe": 2650.0}
    W_ = {"upside": 0.20, "base": 0.50, "downside": 0.20, "severe": 0.10}

    def test_weighted_ecl_and_non_linearity(self):
        out = E.scenario_weighted_ecl(self.SC, self.W_)
        assert out["ecl_weighted"] == pytest.approx(747.40, abs=0.01)
        assert out["non_linearity_uplift"] == pytest.approx(311.40, abs=0.01)
        assert out["non_linearity_pct"] == pytest.approx(0.714, abs=1e-3)

    def test_weighted_exceeds_base_when_losses_are_convex(self):
        out = E.scenario_weighted_ecl(self.SC, self.W_)
        assert out["ecl_weighted"] > out["ecl_base_only"]

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            E.scenario_weighted_ecl(self.SC, {**self.W_, "base": 0.9})

    def test_missing_scenario_raises(self):
        with pytest.raises(KeyError):
            E.scenario_weighted_ecl({"base": 100.0}, self.W_)

    def test_config_weights_are_used_by_default(self):
        out = E.scenario_weighted_ecl(self.SC)
        assert out["ecl_weighted"] == pytest.approx(747.40, abs=0.01)


class TestStaging:
    def test_sicr_is_relative_not_absolute(self):
        """0.3% -> 1.8% is a significant increase; 7% -> 8% is not."""
        stage = E.assign_stage(
            lifetime_pd_now=np.array([0.018, 0.08]),
            lifetime_pd_at_origination=np.array([0.003, 0.07]),
            dpd_bucket=np.array([0, 0]))
        assert stage[0] == E.Stage.SICR
        assert stage[1] == E.Stage.PERFORMING

    def test_absolute_backstop_catches_high_risk_at_origination(self):
        stage = E.assign_stage(np.array([0.30]), np.array([0.28]), np.array([0]))
        assert stage[0] == E.Stage.SICR

    def test_thirty_dpd_presumption(self):
        stage = E.assign_stage(np.array([0.01]), np.array([0.01]), np.array([1]))
        assert stage[0] == E.Stage.SICR

    def test_ninety_dpd_is_stage_three(self):
        stage = E.assign_stage(np.array([0.01]), np.array([0.01]), np.array([3]))
        assert stage[0] == E.Stage.CREDIT_IMPAIRED

    def test_stage_three_dominates(self):
        stage = E.assign_stage(np.array([0.5]), np.array([0.001]), np.array([4]))
        assert stage[0] == E.Stage.CREDIT_IMPAIRED


class TestUnbiasedGuard:
    def test_moc_rejected(self):
        with pytest.raises(ValueError, match="Margin of conservatism"):
            E.assert_unbiased_inputs([0.01], moc_applied=True, downturn_lgd=False)

    def test_downturn_lgd_rejected(self):
        with pytest.raises(ValueError, match="Downturn LGD"):
            E.assert_unbiased_inputs([0.01], moc_applied=False, downturn_lgd=True)

    def test_clean_inputs_pass(self):
        E.assert_unbiased_inputs([0.01, 0.02], moc_applied=False, downturn_lgd=False)


class TestWOE:
    GOOD = np.array([2400, 3100, 2600, 1500, 400])
    BAD = np.array([24, 50, 78, 90, 58])

    def test_worked_example_values(self):
        t = W.woe_table(self.GOOD, self.BAD,
                        labels=["<=60", "60-75", "75-85", "85-95", ">95"])
        np.testing.assert_allclose(
            t.woe, [1.0986, 0.6206, 0.0, -0.6931, -1.5755], atol=5e-4)
        assert W.information_value(t) == pytest.approx(0.6103, abs=5e-4)

    def test_woe_is_zero_when_bin_matches_portfolio_average(self):
        t = W.woe_table(self.GOOD, self.BAD)
        overall = self.BAD.sum() / (self.GOOD + self.BAD).sum()
        idx = int(np.argmin(np.abs(t.bad_rate - overall)))
        assert t.woe.iloc[idx] == pytest.approx(0.0, abs=1e-6)

    def test_sign_convention_higher_woe_is_lower_risk(self):
        t = W.woe_table(self.GOOD, self.BAD)
        assert t.loc[t.woe.idxmax(), "bad_rate"] == t.bad_rate.min()

    def test_monotonicity_check_detects_violation(self):
        t = W.woe_table(np.array([100, 500, 100]), np.array([10, 5, 20]))
        assert not W.check_monotonic(t)["monotonic"]

    def test_monotonicity_check_passes_on_clean_bins(self):
        t = W.woe_table(self.GOOD, self.BAD)
        assert W.check_monotonic(t)["monotonic"]

    def test_zero_bads_raises_rather_than_fudging(self):
        with pytest.raises(ValueError, match="zero bads"):
            W.woe_table(np.array([100, 200]), np.array([0, 10]))

    def test_iv_band_flags_probable_leakage(self):
        assert "LEAKAGE" in W.iv_band(0.85)
        assert W.iv_band(0.005) == "no predictive power - drop"


class TestScoreScaling:
    def test_pdo_doubles_the_odds(self):
        s = W.score_scaling(pdo=20, base_score=600, base_odds=50)
        assert s["factor"] == pytest.approx(28.8539, abs=1e-4)
        assert s["offset"] == pytest.approx(487.1229, abs=1e-4)
        p600, p620 = W.score_to_pd(600, s), W.score_to_pd(620, s)
        odds = lambda p: (1 - p) / p
        assert odds(p620) / odds(p600) == pytest.approx(2.0)

    def test_known_points(self):
        s = W.score_scaling(20, 600, 50)
        assert W.score_to_pd(600, s) == pytest.approx(1 / 51, abs=1e-9)
        assert W.score_to_pd(620, s) == pytest.approx(1 / 101, abs=1e-9)

    def test_round_trip(self):
        s = W.score_scaling(30, 500, 20)
        for p in [0.001, 0.01, 0.1, 0.4]:
            assert W.score_to_pd(W.pd_to_score(p, s), s) == pytest.approx(p)

    def test_book_answer_80_to_1_is_560(self):
        s = W.score_scaling(pdo=30, base_score=500, base_odds=20)
        assert W.pd_to_score(1 / 81, s) == pytest.approx(560.0, abs=1e-6)
