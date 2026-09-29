"""Scorecard fitting on synthetic data whose true structure is known."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hcr.pd import scorecard as SC

CFG = {
    "binning": {"numeric": {"max_bins": 8, "fine_bins": 20, "min_share": 0.02, "min_bads": 30},
                "categorical": {"min_share": 0.02, "min_bads": 30}},
    "selection": {"min_iv": 0.02, "flag_iv_above": 0.5, "max_abs_correlation": 0.7,
                  "max_p_value": 0.001},
    "scaling": {"pdo": 20, "base_score": 600, "base_odds": 50},
    "master_scale": [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128,
                     0.256, 1.0],
}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(11)
    n = 80_000
    x1 = rng.normal(size=n)                               # strong, risk rises with x1
    x2 = rng.uniform(size=n)                              # moderate, risk falls with x2
    c1 = rng.choice(["A", "B", "C"], size=n, p=[0.5, 0.3, 0.2])
    eff = pd.Series(c1).map({"A": 0.0, "B": 0.5, "C": 1.0}).to_numpy()
    logit = -3.8 + 1.1 * x1 - 0.9 * x2 + eff
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({
        "loan_id": [f"L{i // 4}" for i in range(n)],
        "snapshot_year": 2010 + (np.arange(n) % 4),
        "target": y, "x1": x1, "x2": x2, "c1": c1,
        "x3_noise": rng.normal(size=n),
        "x4_copy": x1 + rng.normal(scale=0.05, size=n),   # near-duplicate of x1
    })


@pytest.fixture(scope="module")
def fitted(data):
    split = SC.assign_split(data, oot_from_year=2013, oos_share=0.3, seed=1)
    dev = data[split == "dev"]
    model = SC.fit_scorecard(dev, ["x1", "x2", "x3_noise", "x4_copy"], ["c1"], CFG)
    return model, data.assign(split=split)


def test_split_is_by_loan_and_by_time(fitted):
    _, df = fitted
    dev_loans = set(df.loc[df.split == "dev", "loan_id"])
    oos_loans = set(df.loc[df.split == "oos", "loan_id"])
    assert not dev_loans & oos_loans, "a loan must never be on both sides"
    assert (df.loc[df.split == "oot", "snapshot_year"] >= 2013).all()
    assert (df.loc[df.split != "oot", "snapshot_year"] < 2013).all()
    assert 0.25 < (df.split == "oos").sum() / (df.split != "oot").sum() < 0.35


def test_selects_true_drivers_and_drops_noise_and_duplicates(fitted):
    model, _ = fitted
    assert {"x2", "c1"} <= set(model.features)
    assert len({"x1", "x4_copy"} & set(model.features)) == 1, "one of the near-duplicates"
    assert "x3_noise" not in model.features
    s = model.selection.set_index("feature").status
    assert s["x3_noise"].startswith("dropped: IV")
    dup = "x4_copy" if "x1" in model.features else "x1"
    assert "corr" in s[dup]


def test_every_coefficient_is_negative(fitted):
    model, _ = fitted
    assert (model.params[model.features] < 0).all()


def test_discriminates_out_of_sample_and_out_of_time(fitted):
    model, df = fitted
    df = df.assign(pd_hat=model.predict_pd(df))
    for part in ("dev", "oos", "oot"):
        g = df[df.split == part]
        assert SC.gini(g.target, g.pd_hat) > 0.45, part


def test_score_is_the_sum_of_points_and_falls_with_risk(fitted):
    model, df = fitted
    rows = df.head(200)
    pts = model.points_table().set_index(["feature", "bin"]).points
    total = np.zeros(len(rows))
    for f in model.features:
        total += np.array([pts[(f, b)] for b in model.binnings[f].bin_label(rows[f])])
    np.testing.assert_allclose(total, model.score(rows), atol=1e-3)
    order = np.argsort(model.predict_pd(rows))
    assert np.all(np.diff(model.score(rows)[order]) <= 1e-9), "higher PD, lower score"


def test_score_scale_anchor(fitted):
    """600 points = 50:1 odds of good, i.e. PD = 1/51."""
    model, _ = fitted
    s = model.scaling
    assert s["offset"] + s["factor"] * np.log(50) == pytest.approx(600)


def test_master_scale_grading(fitted):
    model, _ = fitted
    g = model.grade([0.0001, 0.0005, 0.00051, 0.3, 1.0])
    assert list(g) == [1, 1, 2, 11, 11]


def test_insignificant_drivers_are_dropped_in_regression(data):
    split = SC.assign_split(data, oot_from_year=2013, oos_share=0.3, seed=1)
    strict = {**CFG, "selection": {**CFG["selection"], "max_p_value": 1e-200}}
    model = SC.fit_scorecard(data[split == "dev"], ["x1", "x2"], ["c1"], strict)
    s = model.selection.set_index("feature").status
    assert any(v.startswith("dropped in regression") for v in s)


def test_grade_table_and_concentration():
    t = SC.grade_table([1, 1, 2, 3, 3, 3], [0, 0, 0, 1, 0, 1], [0.1] * 6, [0.1, 0.2, 1.0])
    assert list(t.n) == [2, 1, 3] and t.share.sum() == pytest.approx(1.0)
    assert SC.herfindahl([100] * 5)["effective_grades"] == pytest.approx(5.0)
