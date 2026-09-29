"""Monotonic WOE binning."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from hcr.features.binning import MISSING, OTHER, bin_categorical, bin_numeric

RNG = np.random.default_rng(7)


def _logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


@pytest.fixture(scope="module")
def data():
    n = 60_000
    x = RNG.uniform(0, 1, n)
    y = (RNG.uniform(size=n) < _logistic(-4.5 + 3.5 * x)).astype(int)
    return x, y


def _bad_rates(b):
    t = b.table[b.table.bin != MISSING]
    return t.bad_rate.to_numpy()


def test_detects_direction_and_is_monotonic(data):
    x, y = data
    b = bin_numeric(x, y, min_share=0.02, min_bads=50)
    assert b.direction == "increasing"
    assert np.all(np.diff(_bad_rates(b)) > 0)
    woe = b.table[b.table.bin != MISSING].woe.to_numpy()
    assert np.all(np.diff(woe) < 0), "WOE = ln(good/bad) falls as risk rises"
    assert b.iv > 0.1


def test_forced_direction_is_monotonic_even_on_u_shaped_data():
    x = RNG.uniform(-1, 1, 40_000)
    y = (RNG.uniform(size=x.size) < _logistic(-3 + 2.5 * x ** 2)).astype(int)
    b = bin_numeric(x, y, direction="increasing", min_share=0.02, min_bads=30)
    assert np.all(np.diff(_bad_rates(b)) >= 0)


def test_size_default_and_count_limits(data):
    x, y = data
    b = bin_numeric(x, y, max_bins=4, min_share=0.10, min_bads=200)
    t = b.table[b.table.bin != MISSING]
    assert len(t) <= 4
    assert (t.share >= 0.10).all()
    assert (t.bads >= 200).all()


def test_small_but_predictive_flag_survives_with_large_sample_thresholds():
    """A 1% flag with a 30% default rate must not be merged away."""
    n = 200_000
    flag = (RNG.uniform(size=n) < 0.01).astype(int)
    y = (RNG.uniform(size=n) < np.where(flag == 1, 0.30, 0.01)).astype(int)
    kept = bin_numeric(flag, y, min_share=0.005, min_bads=100)
    assert len(kept.table) == 2 and kept.iv > 0.3
    textbook = bin_numeric(flag, y, min_share=0.05, min_bads=100)
    assert len(textbook.table) == 1 and textbook.iv == pytest.approx(0.0), \
        "the 5% rule merges the flag away - why the project config lowers it"


def test_material_missing_gets_its_own_bin(data):
    x, y = data
    x = x.copy()
    x[:6000] = np.nan
    b = bin_numeric(x, y, min_share=0.02, min_bads=50)
    assert b.missing_policy == "own_bin"
    assert MISSING in set(b.table.bin)
    assert b.transform([np.nan])[0] == pytest.approx(b.missing_woe)


def test_tiny_missing_is_pooled_with_the_riskiest_bin(data):
    x, y = data
    x = x.copy()
    x[:20] = np.nan
    b = bin_numeric(x, y, min_share=0.02, min_bads=50)
    assert b.missing_policy == "pooled_with_riskiest"
    assert MISSING not in set(b.table.bin)
    assert b.missing_woe == pytest.approx(b.table.woe.min())


def test_transform_reproduces_the_table(data):
    x, y = data
    b = bin_numeric(x, y, min_share=0.02, min_bads=50)
    w = b.transform(x)
    counts = pd.Series(w).round(10).value_counts()
    expected = b.table.set_index(b.table.woe.round(10)).n
    for woe_value, n in counts.items():
        assert expected.loc[woe_value] == n


def test_single_woe_variable_gives_coefficient_minus_one(data):
    """The identity from primer section 3.6, on real binning output."""
    x, y = data
    w = bin_numeric(x, y, min_share=0.02, min_bads=50).transform(x)
    fit = sm.Logit(y, sm.add_constant(w)).fit(disp=0)
    assert fit.params[1] == pytest.approx(-1.0, abs=1e-6)
    assert fit.params[0] == pytest.approx(np.log(y.sum() / (len(y) - y.sum())), abs=1e-6)


def test_categorical_pools_rare_levels_and_handles_unseen():
    n = 50_000
    lvl = RNG.choice(["P", "C", "N", "R"], size=n, p=[0.6, 0.25, 0.145, 0.005])
    base = {"P": 0.02, "C": 0.05, "N": 0.03, "R": 0.04}
    y = (RNG.uniform(size=n) < np.vectorize(base.get)(lvl)).astype(int)
    b = bin_categorical(lvl, y, min_share=0.02, min_bads=50)
    assert b.groups["R"] != "R", "0.5% level pooled away"
    assert set(b.table.bin) <= {"P", "C", "N", OTHER}
    assert b.transform(["Z"])[0] == pytest.approx(b.fallback_woe), "unseen level"
    assert b.transform(["C"])[0] < b.transform(["P"])[0], "riskier level, lower WOE"


def test_rejects_degenerate_target():
    with pytest.raises(ValueError):
        bin_numeric([1, 2, 3], [0, 0, 0])
