"""Vasicek one-factor link between through-the-cycle and point-in-time PD.

    PD_PIT(t) = Phi[ ( Phi^-1(PD_TTC) - sqrt(R) * Z_t ) / sqrt(1-R) ]

Sign convention: Z > 0 is a benign economy, Z < 0 is stress.

Two identities this module guarantees, both asserted in tests:

1. E[PD_PIT] over Z ~ N(0,1) equals PD_TTC. This is what makes the TTC estimate
   genuinely a long-run average.
2. Evaluating at Z = Phi^-1(0.001) reproduces the Basel conditional PD exactly.
   Regulatory capital IS point-in-time PD at a 1-in-1,000 economy, with expected
   loss stripped out.

Note that PD_PIT at Z = 0 is BELOW PD_TTC. The transform is convex, so the mean
sits above the median (Jensen). In a typical year PIT PD is below TTC PD; only in
genuinely bad years does it exceed it.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

__all__ = ["ttc_to_pit", "implied_z", "z_from_default_rates", "mean_revert"]


def ttc_to_pit(pd_ttc, z, correlation):
    """Convert a through-the-cycle PD to point-in-time given economic state z."""
    pd_ttc = np.clip(np.asarray(pd_ttc, dtype=float), 1e-12, 1.0 - 1e-12)
    r = np.asarray(correlation, dtype=float)
    return norm.cdf((norm.ppf(pd_ttc) - np.sqrt(r) * np.asarray(z, float)) / np.sqrt(1.0 - r))


def implied_z(pd_observed, pd_ttc, correlation):
    """Recover the systematic factor from an observed default rate.

    Step one of building a PIT model: back out the historical Z series, then
    regress Z on macro variables so it can be projected under scenarios.
    """
    pd_observed = np.clip(np.asarray(pd_observed, dtype=float), 1e-12, 1.0 - 1e-12)
    pd_ttc = np.clip(np.asarray(pd_ttc, dtype=float), 1e-12, 1.0 - 1e-12)
    r = np.asarray(correlation, dtype=float)
    return (norm.ppf(pd_ttc) - np.sqrt(1.0 - r) * norm.ppf(pd_observed)) / np.sqrt(r)


def z_from_default_rates(default_rates_by_period: dict, pd_ttc: float, correlation: float) -> dict:
    """Vectorised implied_z over a {period: observed_default_rate} mapping."""
    return {
        period: float(implied_z(dr, pd_ttc, correlation))
        for period, dr in default_rates_by_period.items()
    }


def mean_revert(z_start: float, n_periods: int, speed: float, target: float = 0.0) -> np.ndarray:
    """Revert Z towards `target` beyond the reasonable-and-supportable horizon.

    IFRS 9 requires forecasts to be reasonable and supportable. Nobody can forecast
    unemployment in year fifteen of a mortgage, so beyond the explicit horizon the
    systematic factor reverts and the model runs at its through-the-cycle level.
    """
    if not 0.0 <= speed <= 1.0:
        raise ValueError("mean reversion speed must be in [0, 1]")
    out = np.empty(n_periods, dtype=float)
    z = float(z_start)
    for i in range(n_periods):
        z = target + (z - target) * (1.0 - speed)
        out[i] = z
    return out
