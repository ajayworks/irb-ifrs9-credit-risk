"""UK Basel 3.1 IRB capital engine.

    K   = LGD * Phi[ (Phi^-1(PD) + sqrt(R) * Phi^-1(0.999)) / sqrt(1-R) ] - PD * LGD
    RWA = K * 12.5 * EAD

The subtracted PD*LGD term is expected loss, removed because provisions cover it.
Retail has no maturity adjustment. The Basel II 1.06 scaling factor is removed.

Verified against Basel benchmark risk weights in tests/test_irb.py.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from hcr.config import load

_CFG = load("uk_basel31")
_Q = _CFG["confidence_level"]
_CORR = _CFG["correlation"]

__all__ = [
    "correlation_residential_mortgage",
    "correlation_qrre",
    "correlation_other_retail",
    "correlation_corporate_sme",
    "maturity_adjustment",
    "capital_requirement",
    "risk_weight",
    "rwa",
    "expected_loss",
    "apply_pd_floor",
    "apply_lgd_floor",
    "apply_portfolio_lgd_floor",
    "apply_output_floor",
]


# --------------------------------------------------------------------------- #
# Asset correlations - prescribed by the regulator, never estimated
# --------------------------------------------------------------------------- #

def correlation_residential_mortgage() -> float:
    return float(_CORR["residential_mortgage"])


def correlation_qrre() -> float:
    return float(_CORR["qrre"])


def _weight(pd: np.ndarray, k: float) -> np.ndarray:
    return (1.0 - np.exp(-k * pd)) / (1.0 - np.exp(-k))


def correlation_other_retail(pd):
    """Supervisory function interpolating from `high` at PD->0 to `low` at PD->1."""
    c = _CORR["other_retail"]
    pd = np.asarray(pd, dtype=float)
    w = _weight(pd, c["k"])
    return c["low"] * w + c["high"] * (1.0 - w)


def correlation_corporate_sme(pd, turnover_m: float):
    """Corporate SME, including the size adjustment (0 at turnover >= cap)."""
    c = _CORR["corporate_sme"]
    pd = np.asarray(pd, dtype=float)
    s = float(np.clip(turnover_m, c["turnover_floor_m"], c["turnover_cap_m"]))
    w = _weight(pd, c["k"])
    span = c["turnover_cap_m"] - c["turnover_floor_m"]
    size_adj = c["size_adjustment"] * (1.0 - (s - c["turnover_floor_m"]) / span)
    return c["low"] * w + c["high"] * (1.0 - w) - size_adj


def maturity_adjustment(pd, maturity_years):
    """Corporate only. Normalised to exactly 1.0 at M = 1 year, NOT at M = 2.5."""
    if _CFG["maturity_adjustment"]["applies_to_retail"]:
        raise RuntimeError("config says maturity adjustment applies to retail - it does not")
    pd = np.clip(np.asarray(pd, dtype=float), 1e-12, 1.0)
    b = (0.11852 - 0.05478 * np.log(pd)) ** 2
    return (1.0 + (maturity_years - 2.5) * b) / (1.0 - 1.5 * b)


# --------------------------------------------------------------------------- #
# Capital
# --------------------------------------------------------------------------- #

def conditional_pd(pd, correlation):
    """Default rate implied by the ASRF model in a 99.9th-percentile bad economy."""
    pd = np.clip(np.asarray(pd, dtype=float), 1e-12, 1.0 - 1e-12)
    r = np.asarray(correlation, dtype=float)
    return norm.cdf((norm.ppf(pd) + np.sqrt(r) * norm.ppf(_Q)) / np.sqrt(1.0 - r))


def capital_requirement(pd, lgd, correlation):
    """K as a fraction of EAD. No maturity adjustment, no 1.06 scaling factor."""
    pd = np.clip(np.asarray(pd, dtype=float), 1e-12, 1.0 - 1e-12)
    lgd = np.asarray(lgd, dtype=float)
    k = lgd * conditional_pd(pd, correlation) - pd * lgd
    return k * _CFG["scaling_factor"]


def risk_weight(pd, lgd, correlation):
    return capital_requirement(pd, lgd, correlation) / _CFG["minimum_capital_ratio"]


def rwa(pd, lgd, ead, correlation):
    return risk_weight(pd, lgd, correlation) * np.asarray(ead, dtype=float)


def expected_loss(pd, lgd, ead):
    """Regulatory EL - uses regulatory PD and DOWNTURN LGD, not the IFRS 9 pair."""
    return np.asarray(pd, float) * np.asarray(lgd, float) * np.asarray(ead, float)


# --------------------------------------------------------------------------- #
# UK parameter floors
# --------------------------------------------------------------------------- #

def apply_pd_floor(pd, exposure_class: str):
    floors = _CFG["pd_floor"]
    if exposure_class not in floors:
        raise KeyError(f"No PD floor configured for '{exposure_class}'. Known: {sorted(floors)}")
    return np.maximum(np.asarray(pd, dtype=float), floors[exposure_class])


def apply_lgd_floor(lgd, exposure_class: str):
    """Account-level LGD floor."""
    key = {
        "residential_mortgage": "residential_mortgage_account",
        "qrre_transactor": "qrre",
        "qrre_revolver": "qrre",
        "other_retail": "other_unsecured_retail",
    }.get(exposure_class)
    if key is None:
        raise KeyError(f"No LGD floor mapping for '{exposure_class}'")
    return np.maximum(np.asarray(lgd, dtype=float), _CFG["lgd_floor"][key])


def apply_portfolio_lgd_floor(lgd, ead, exposure_class: str) -> dict:
    """Exposure-weighted portfolio LGD floor (residential mortgages, 10%).

    Applies AFTER the account-level floor. If the weighted average still falls
    below the portfolio floor, every account is scaled up proportionally so the
    average meets it.
    """
    lgd = np.asarray(lgd, dtype=float)
    ead = np.asarray(ead, dtype=float)
    if exposure_class != "residential_mortgage":
        return {"lgd": lgd, "portfolio_floor_binding": False, "weighted_lgd": float(
            np.average(lgd, weights=ead))}

    floor = _CFG["lgd_floor"]["residential_mortgage_portfolio"]
    weighted = float(np.average(lgd, weights=ead))
    if weighted >= floor:
        return {"lgd": lgd, "portfolio_floor_binding": False, "weighted_lgd": weighted}
    scaled = lgd * (floor / weighted)
    return {
        "lgd": scaled,
        "portfolio_floor_binding": True,
        "weighted_lgd": weighted,
        "weighted_lgd_after": float(np.average(scaled, weights=ead)),
    }


def apply_output_floor(rwa_irb_total: float, rwa_sa_total: float, year: int) -> dict:
    """RWA_final = max(RWA_IRB, floor% * RWA_SA), with the PS1/26 phase-in (60% in 2027
    rising to 72.5% from 2030)."""
    schedule = _CFG["output_floor"]
    floor = schedule.get(year, schedule[max(schedule)])
    floored = floor * rwa_sa_total
    return {
        "year": year,
        "floor_pct": floor,
        "rwa_irb": rwa_irb_total,
        "rwa_sa": rwa_sa_total,
        "rwa_floored": floored,
        "rwa_final": max(rwa_irb_total, floored),
        "floor_binding": floored > rwa_irb_total,
        "uplift": max(0.0, floored - rwa_irb_total),
        "uplift_pct": max(0.0, floored - rwa_irb_total) / rwa_irb_total
        if rwa_irb_total else float("nan"),
    }
