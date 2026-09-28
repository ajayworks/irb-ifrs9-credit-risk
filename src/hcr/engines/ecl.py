"""IFRS 9 expected credit loss engine.

    ECL = sum_t  PD_marginal(t) * LGD(t) * EAD(t) * (1 + EIR)^(-t)

Stage 1 sums 12 months; Stage 2 the full expected life; Stage 3 has PD = 1.

Parameters here are POINT-IN-TIME and UNBIASED. Passing regulatory PD or downturn
LGD into this module is an accounting error - it misstates profit. The guard in
`assert_unbiased_inputs` exists to catch that in the pipeline.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from hcr.config import load

_CFG = load("ifrs9")

__all__ = ["Stage", "marginal_pd_from_survival", "survival_from_hazard", "ecl",
           "scenario_weighted_ecl", "assign_stage", "assert_unbiased_inputs"]


class Stage(IntEnum):
    PERFORMING = 1          # 12-month ECL, interest on gross carrying amount
    SICR = 2                # lifetime ECL, interest on gross carrying amount
    CREDIT_IMPAIRED = 3     # lifetime ECL, interest on NET carrying amount


def survival_from_hazard(hazard) -> np.ndarray:
    """S(t) = prod (1 - h(k)). Returns S for t = 1..T (S(0) = 1 is implicit)."""
    return np.cumprod(1.0 - np.asarray(hazard, dtype=float))


def marginal_pd_from_survival(survival) -> np.ndarray:
    """PD_marginal(t) = S(t-1) - S(t) = h(t) * S(t-1).

    NOT the hazard rate. Using h(t) directly overstates ECL, and the error
    compounds through the term. tests/test_ecl.py asserts the sum identity.
    """
    s = np.asarray(survival, dtype=float)
    if s.size and not np.isclose(s[0], 1.0):
        s = np.concatenate([[1.0], s])
    return -np.diff(s)


def ecl(marginal_pd, lgd_path, ead_path, eir: float | None = None,
        periods_per_year: int | None = None) -> float:
    """Discounted expected credit loss over the supplied horizon."""
    eir = _CFG["discounting"]["fallback_eir"] if eir is None else eir
    ppy = periods_per_year or _CFG["discounting"]["periods_per_year"]

    mpd = np.asarray(marginal_pd, dtype=float)
    lgd = np.broadcast_to(np.asarray(lgd_path, dtype=float), mpd.shape)
    ead = np.broadcast_to(np.asarray(ead_path, dtype=float), mpd.shape)

    if np.any(mpd < 0):
        raise ValueError("negative marginal PD - survival curve is not monotonic")
    if mpd.sum() > 1.0 + 1e-9:
        raise ValueError(f"marginal PDs sum to {mpd.sum():.6f} > 1")

    t = np.arange(1, mpd.size + 1) / ppy
    return float(np.sum(mpd * lgd * ead * (1.0 + eir) ** (-t)))


def scenario_weighted_ecl(ecl_by_scenario: dict[str, float],
                          weights: dict[str, float] | None = None) -> dict:
    """Probability-weighted ECL plus the non-linearity uplift versus base-only.

    The uplift is the whole justification for multi-scenario measurement: the loss
    function is convex in the economy, so E[loss] exceeds loss at E[economy].
    """
    if weights is None:
        weights = {k: v["weight"] for k, v in _CFG["scenarios"].items()}
    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-9:
        raise ValueError(f"scenario weights must sum to 1.0, got {total_w}")
    missing = set(weights) - set(ecl_by_scenario)
    if missing:
        raise KeyError(f"missing ECL for scenarios: {sorted(missing)}")

    weighted = sum(ecl_by_scenario[s] * w for s, w in weights.items())
    base = ecl_by_scenario.get("base")
    return {
        "ecl_weighted": weighted,
        "ecl_base_only": base,
        "non_linearity_uplift": None if base is None else weighted - base,
        "non_linearity_pct": None if not base else (weighted - base) / base,
        "contribution": {s: ecl_by_scenario[s] * w for s, w in weights.items()},
    }


def assign_stage(lifetime_pd_now, lifetime_pd_at_origination, dpd_bucket,
                 segment: str = "default_threshold", credit_impaired=None) -> np.ndarray:
    """Stage allocation: relative SICR trigger, absolute backstop, 30 DPD presumption.

    SICR is RELATIVE to origination, not an absolute risk level. A borrower
    originated at 8% and now at 9% has not had a significant increase; one
    originated at 0.2% and now at 1.5% has.
    """
    s = _CFG["sicr"]
    now = np.asarray(lifetime_pd_now, dtype=float)
    orig = np.asarray(lifetime_pd_at_origination, dtype=float)
    dpd = np.asarray(dpd_bucket)

    stage = np.full(now.shape, Stage.PERFORMING, dtype=int)

    if s["relative"]["enabled"]:
        thr = s["relative"]["thresholds_by_segment"].get(
            segment, s["relative"]["thresholds_by_segment"]["default_threshold"])
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(orig > 0, now / orig, np.inf)
        stage = np.where(ratio >= thr, Stage.SICR, stage)

    if s["absolute"]["enabled"]:
        stage = np.where(now >= s["absolute"]["lifetime_pd_threshold"], Stage.SICR, stage)

    if s["dpd_backstop"]["enabled"] and not s["dpd_backstop"]["rebutted"]:
        stage = np.where(dpd >= s["dpd_backstop"]["dpd_bucket_threshold"], Stage.SICR, stage)

    impaired = (np.asarray(credit_impaired, dtype=bool) if credit_impaired is not None
                else dpd >= _CFG["staging"]["credit_impaired_dpd_bucket"])
    stage = np.where(impaired, Stage.CREDIT_IMPAIRED, stage)
    return stage


def assert_unbiased_inputs(pd_values, *, moc_applied: bool, downturn_lgd: bool) -> None:
    """Guard against feeding regulatory parameters into the accounting engine."""
    if moc_applied:
        raise ValueError(
            "Margin of conservatism must not be applied to IFRS 9 PD. "
            "IFRS 9 requires an unbiased, probability-weighted estimate."
        )
    if downturn_lgd:
        raise ValueError(
            "Downturn LGD must not be used for IFRS 9. Use point-in-time, "
            "scenario-conditional LGD."
        )
    pd_values = np.asarray(pd_values, dtype=float)
    if np.any(pd_values < 0) or np.any(pd_values > 1):
        raise ValueError("PD outside [0, 1]")
