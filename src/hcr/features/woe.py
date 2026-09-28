"""Weight of evidence, information value and score scaling.

SIGN CONVENTION - fixed here for the whole project and stated in every document:

    WOE = ln(proportion of goods in bin / proportion of bads in bin)

so HIGHER WOE MEANS LOWER RISK, and WOE decreases as risk increases. Many packages
use ln(bads/goods), which flips every sign. Mixing the two inside one project is a
real and embarrassing bug, so this module refuses to be configured either way.

"good" = non-default, "bad" = default.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["woe_table", "information_value", "iv_band", "check_monotonic",
           "apply_woe", "score_scaling", "score_to_pd", "pd_to_score"]

_EPS = 1e-10


def woe_table(good: np.ndarray, bad: np.ndarray, labels: list | None = None) -> pd.DataFrame:
    """WOE and IV contribution per bin from good/bad counts."""
    good = np.asarray(good, dtype=float)
    bad = np.asarray(bad, dtype=float)
    if good.shape != bad.shape:
        raise ValueError("good and bad must have the same length")
    if good.sum() == 0 or bad.sum() == 0:
        raise ValueError("need at least one good and one bad overall")
    if np.any(bad == 0):
        raise ValueError(
            "a bin has zero bads - WOE is undefined. Merge bins rather than "
            "adding a fudge constant."
        )

    total = good + bad
    dist_good = good / good.sum()
    dist_bad = bad / bad.sum()
    woe = np.log(np.clip(dist_good, _EPS, None) / np.clip(dist_bad, _EPS, None))

    return pd.DataFrame({
        "bin": labels if labels is not None else np.arange(len(good)),
        "good": good.astype(int),
        "bad": bad.astype(int),
        "total": total.astype(int),
        "bad_rate": bad / total,
        "dist_good": dist_good,
        "dist_bad": dist_bad,
        "woe": woe,
        "iv_contribution": (dist_good - dist_bad) * woe,
    })


def information_value(table: pd.DataFrame) -> float:
    return float(table["iv_contribution"].sum())


def iv_band(iv: float) -> str:
    """Conventional interpretation. IV above 0.5 usually means leakage, not skill."""
    if iv < 0.02:
        return "no predictive power - drop"
    if iv < 0.10:
        return "weak"
    if iv < 0.30:
        return "medium"
    if iv < 0.50:
        return "strong"
    return "very strong - CHECK FOR LEAKAGE"


def check_monotonic(table: pd.DataFrame, direction: str = "decreasing") -> dict:
    """Bins should be monotonic in the risk direction unless economically justified.

    Under this module's convention, risk rising across bins means WOE decreasing.
    """
    woe = table["woe"].to_numpy()
    diffs = np.diff(woe)
    ok = bool(np.all(diffs < 0)) if direction == "decreasing" else bool(np.all(diffs > 0))
    return {
        "monotonic": ok,
        "direction": direction,
        "violations": [int(i + 1) for i, d in enumerate(diffs)
                       if (d >= 0 if direction == "decreasing" else d <= 0)],
    }


def apply_woe(values: pd.Series, bin_edges: list[float], table: pd.DataFrame) -> pd.Series:
    """Map raw values onto their bin's WOE. Missing becomes its own bin (last row)."""
    idx = pd.cut(values, bins=bin_edges, labels=False, include_lowest=True)
    woe = table["woe"].to_numpy()
    out = pd.Series(np.where(pd.isna(idx), woe[-1], woe[np.nan_to_num(idx, nan=0).astype(int)]),
                    index=values.index, name=f"{values.name}_woe")
    return out


# --------------------------------------------------------------------------- #
# Score scaling
# --------------------------------------------------------------------------- #

def score_scaling(pdo: float, base_score: float, base_odds: float) -> dict:
    """factor = PDO / ln 2;  offset = base_score - factor * ln(base_odds)."""
    factor = pdo / np.log(2.0)
    return {"factor": factor, "offset": base_score - factor * np.log(base_odds), "pdo": pdo}


def score_to_pd(score, scaling: dict):
    odds = np.exp((np.asarray(score, dtype=float) - scaling["offset"]) / scaling["factor"])
    return 1.0 / (1.0 + odds)


def pd_to_score(pd_value, scaling: dict):
    p = np.clip(np.asarray(pd_value, dtype=float), 1e-12, 1.0 - 1e-12)
    return scaling["offset"] + scaling["factor"] * np.log((1.0 - p) / p)
