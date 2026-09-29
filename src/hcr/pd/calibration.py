"""PD calibration: long-run average by grade, margin of conservatism, UK floor, and the
Vasicek bridge from the regulatory PD to point-in-time (IFRS 9) PD.

Regulatory (IRB) PD for grade g
    LRA_g    long-run average of the grade's one-year default rates over the whole
             observation period (equal weight per year by default)
    + MoC_g  margin of conservatism: category C by bootstrapping years; A and B from the
             assessed items in config/moc.yaml
    floor    UK PD input floor, e.g. 0.10% for residential mortgages

Point-in-time (IFRS 9) PD for grade g in year t
    PD_PIT = Phi[(Phi^-1(LRA_g) - sqrt(R) * Z_t) / sqrt(1 - R)]
    Z_t is the systematic factor that makes the grade PDs reproduce year t's observed
    defaults. The input is the UNBIASED LRA - no MoC, no floor - because IFRS 9 requires
    an unbiased estimate.

So both PDs come from the same grades. The regulatory one adds conservatism and a floor;
the accounting one moves the unbiased average with the economy. The bridge between them
is exactly those steps, and every one is reported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from hcr.pd import vasicek

__all__ = ["annual_grade_rates", "long_run_average", "enforce_monotonic", "bootstrap_moc",
           "moc_add_ons", "regulatory_pd", "implied_z", "pit_table", "bridge_by_year",
           "waterfall", "estimate_pit_correlation", "pit_log_likelihood",
           "correlation_from_default_rates"]


# --------------------------------------------------------------------------- #
# long-run average
# --------------------------------------------------------------------------- #

def annual_grade_rates(grade, target, year) -> pd.DataFrame:
    """One-year default rate per (year, grade)."""
    df = pd.DataFrame({"year": np.asarray(year), "grade": np.asarray(grade),
                       "y": np.asarray(target, dtype=float)})
    out = df.groupby(["year", "grade"]).y.agg(n="size", defaults="sum").reset_index()
    out["dr"] = out.defaults / out.n
    return out


def long_run_average(annual: pd.DataFrame, weighting: str = "equal") -> pd.DataFrame:
    """Per grade. 'equal': mean of annual default rates (each year one observation of the
    cycle). 'pooled': total defaults / total observations."""
    g = annual.groupby("grade")
    out = pd.DataFrame({"years": g.year.nunique(), "n": g.n.sum(), "defaults": g.defaults.sum(),
                        "lra_equal": g.dr.mean()})
    out["lra_pooled"] = out.defaults / out.n
    if weighting not in {"equal", "pooled"}:
        raise ValueError("weighting must be 'equal' or 'pooled'")
    out["lra"] = out[f"lra_{weighting}"]
    return out.reset_index()


def enforce_monotonic(values, weights) -> np.ndarray:
    """Pool adjacent violators: the closest non-decreasing sequence, weighted."""
    v = [float(x) for x in values]
    w = [float(x) for x in weights]
    blocks = [[v[i], w[i], 1] for i in range(len(v))]          # value, weight, size
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] > blocks[i + 1][0]:
            a, b = blocks[i], blocks.pop(i + 1)
            tw = a[1] + b[1]
            blocks[i] = [(a[0] * a[1] + b[0] * b[1]) / tw, tw, a[2] + b[2]]
            i = max(i - 1, 0)
        else:
            i += 1
    return np.concatenate([[b[0]] * b[2] for b in blocks])


# --------------------------------------------------------------------------- #
# margin of conservatism
# --------------------------------------------------------------------------- #

def bootstrap_moc(annual: pd.DataFrame, n_resamples: int = 2000, percentile: float = 0.75,
                  seed: int = 0) -> pd.DataFrame:
    """Category C: resample whole YEARS with replacement, recompute each grade's
    equal-weight LRA, and take add-on = percentile - estimate (never negative).

    Resampling years rather than loans keeps the correlation within a year - defaults
    cluster by year, which is exactly the uncertainty a long-run average carries.
    """
    dr = annual.pivot(index="year", columns="grade", values="dr")
    mat = dr.to_numpy()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, mat.shape[0], size=(n_resamples, mat.shape[0]))
    with np.errstate(invalid="ignore"):
        boot = np.nanmean(mat[idx], axis=1)                      # resamples x grades
    est = np.nanmean(mat, axis=0)
    upper = np.nanquantile(boot, percentile, axis=0)
    return pd.DataFrame({"grade": dr.columns, "lra_equal": est,
                         f"p{int(round(percentile * 100))}": upper,
                         "moc_c": np.maximum(upper - est, 0.0)})


def moc_add_ons(moc_cfg: dict) -> dict:
    """Categories A and B: sum the PD add-ons of assessed items; list pending items."""
    out = {"A": 0.0, "B": 0.0, "pending": [], "items": []}
    for cat_id, cat in moc_cfg["categories"].items():
        group = "A" if cat_id.startswith("A") else "B" if cat_id.startswith("B") else None
        for item in cat["items"]:
            if item.get("applies_to") != "pd" or group is None:
                continue
            add = item.get("add_on_pd")
            out["items"].append({"id": item["id"], "status": item["status"],
                                 "add_on_pd": add if isinstance(add, (int, float)) else 0.0})
            if item["status"] == "pending" or add is None:
                out["pending"].append(item["id"])
            elif isinstance(add, (int, float)):
                out[group] += float(add)
    return out


def regulatory_pd(lra, moc_a, moc_b, moc_c, floor: float) -> pd.DataFrame:
    """Components of the regulatory PD. Regulatory = LRA + MoC, then the floor."""
    lra = np.asarray(lra, float)
    moc_a, moc_b, moc_c = (np.broadcast_to(np.asarray(m, float), lra.shape)
                           for m in (moc_a, moc_b, moc_c))
    pre = lra + moc_a + moc_b + moc_c
    reg = np.maximum(pre, floor)
    return pd.DataFrame({"lra": lra, "moc_a": moc_a, "moc_b": moc_b, "moc_c": moc_c,
                         "pd_before_floor": pre, "floor_uplift": reg - pre,
                         "regulatory_pd": reg})


# --------------------------------------------------------------------------- #
# point-in-time
# --------------------------------------------------------------------------- #

def implied_z(annual: pd.DataFrame, lra: pd.Series, correlation: float) -> pd.DataFrame:
    """Per year: the Z at which the Vasicek-adjusted grade PDs reproduce the observed
    number of defaults. Positive Z = benign, negative = stress."""
    rows = []
    for year, g in annual.groupby("year"):
        base = lra.reindex(g.grade).to_numpy()
        n, d = g.n.to_numpy(float), g.defaults.sum()

        def gap(z, base=base, n=n, d=d):
            return float(np.sum(n * vasicek.ttc_to_pit(base, z, correlation)) - d)

        # gap falls monotonically from (N - D) at Z = -inf to -D at Z = +inf, so a root
        # exists whenever 0 < D < N. With a small correlation it can sit at a large |Z|
        # (Z moves PD by only sqrt(R) per unit), so widen the bracket until it is found.
        if d <= 0:
            z = np.inf
        elif d >= n.sum():
            z = -np.inf
        else:
            for bound in (12.0, 50.0, 200.0, 1000.0):
                if gap(-bound) > 0 > gap(bound):
                    z = brentq(gap, -bound, bound)
                    break
            else:
                raise ValueError(f"No Z found for {year} at correlation {correlation}")
        rows.append({"year": year, "n": int(n.sum()), "defaults": int(d),
                     "observed_dr": d / n.sum(),
                     "ttc_dr": float(np.sum(n * base) / n.sum()), "z": z})
    return pd.DataFrame(rows)


def pit_table(lra: pd.Series, z: pd.Series, correlation: float) -> pd.DataFrame:
    """Grade x year PIT PDs."""
    return pd.DataFrame({yr: vasicek.ttc_to_pit(lra.to_numpy(), zt, correlation)
                         for yr, zt in z.items()}, index=lra.index)


def bridge_by_year(annual: pd.DataFrame, reg: pd.Series, lra: pd.Series,
                   z: pd.Series, correlation: float) -> pd.DataFrame:
    """Portfolio regulatory PD vs PIT PD each year, both weighted by that year's grade mix.
    PIT reproduces the observed default rate by construction of Z."""
    rows = []
    for year, g in annual.groupby("year"):
        w = g.n.to_numpy(float) / g.n.sum()
        grades = g.grade.to_numpy()
        pit = vasicek.ttc_to_pit(lra.reindex(grades).to_numpy(), z[year], correlation)
        rows.append({"year": year, "regulatory_pd": float(np.sum(w * reg.reindex(grades))),
                     "ttc_unbiased_pd": float(np.sum(w * lra.reindex(grades))),
                     "pit_pd": float(np.sum(w * pit)), "observed_dr": g.defaults.sum() / g.n.sum()})
    out = pd.DataFrame(rows)
    out["regulatory_over_pit"] = out.regulatory_pd / out.pit_pd
    return out


def waterfall(reg_table: pd.DataFrame, z: float, correlation: float,
              weights=None) -> pd.DataFrame:
    """Regulatory PD -> IFRS 9 PD, step by step, per grade and for the portfolio.
    Steps: remove the floor uplift, remove MoC C, B, A (reaching the unbiased LRA),
    then apply the Vasicek adjustment for the year's economy."""
    t = reg_table.copy()
    t["pit_pd"] = vasicek.ttc_to_pit(t.lra.to_numpy(), z, correlation)
    t["z_adjustment"] = t.pit_pd - t.lra
    steps = ["regulatory_pd", "floor_uplift", "moc_c", "moc_b", "moc_a", "lra",
             "z_adjustment", "pit_pd"]
    if weights is not None:
        w = np.asarray(weights, float) / np.sum(weights)
        port = {c: float(np.sum(w * t[c])) for c in steps}
        t = pd.concat([t, pd.DataFrame([{**port, "grade": "portfolio"}])], ignore_index=True)
    return t[["grade"] + steps]


def estimate_pit_correlation(annual: pd.DataFrame, lra: pd.Series,
                             bounds: tuple[float, float] = (0.002, 0.5)) -> float:
    """The correlation at which the implied Z series has unit variance.

    The Vasicek transform treats Z as a standard normal economic factor. If the chosen
    correlation is too high, the implied Z is too calm (and vice versa), so Z stops
    being interpretable - and mapping macro scenarios to Z ("a severe recession is
    about Z = -2") stops working. A behavioural rating system absorbs much of the cycle
    through grade migration, so its WITHIN-grade correlation is typically well below the
    Basel value, which remains the right number for the capital formula.
    """
    def excess_std(r):
        return float(implied_z(annual, lra, r).z.std() - 1.0)
    lo, hi = bounds
    if excess_std(lo) * excess_std(hi) > 0:
        raise ValueError(f"No correlation in {bounds} gives unit-variance Z")
    return brentq(excess_std, lo, hi, xtol=1e-6)


def pit_log_likelihood(annual: pd.DataFrame, lra: pd.Series, correlation: float) -> float:
    """Binomial log-likelihood of every grade-year default count, given the implied Z."""
    from scipy.stats import binom
    z = implied_z(annual, lra, correlation).set_index("year").z
    p = vasicek.ttc_to_pit(lra.reindex(annual.grade).to_numpy(),
                           z.reindex(annual.year).to_numpy(), correlation)
    return float(binom.logpmf(annual.defaults, annual.n, np.clip(p, 1e-12, 1 - 1e-12)).sum())


def correlation_from_default_rates(dr) -> float:
    """Moment estimator under the one-factor model: Var(Phi^-1(DR_t)) = R / (1 - R),
    so R = V / (1 + V). Assumes a homogeneous, infinitely granular portfolio."""
    x = norm.ppf(np.clip(np.asarray(dr, float), 1e-9, 1 - 1e-9))
    v = float(np.var(x, ddof=1))
    return v / (1.0 + v)
