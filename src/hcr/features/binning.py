"""Coarse classification: monotonic WOE binning for numeric and categorical drivers.

Numeric drivers
    1. Cut into fine quantile bins (or one bin per value when there are few values).
    2. Merge adjacent bins until the bad rate is MONOTONIC in the chosen direction.
    3. Merge bins that are too small (share of population) or too thin (defaults),
       each into whichever neighbour has the closer bad rate.
    4. Merge the closest adjacent pair until at most `max_bins` remain.
    Merging adjacent bins of a monotonic sequence keeps it monotonic, so steps 3-4
    cannot undo step 2.

Categorical drivers
    Levels below the size or default thresholds are pooled into OTHER. If OTHER is
    itself too thin, it is merged into the riskiest level (conservative).

Missing values
    Their own bin when material (enough rows and defaults). Otherwise they are pooled
    with the RISKIEST bin - conservative, and stated in the table - rather than being
    silently dropped or assigned zero WOE.

WOE = ln(share of goods / share of bads), as in features/woe.py: higher WOE = lower
risk. In a model of the probability of default, coefficients on WOE are negative.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Binning", "bin_numeric", "bin_categorical", "MISSING", "OTHER"]

MISSING = "MISSING"
OTHER = "OTHER"


@dataclass
class Binning:
    name: str
    kind: str                                   # "numeric" | "categorical"
    table: pd.DataFrame                         # one row per bin; includes MISSING if own bin
    iv: float
    direction: str | None = None                # numeric: "increasing" | "decreasing" risk
    edges: list[float] = field(default_factory=list)       # numeric interior cut points
    groups: dict[str, str] = field(default_factory=dict)   # categorical: level -> bin label
    missing_policy: str = "none"                # "own_bin" | "pooled_with_riskiest" | "none"
    missing_woe: float = 0.0
    fallback_woe: float = 0.0                   # categorical: unseen levels
    missing_label: str = MISSING                # bin that missing values fall into
    fallback_label: str = ""                    # categorical: bin for unseen levels

    def positions(self, x) -> np.ndarray:
        """Row of `table` that each raw value falls into. The single source of truth
        for bin membership: bin_label(), transform() and the points table all use it.
        Works on integer positions, so scoring millions of rows stays fast."""
        bins = list(self.table["bin"])
        pos_of = {b: i for i, b in enumerate(bins)}
        s = pd.Series(x)
        if self.kind == "numeric":
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
            nonmiss = np.array([i for i, b in enumerate(bins) if b != MISSING])
            idx = np.searchsorted(np.asarray(self.edges, dtype=float), v, side="left")
            pos = nonmiss[np.minimum(idx, len(nonmiss) - 1)]
            return np.where(np.isnan(v), pos_of[self.missing_label], pos)
        missing_pos = pos_of[self.missing_label]
        if isinstance(s.dtype, pd.CategoricalDtype):
            codes = s.cat.codes.to_numpy()
            cats = [str(c) for c in s.cat.categories]
            if not cats:
                return np.full(len(s), missing_pos)
            cat_pos = np.array([pos_of[self.groups.get(c, self.fallback_label)] for c in cats])
            return np.where(codes < 0, missing_pos, cat_pos[np.maximum(codes, 0)])
        raw = s.astype("object").where(s.notna(), MISSING).astype(str)
        return raw.map(self.groups).fillna(self.fallback_label).map(pos_of).to_numpy(dtype=int)

    def bin_label(self, x) -> np.ndarray:
        """Bin label for each raw value."""
        return np.asarray(self.table["bin"], dtype=object)[self.positions(x)]

    def transform(self, x) -> np.ndarray:
        """Map raw values to WOE."""
        return self.table["woe"].to_numpy(dtype=float)[self.positions(x)]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _woe_frame(labels, n, bad, total_good, total_bad) -> pd.DataFrame:
    n, bad = np.asarray(n, float), np.asarray(bad, float)
    good = n - bad
    dg, db = good / total_good, bad / total_bad
    with np.errstate(divide="ignore"):
        woe = np.log(np.clip(dg, 1e-12, None) / np.clip(db, 1e-12, None))
    return pd.DataFrame({
        "bin": labels, "n": n.astype(int), "bads": bad.astype(int),
        "share": n / (total_good + total_bad), "bad_rate": bad / n,
        "woe": woe, "iv_contribution": (dg - db) * woe,
    })


def _merge(n: list, bad: list, left: list, i: int) -> None:
    """Merge bin i+1 into bin i (in place). `left` holds each bin's lower edge index."""
    n[i] += n.pop(i + 1)
    bad[i] += bad.pop(i + 1)
    left.pop(i + 1)


def _violations(rate: np.ndarray, increasing: bool) -> np.ndarray:
    d = np.diff(rate)
    return np.where(d < 0)[0] if increasing else np.where(d > 0)[0]


# --------------------------------------------------------------------------- #
# numeric
# --------------------------------------------------------------------------- #

def bin_numeric(x, y, name: str = "x", max_bins: int = 8, fine_bins: int = 20,
                min_share: float = 0.05, min_bads: int = 30,
                direction: str = "auto") -> Binning:
    """Monotonic WOE binning of a numeric driver. `y` is 1 for default, 0 otherwise."""
    x = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    total_n, total_bad = len(y), y.sum()
    total_good = total_n - total_bad
    if total_bad == 0 or total_good == 0:
        raise ValueError(f"{name}: need both defaults and non-defaults")

    miss = np.isnan(x)
    xv, yv = x[~miss], y[~miss]
    if len(xv) == 0:
        raise ValueError(f"{name}: every value is missing")

    uniq = np.unique(xv)
    if len(uniq) <= fine_bins:
        edges = list(uniq[:-1])                       # one bin per distinct value
    else:
        qs = np.quantile(xv, np.linspace(0, 1, fine_bins + 1)[1:-1])
        edges = list(np.unique(qs))
        edges = [e for e in edges if e < uniq[-1]]
    idx = np.searchsorted(np.asarray(edges), xv, side="left")
    k = len(edges) + 1
    n = list(np.bincount(idx, minlength=k).astype(float))
    bad = list(np.bincount(idx, weights=yv, minlength=k))
    left = list(range(k))                             # bin j starts after edges[left[j]-1]

    # drop empty fine bins
    for j in range(k - 1, -1, -1):
        if n[j] == 0 and len(n) > 1:
            (_merge(n, bad, left, j - 1) if j > 0 else _merge(n, bad, left, 0))

    rate = np.array(bad) / np.array(n)
    if direction == "auto":
        w = np.array(n)
        xs = np.arange(len(rate), dtype=float)
        cov = np.average((xs - np.average(xs, weights=w)) * (rate - np.average(rate, weights=w)),
                         weights=w)
        direction = "increasing" if cov >= 0 else "decreasing"
    if direction not in {"increasing", "decreasing"}:
        raise ValueError("direction must be 'auto', 'increasing' or 'decreasing'")
    inc = direction == "increasing"

    # 2. monotonicity
    while len(n) > 1:
        v = _violations(np.array(bad) / np.array(n), inc)
        if len(v) == 0:
            break
        _merge(n, bad, left, int(v[0]))

    # 3. size and default thresholds (share measured on the whole population)
    def too_small(j):
        return n[j] / total_n < min_share or bad[j] < min_bads

    while len(n) > 1 and any(too_small(j) for j in range(len(n))):
        r = np.array(bad) / np.array(n)
        j = min((j for j in range(len(n)) if too_small(j)), key=lambda j: n[j])
        if j == 0:
            _merge(n, bad, left, 0)
        elif j == len(n) - 1:
            _merge(n, bad, left, j - 1)
        else:
            (_merge(n, bad, left, j - 1) if abs(r[j] - r[j - 1]) <= abs(r[j] - r[j + 1])
             else _merge(n, bad, left, j))

    # 4. at most max_bins: merge the closest adjacent pair
    while len(n) > max_bins:
        r = np.array(bad) / np.array(n)
        _merge(n, bad, left, int(np.argmin(np.abs(np.diff(r)))))

    final_edges = [edges[left[j] - 1] for j in range(1, len(left))]
    labels = []
    lo = "-inf"
    for j in range(len(n)):
        hi = f"{final_edges[j]:g}" if j < len(final_edges) else "inf"
        labels.append(f"({lo}, {hi}]" if hi != "inf" else f"({lo}, inf)")
        lo = hi

    # missing values
    n_m, bad_m = float(miss.sum()), float(y[miss].sum())
    riskiest = int(np.argmax(np.array(bad) / np.array(n)))
    if n_m == 0:
        policy = "none"
    elif n_m / total_n >= min_share and bad_m >= min_bads and (n_m - bad_m) > 0:
        policy = "own_bin"
    else:
        policy = "pooled_with_riskiest"
        n[riskiest] += n_m
        bad[riskiest] += bad_m

    rows_n, rows_bad, rows_lab = list(n), list(bad), list(labels)
    if policy == "own_bin":
        rows_n.append(n_m), rows_bad.append(bad_m), rows_lab.append(MISSING)
    table = _woe_frame(rows_lab, rows_n, rows_bad, total_good, total_bad)
    # Missing values seen in scoring but not in development also go to the riskiest bin.
    missing_label = MISSING if policy == "own_bin" else labels[riskiest]
    missing_woe = float(table.loc[table.bin == missing_label, "woe"].iloc[0])

    return Binning(name=name, kind="numeric", table=table,
                   iv=float(table.iv_contribution.sum()), direction=direction,
                   edges=final_edges, missing_policy=policy, missing_woe=missing_woe,
                   missing_label=missing_label)


# --------------------------------------------------------------------------- #
# categorical
# --------------------------------------------------------------------------- #

def bin_categorical(x, y, name: str = "x", min_share: float = 0.02,
                    min_bads: int = 30) -> Binning:
    """WOE per level, with rare levels pooled into OTHER."""
    s = pd.Series(x).astype("object")
    s = s.where(s.notna(), MISSING).astype(str)
    y = np.asarray(y, dtype=float)
    total_n, total_bad = len(y), y.sum()
    total_good = total_n - total_bad
    if total_bad == 0 or total_good == 0:
        raise ValueError(f"{name}: need both defaults and non-defaults")

    agg = pd.DataFrame({"lvl": s.to_numpy(), "y": y}).groupby("lvl")["y"].agg(["size", "sum"])
    small = (agg["size"] / total_n < min_share) | (agg["sum"] < min_bads)
    groups = {lvl: (OTHER if small[lvl] else lvl) for lvl in agg.index}

    g = agg.assign(bin=[groups[i] for i in agg.index]).groupby("bin")[["size", "sum"]].sum()
    if OTHER in g.index and len(g) > 1 and (g.loc[OTHER, "sum"] < min_bads
                                             or g.loc[OTHER, "size"] - g.loc[OTHER, "sum"] <= 0):
        rest = g.drop(index=OTHER)
        worst = (rest["sum"] / rest["size"]).idxmax()
        groups = {k: (worst if v == OTHER else v) for k, v in groups.items()}
        g = agg.assign(bin=[groups[i] for i in agg.index]).groupby("bin")[["size", "sum"]].sum()

    table = _woe_frame(list(g.index), g["size"].to_numpy(), g["sum"].to_numpy(),
                       total_good, total_bad)
    table = table.sort_values("bad_rate", ignore_index=True)
    # Unseen levels (and missing, if never seen) go to OTHER, else the riskiest level.
    fallback_label = OTHER if OTHER in set(table.bin) else str(table.bin.iloc[-1])
    fallback = float(table.loc[table.bin == fallback_label, "woe"].iloc[0])
    groups.setdefault(MISSING, fallback_label)
    missing_label = groups[MISSING]
    return Binning(name=name, kind="categorical", table=table,
                   iv=float(table.iv_contribution.sum()), groups=groups,
                   missing_policy="own_bin" if missing_label == MISSING else "none",
                   missing_woe=float(table.loc[table.bin == missing_label, "woe"].iloc[0]),
                   fallback_woe=fallback, missing_label=missing_label,
                   fallback_label=fallback_label)
