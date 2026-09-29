"""Behavioural PD scorecard: WOE logistic regression, points table and rating grades.

This is the RANKING model. It orders loans by risk and produces a raw PD. Calibration
to the long-run average (TTC), regulatory floors and margin of conservatism, and the
TTC -> PIT bridge are separate steps that consume its output.

Fitting procedure, every step recorded in the selection report:
  1. Bin every candidate driver on the development sample (features/binning.py).
  2. Drop drivers with information value below `min_iv`; FLAG very high IV for a
     leakage review rather than dropping it (behavioural arrears are legitimately strong).
  3. Correlation filter on WOE values: walk drivers in descending IV and drop any whose
     absolute correlation with an already-kept driver exceeds the limit.
  4. Fit logistic regression for the probability of DEFAULT. With WOE = ln(good/bad)
     every coefficient must be NEGATIVE; drop the lowest-IV offender (wrong sign or
     insignificant) and refit until none remain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score

from hcr.features.binning import Binning, bin_categorical, bin_numeric
from hcr.features.woe import score_scaling

__all__ = ["Scorecard", "load_sample", "assign_split", "fit_scorecard", "gini",
           "gini_by", "grade_table", "herfindahl"]


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #

def load_sample(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read the snapshot Parquet files into one DataFrame, economically.

    Strings are read dictionary-encoded, dictionaries are unified across files, and the
    Arrow -> pandas conversion frees each column as it goes (`self_destruct`), so peak
    memory stays close to the final frame. A naive read-then-concatenate held two or
    three full copies at once - enough to exhaust a laptop with 8 GB.
    """
    files = [f for f in sorted(Path(path).glob("*.parquet")) if pq.read_metadata(f).num_rows]
    if not files:
        raise FileNotFoundError(f"No non-empty Parquet files in {path}. "
                                f"Run scripts/build_pd_sample.py")
    schema = pq.read_schema(files[0])
    strings = [f.name for f in schema if str(f.type) in ("string", "large_string")
               and (columns is None or f.name in columns)]
    tables = [pq.read_table(f, columns=columns, read_dictionary=strings) for f in files]
    table = pa.concat_tables(tables).unify_dictionaries()
    del tables
    df = table.to_pandas(split_blocks=True, self_destruct=True)
    del table
    # Arrow's allocator keeps freed read buffers for reuse; hand them back to the OS.
    # On the full sample this cut resident memory from 2.6 GB to 1.5 GB.
    pa.default_memory_pool().release_unused()
    return df


def _bucket(values) -> np.ndarray:
    """Stable 0-9999 bucket per value from MD5 - identical on every machine and version."""
    return np.array([int(hashlib.md5(v.encode()).hexdigest()[:8], 16) % 10_000
                     for v in values], dtype=np.int64)


def assign_split(df: pd.DataFrame, oot_from_year: int, oos_share: float,
                 seed: int) -> pd.Series:
    """'dev' / 'oos' / 'oot'. OOS is chosen by LOAN, so a loan is never on both sides."""
    ids = df["loan_id"]
    if isinstance(ids.dtype, pd.CategoricalDtype):         # hash each distinct loan once
        per_cat = _bucket([f"{c}|{seed}" for c in ids.cat.categories])
        bucket = per_cat[ids.cat.codes.to_numpy()]
    else:
        uniq, inv = np.unique(ids.astype(str).to_numpy(), return_inverse=True)
        bucket = _bucket([f"{u}|{seed}" for u in uniq])[inv]
    split = np.where(bucket < oos_share * 10_000, "oos", "dev")
    split = np.where(df["snapshot_year"].to_numpy() >= oot_from_year, "oot", split)
    return pd.Series(pd.Categorical(split, categories=["dev", "oos", "oot"]),
                     index=df.index, name="split")


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #

@dataclass
class Scorecard:
    binnings: dict[str, Binning]
    features: list[str]
    params: pd.Series                         # 'const' + one coefficient per feature
    scaling: dict
    master_scale: list[float]
    selection: pd.DataFrame = field(default_factory=pd.DataFrame)

    def woe(self, df: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
        feats = features or self.features
        return pd.DataFrame({f: self.binnings[f].transform(df[f]).astype(np.float32)
                             for f in feats}, index=df.index)

    def log_odds(self, df: pd.DataFrame) -> np.ndarray:
        w = self.woe(df).to_numpy(dtype=float)
        return self.params["const"] + w @ self.params[self.features].to_numpy()

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-self.log_odds(df)))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        """Points: offset + factor * ln(odds of good) = offset - factor * log-odds(default)."""
        return self.scaling["offset"] - self.scaling["factor"] * self.log_odds(df)

    def grade(self, pd_values) -> np.ndarray:
        """Master-scale grade, 1 = safest."""
        return np.searchsorted(np.asarray(self.master_scale), np.asarray(pd_values),
                               side="left") + 1

    def points_table(self) -> pd.DataFrame:
        """Every bin's points. The total score is the sum across drivers - the whole
        model printed on one page."""
        k = len(self.features)
        base = (self.scaling["offset"] - self.scaling["factor"] * self.params["const"]) / k
        rows = []
        for f in self.features:
            t = self.binnings[f].table
            for _, r in t.iterrows():
                rows.append({"feature": f, "bin": r["bin"], "share": r["share"],
                             "bad_rate": r["bad_rate"], "woe": r["woe"],
                             "points": base - self.scaling["factor"] * self.params[f] * r["woe"]})
        return pd.DataFrame(rows)


class _Fit:
    """The parts of a statsmodels result the selection loop uses, with names."""

    def __init__(self, res, names: list[str]):
        self.params = pd.Series(res.params, index=["const"] + names)
        self.pvalues = pd.Series(res.pvalues, index=["const"] + names)


def _fit_logit(W: pd.DataFrame, y: np.ndarray) -> _Fit:
    """Logistic regression for P(default). One float64 design matrix, built once -
    passing DataFrames through statsmodels' helpers made several full copies."""
    X = np.empty((len(W), W.shape[1] + 1), dtype=np.float64)
    X[:, 0] = 1.0
    X[:, 1:] = W.to_numpy(dtype=np.float64)
    res = sm.Logit(y, X).fit(disp=0, method="newton", maxiter=50)
    return _Fit(res, list(W.columns))


def fit_scorecard(dev: pd.DataFrame, numeric: list[str], categorical: list[str],
                  cfg: dict) -> Scorecard:
    y = dev["target"].to_numpy(dtype=float)
    bn, bc, sel = cfg["binning"]["numeric"], cfg["binning"]["categorical"], cfg["selection"]

    binnings: dict[str, Binning] = {}
    for f in numeric:
        binnings[f] = bin_numeric(dev[f], y, name=f, max_bins=bn["max_bins"],
                                  fine_bins=bn["fine_bins"], min_share=bn["min_share"],
                                  min_bads=bn["min_bads"])
    for f in categorical:
        binnings[f] = bin_categorical(dev[f], y, name=f, min_share=bc["min_share"],
                                      min_bads=bc["min_bads"])

    report = pd.DataFrame({"feature": list(binnings),
                           "iv": [b.iv for b in binnings.values()],
                           "bins": [len(b.table) for b in binnings.values()]})
    report = report.sort_values("iv", ascending=False, ignore_index=True)
    report["status"] = np.where(report.iv < sel["min_iv"], "dropped: IV below minimum",
                                "candidate")
    report["flag"] = np.where(report.iv > sel["flag_iv_above"], "high IV - leakage review", "")

    # correlation filter, in descending IV
    cands = report.loc[report.status == "candidate", "feature"].tolist()
    probe = dev.sample(n=min(len(dev), 500_000), random_state=0)
    wp = pd.DataFrame({f: binnings[f].transform(probe[f]) for f in cands})
    corr = wp.corr().abs()
    kept: list[str] = []
    for f in cands:
        clash = [k for k in kept if corr.loc[f, k] > sel["max_abs_correlation"]]
        if clash:
            report.loc[report.feature == f, "status"] = (
                f"dropped: |corr| {corr.loc[f, clash[0]]:.2f} with {clash[0]}")
        else:
            kept.append(f)

    # logistic regression with sign and significance checks
    W = pd.DataFrame({f: binnings[f].transform(dev[f]).astype(np.float32) for f in kept},
                     index=dev.index)
    iv = dict(zip(report.feature, report.iv))
    while True:
        res = _fit_logit(W[kept], y)
        coef, pval = res.params.drop("const"), res.pvalues.drop("const")
        bad = [f for f in kept if coef[f] >= 0 or pval[f] > sel["max_p_value"]]
        if not bad:
            break
        worst = min(bad, key=lambda f: iv[f])
        why = "wrong sign" if coef[worst] >= 0 else f"p = {pval[worst]:.3g}"
        report.loc[report.feature == worst, "status"] = f"dropped in regression: {why}"
        kept.remove(worst)

    for f in kept:
        report.loc[report.feature == f, "status"] = "selected"
    report["coefficient"] = report.feature.map(res.params)
    report["p_value"] = report.feature.map(res.pvalues)
    # variance inflation factors of the final drivers
    Xv = W[kept].sample(n=min(len(W), 500_000), random_state=0).to_numpy(dtype=np.float64)
    ones = np.ones((len(Xv), 1))
    vif = {}
    for j, f in enumerate(kept):
        others = np.hstack([ones, np.delete(Xv, j, axis=1)])
        r2 = sm.OLS(Xv[:, j], others).fit().rsquared
        vif[f] = 1.0 / max(1e-12, 1.0 - r2)
    report["vif"] = report.feature.map(vif)

    sc = cfg["scaling"]
    return Scorecard(binnings=binnings, features=kept, params=res.params,
                     scaling=score_scaling(sc["pdo"], sc["base_score"], sc["base_odds"]),
                     master_scale=list(cfg["master_scale"]), selection=report)


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #

def gini(y, p) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return 2.0 * roc_auc_score(y, p) - 1.0


def gini_by(df: pd.DataFrame, pd_col: str, by: str) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        rows.append({by: key, "n": len(g), "defaults": int(g.target.sum()),
                     "default_rate": g.target.mean(), "gini": gini(g.target, g[pd_col])})
    return pd.DataFrame(rows)


def herfindahl(counts) -> dict:
    p = np.asarray(counts, float) / np.sum(counts)
    hi = float(np.sum(p ** 2))
    return {"herfindahl": hi, "effective_grades": 1.0 / hi}


def grade_table(grades, y, pd_values, master_scale) -> pd.DataFrame:
    df = pd.DataFrame({"grade": grades, "y": y, "pd": pd_values})
    t = df.groupby("grade").agg(n=("y", "size"), defaults=("y", "sum"),
                                default_rate=("y", "mean"), mean_model_pd=("pd", "mean"))
    t = t.reindex(range(1, len(master_scale) + 1), fill_value=0)
    t["share"] = t.n / t.n.sum()
    t["pd_upper_bound"] = master_scale
    return t.reset_index()[["grade", "pd_upper_bound", "n", "share", "defaults",
                            "default_rate", "mean_model_pd"]]
