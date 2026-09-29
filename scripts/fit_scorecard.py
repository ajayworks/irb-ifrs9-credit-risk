#!/usr/bin/env python
"""Fit the behavioural PD scorecard and report how well it ranks.

    python scripts/fit_scorecard.py                 # sample path from config/scorecard.yaml
    python scripts/fit_scorecard.py --sample DIR

Writes to outputs/scorecard/ (tables for the MDD):
    selection.csv      every candidate driver: IV, status and reason, coefficient, VIF
    points_table.csv   the whole model on one page
    gini.csv           discrimination: development, out-of-sample, out-of-time
    gini_by_year.csv   stability through the cycle
    grades.csv         master-scale grade distribution and default rates
and the fitted model to data/processed/models/scorecard.pkl (git-ignored).
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcr.config import REPO_ROOT, load                 # noqa: E402
from hcr.pd import sample as S                         # noqa: E402
from hcr.pd import scorecard as SC                     # noqa: E402

OUT = REPO_ROOT / "outputs" / "scorecard"
MODEL = REPO_ROOT / "data" / "processed" / "models" / "scorecard.pkl"
pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 200)


def main() -> None:
    cfg = load("scorecard")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=Path, default=REPO_ROOT / cfg["sample"]["path"])
    args = ap.parse_args()
    t0 = time.time()

    excl = set(cfg["features"]["exclude"])
    num = [f for f in S.NUMERIC_FEATURES if f not in excl]
    cat = [f for f in S.CATEGORICAL_FEATURES if f not in excl]
    df = SC.load_sample(args.sample, ["loan_id", "snapshot_year", "target"] + num + cat)
    sp = cfg["split"]
    df["split"] = SC.assign_split(df, sp["oot_from_year"], sp["oos_share"], sp["seed"])
    print(f"{len(df):,} snapshots | " + ", ".join(
        f"{k} {v:,}" for k, v in df.split.value_counts().sort_index().items())
        + f" | loaded in {time.time() - t0:.0f}s")

    model = SC.fit_scorecard(df[df.split == "dev"], num, cat, cfg)
    df["pd_hat"] = model.predict_pd(df)
    df["grade"] = model.grade(df.pd_hat)
    print(f"fitted in {time.time() - t0:.0f}s\n")

    sel = model.selection[["feature", "iv", "bins", "status", "flag", "coefficient", "vif"]]
    print("Driver selection\n" + sel.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    gini = pd.DataFrame([{"sample": s, "n": int((df.split == s).sum()),
                          "default_rate": df.loc[df.split == s, "target"].mean(),
                          "mean_model_pd": df.loc[df.split == s, "pd_hat"].mean(),
                          "gini": SC.gini(df.loc[df.split == s, "target"],
                                          df.loc[df.split == s, "pd_hat"])}
                         for s in ("dev", "oos", "oot")])
    print("\nDiscrimination\n" + gini.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    by_year = SC.gini_by(df, "pd_hat", "snapshot_year")
    print("\nGini by snapshot year\n" + by_year.to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))

    grades = []
    for label, mask in (("development years", df.split != "oot"), ("out-of-time", df.split == "oot")):
        g = SC.grade_table(df.loc[mask, "grade"], df.loc[mask, "target"],
                           df.loc[mask, "pd_hat"], model.master_scale)
        h = SC.herfindahl(g.n[g.n > 0])
        print(f"\nGrades - {label} (effective grades {h['effective_grades']:.2f} of "
              f"{int((g.n > 0).sum())} used)\n"
              + g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        grades.append(g.assign(sample=label))

    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "selection.csv", index=False)
    model.points_table().to_csv(OUT / "points_table.csv", index=False)
    gini.to_csv(OUT / "gini.csv", index=False)
    by_year.to_csv(OUT / "gini_by_year.csv", index=False)
    pd.concat(grades).to_csv(OUT / "grades.csv", index=False)
    MODEL.parent.mkdir(parents=True, exist_ok=True)
    with MODEL.open("wb") as fh:
        pickle.dump(model, fh)
    print(f"\nDone in {time.time() - t0:.0f}s. Tables in {OUT.relative_to(REPO_ROOT)}/")


if __name__ == "__main__":
    main()
