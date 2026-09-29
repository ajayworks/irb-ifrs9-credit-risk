#!/usr/bin/env python
"""Calibrate the scorecard's grades: regulatory PD, IFRS 9 PD, and the bridge between them.

    python scripts/calibrate.py
    python scripts/calibrate.py --sample DIR --model PATH

Needs the development sample (scripts/build_pd_sample.py) and the fitted scorecard
(scripts/fit_scorecard.py). Writes to outputs/calibration/:
    grade_calibration.csv   per grade: long-run average, MoC, floor, regulatory PD
    moc_items.csv           every margin-of-conservatism item and its status
    annual_grade_rates.csv  one-year default rate per grade and year
    z_by_year.csv           the economic factor implied by each year's defaults
    bridge_by_year.csv      regulatory vs IFRS 9 PD for the whole portfolio, every year
    waterfall_<year>.csv    regulatory PD -> IFRS 9 PD, step by step, for the latest year
    bridge_by_year.png      the exhibit
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcr.config import REPO_ROOT, load                 # noqa: E402
from hcr.engines import irb                            # noqa: E402
from hcr.pd import calibration as C                    # noqa: E402
from hcr.pd import scorecard as SC                     # noqa: E402

OUT = REPO_ROOT / "outputs" / "calibration"
pd.set_option("display.width", 220)
pd.set_option("display.max_rows", 100)
pct = lambda x: f"{x:.4%}"                             # noqa: E731


def chart(bridge: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.2))
    x = bridge.year.to_numpy()
    ax.plot(x, bridge.regulatory_pd * 100, lw=2.2, color="#1f4e79",
            label="Regulatory PD (through-the-cycle, with MoC and floor)")
    ax.plot(x, bridge.pit_pd * 100, lw=2.2, color="#c0504d",
            label="IFRS 9 PD (point-in-time)")
    ax.fill_between(x, bridge.regulatory_pd * 100, bridge.pit_pd * 100,
                    where=bridge.pit_pd > bridge.regulatory_pd, color="#c0504d", alpha=0.15,
                    interpolate=True, label="IFRS 9 above regulatory")
    ax.set_ylabel("One-year PD, portfolio average (%)")
    ax.set_title("One set of grades, two PDs: the gap reverses in a downturn")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("Snapshot year (PD for the following 12 months)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg, moc_cfg, sc_cfg = load("calibration"), load("moc"), load("scorecard")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sample", type=Path, default=REPO_ROOT / sc_cfg["sample"]["path"])
    ap.add_argument("--model", type=Path,
                    default=REPO_ROOT / "data" / "processed" / "models" / "scorecard.pkl")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with args.model.open("rb") as fh:
        model = pickle.load(fh)
    df = SC.load_sample(args.sample, ["snapshot_year", "target", "current_upb"] + model.features)
    yrs = cfg["observation_years"]
    df = df[(df.snapshot_year >= yrs["start"]) & (df.snapshot_year <= yrs["end"])]
    df["grade"] = model.grade(model.predict_pd(df))
    grades_all = pd.Index(range(1, len(model.master_scale) + 1), name="grade")

    # --- long-run average per grade
    annual = C.annual_grade_rates(df.grade, df.target, df.snapshot_year)
    lra = C.long_run_average(annual, cfg["long_run_average"]["weighting"]).set_index("grade")
    lra = lra.reindex(grades_all.intersection(lra.index))
    raw = lra.lra.copy()
    if cfg["long_run_average"]["enforce_monotonic"]:
        lra["lra"] = C.enforce_monotonic(lra.lra, lra.n)
    lra["monotonic_adjustment"] = lra.lra - raw

    # --- margin of conservatism
    c_item = moc_cfg["categories"]["C"]["items"][0]["bootstrap"]
    boot = C.bootstrap_moc(annual, c_item["n_resamples"], c_item["percentile"],
                           c_item["seed"]).set_index("grade")
    ab = C.moc_add_ons(moc_cfg)
    floor = float(irb.apply_pd_floor(0.0, cfg["regulatory"]["exposure_class"]))
    reg = C.regulatory_pd(lra.lra.to_numpy(), ab["A"], ab["B"],
                          boot.moc_c.reindex(lra.index).to_numpy(), floor)
    reg.index = lra.index
    reg["regulatory_pd"] = C.enforce_monotonic(reg.regulatory_pd, lra.n)

    # --- point in time
    r_cfg, r_cmp = cfg["pit"]["correlation"], float(cfg["pit"]["comparison_correlation"])
    R = C.estimate_pit_correlation(annual, reg.lra) if r_cfg == "estimate" else float(r_cfg)
    z = C.implied_z(annual, reg.lra, R)
    z_cmp = C.implied_z(annual, reg.lra, r_cmp).z
    ll, ll_cmp = C.pit_log_likelihood(annual, reg.lra, R), C.pit_log_likelihood(annual, reg.lra, r_cmp)
    bridge = C.bridge_by_year(annual, reg.regulatory_pd, reg.lra, z.set_index("year").z, R)

    rep = cfg["bridge"]["reporting_year"]
    mix = annual[annual.year == rep].set_index("grade").n.reindex(reg.index, fill_value=0)
    wf = C.waterfall(reg.reset_index(), float(z.set_index("year").z[rep]), R, weights=mix)

    # --- illustrative capital (LGD placeholder until the LGD model exists). Capital always
    # uses the PRESCRIBED correlation, never the estimated PIT one.
    lgd = float(cfg["capital"]["illustrative_downturn_lgd"])
    reg["rw_illustrative"] = irb.risk_weight(reg.regulatory_pd.to_numpy(), lgd,
                                             irb.correlation_residential_mortgage())
    ead = df[df.snapshot_year == rep].groupby("grade").current_upb.sum().reindex(reg.index,
                                                                                 fill_value=0)
    rw_density = float(np.sum(reg.rw_illustrative * ead) / ead.sum())

    table = pd.concat([lra[["years", "n", "defaults", "lra_equal", "lra_pooled",
                            "monotonic_adjustment"]],
                       reg[["lra", "moc_a", "moc_b", "moc_c", "pd_before_floor",
                            "floor_uplift", "regulatory_pd", "rw_illustrative"]]], axis=1)
    table.insert(0, "pd_upper_bound", [model.master_scale[g - 1] for g in table.index])

    # --- report
    print(f"{len(df):,} snapshots, {df.snapshot_year.min()}-{df.snapshot_year.max()}; "
          f"long-run average weighting: {cfg['long_run_average']['weighting']}; "
          f"UK PD floor {floor:.2%}\n")
    show = table[["n", "defaults", "lra_equal", "lra_pooled", "lra", "moc_c",
                  "floor_uplift", "regulatory_pd"]].copy()
    show[["n", "defaults"]] = show[["n", "defaults"]].astype(int)
    print("Grade calibration\n" + show.to_string(
        formatters={c: pct for c in show.columns if c not in ("n", "defaults")}))
    floored = table.index[table.floor_uplift > 0]
    print(f"\nFloor binds for grades {list(floored)}: "
          f"{table.loc[floored, 'n'].sum() / table.n.sum():.1%} of observations")
    print(f"Margin of conservatism: C quantified by year bootstrap (p{int(c_item['percentile']*100)}); "
          f"A = {ab['A']:.4%}, B = {ab['B']:.4%}; PENDING quantification: {', '.join(ab['pending'])}")
    zz = z.set_index("year").z
    print(f"\nPIT correlation: {R:.4f} ({'estimated' if r_cfg == 'estimate' else 'configured'}) "
          f"-> implied Z mean {zz.mean():+.2f}, std {zz.std():.2f}, grade-year log-likelihood "
          f"{ll:,.0f}")
    print(f"Comparison at {r_cmp} (Basel capital value): Z mean {z_cmp.mean():+.2f}, "
          f"std {z_cmp.std():.2f}, log-likelihood {ll_cmp:,.0f} ({ll_cmp - ll:+,.0f})")
    print("\nBridge by year - portfolio averages\n" + bridge[[
        "year", "regulatory_pd", "ttc_unbiased_pd", "pit_pd", "regulatory_over_pit"]].to_string(
        index=False, float_format=lambda x: f"{x:.4f}"))
    rev = bridge[bridge.regulatory_over_pit < 1].year.tolist()
    print(f"\nYears where IFRS 9 PD exceeds regulatory PD: {rev}")
    print(f"\nWaterfall {rep} (Z = {zz[rep]:+.2f})\n"
          + wf.to_string(index=False, float_format=pct))
    print(f"\nIllustrative only (placeholder downturn LGD {lgd:.0%}): EAD-weighted risk weight "
          f"{rw_density:.2%} in {rep}")

    args.out.mkdir(parents=True, exist_ok=True)
    table.reset_index().to_csv(args.out / "grade_calibration.csv", index=False)
    pd.DataFrame(ab["items"]).to_csv(args.out / "moc_items.csv", index=False)
    annual.to_csv(args.out / "annual_grade_rates.csv", index=False)
    z.to_csv(args.out / "z_by_year.csv", index=False)
    bridge.to_csv(args.out / "bridge_by_year.csv", index=False)
    wf.to_csv(args.out / f"waterfall_{rep}.csv", index=False)
    chart(bridge, args.out / "bridge_by_year.png")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
