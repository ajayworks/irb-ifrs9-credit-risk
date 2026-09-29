#!/usr/bin/env python
"""End-to-end pipeline on synthetic data.

Builds the panel, runs every definition-of-default variant, estimates a
through-the-cycle PD, recovers the systematic factor, and demonstrates the
TTC -> PIT bridge alongside IRB capital and IFRS 9 ECL from the same inputs.

    python scripts/run_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hcr.default_def import panel as P            # noqa: E402
from hcr.engines import ecl as E                  # noqa: E402
from hcr.engines import irb                       # noqa: E402
from hcr.pd import vasicek                        # noqa: E402

pd.set_option("display.width", 220)
R_MORTGAGE = irb.correlation_residential_mortgage()


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:
    con = P.connect(":memory:")
    try:
        P.load_synthetic(con)
    except FileNotFoundError as exc:
        print(f"{exc}\n\nRun:  python -m hcr.data.synthetic")
        raise SystemExit(1)

    rule("1. Definition of default - variant sensitivity")
    variants = P.compare_variants(con)
    print(variants.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print("\nThis table goes into the MDD. Showing the sensitivity is what turns a")
    print("judgement into a documented judgement.")

    rule("2. Observed one-year default rate by year")
    by_year = P.default_rate_by_period(con, "obs_baseline")
    print(by_year[["year", "n_obs", "n_default", "default_rate", "mean_z",
                   "unemployment"]].to_string(
        index=False, float_format=lambda x: f"{x:,.4f}"))

    # ---------------------------------------------------------------- TTC
    rule("3. Through-the-cycle calibration")
    lra = by_year.n_default.sum() / by_year.n_obs.sum()
    print(f"Long-run average default rate      {lra:>10.4%}")
    print(f"Peak year ({by_year.loc[by_year.default_rate.idxmax(), 'year']})"
          f"{'':<21}{by_year.default_rate.max():>10.4%}")
    print(f"Trough year"
          f"{'':<24}{by_year.default_rate[by_year.default_rate > 0].min():>10.4%}")
    print(f"Cycle amplitude                    "
          f"{by_year.default_rate.max() / by_year.default_rate[by_year.default_rate > 0].min():>9.1f}x")

    pd_ttc_floored = float(irb.apply_pd_floor(lra, "residential_mortgage"))
    moc = {"A - data deficiencies": 0.0010, "B - underwriting change": 0.0008,
           "C - estimation error": 0.0006}
    pd_regulatory = pd_ttc_floored + sum(moc.values())
    print(f"\nAfter UK PD floor (0.10%)          {pd_ttc_floored:>10.4%}")
    for k, v in moc.items():
        print(f"  + MoC {k:<30} {v:>10.4%}")
    print(f"Regulatory PD                      {pd_regulatory:>10.4%}")

    # ---------------------------------------------------------------- Z
    rule("4. Systematic factor recovered from observed default rates")
    obs = {r.year: r.default_rate for r in by_year.itertuples() if r.default_rate > 0}
    z_implied = vasicek.z_from_default_rates(obs, lra, R_MORTGAGE)
    z_actual = dict(zip(by_year.year, by_year.mean_z))
    comp = pd.DataFrame({
        "year": list(z_implied),
        "default_rate": [obs[y] for y in z_implied],
        "z_implied": list(z_implied.values()),
        "z_true": [z_actual[y] for y in z_implied],
    })
    print(comp.to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    corr = np.corrcoef(comp.z_implied, comp.z_true)[0, 1]
    print(f"\nCorrelation between recovered and true Z: {corr:.4f}")
    print("On synthetic data the true Z is knowable, so this is a real check that the")
    print("inversion works. On real data you can never do this.")

    # ---------------------------------------------------------------- bridge
    rule("5. The bridge - regulatory PD to IFRS 9 PD")
    for label, z in [("benign  (Z = +0.8)", 0.8), ("neutral (Z =  0.0)", 0.0),
                     ("stress  (Z = -1.2)", -1.2)]:
        pit = float(vasicek.ttc_to_pit(pd_ttc_floored, z, R_MORTGAGE))
        ratio = pd_regulatory / pit
        direction = "regulatory higher" if ratio > 1 else "ACCOUNTING HIGHER"
        print(f"  {label}   IFRS 9 PD {pit:>8.4%}   "
              f"regulatory/accounting {ratio:>6.2f}x   {direction}")
    print("\nThe relationship reverses in stress. That reversal is the exhibit.")

    # ---------------------------------------------------------------- capital
    rule("6. IRB capital and the output floor")
    lgd_downturn = float(irb.apply_lgd_floor(0.28, "residential_mortgage"))
    ead = 1_000_000.0
    k = float(irb.capital_requirement(pd_regulatory, lgd_downturn, R_MORTGAGE))
    rw = float(irb.risk_weight(pd_regulatory, lgd_downturn, R_MORTGAGE))
    print(f"Regulatory PD {pd_regulatory:.4%}, downturn LGD {lgd_downturn:.2%}, "
          f"EAD {ead:,.0f}")
    print(f"  K   {k:>10.5f}      RW {rw:>8.2%}      RWA {rw * ead:>14,.0f}")
    print(f"  EL  {float(irb.expected_loss(pd_regulatory, lgd_downturn, ead)):>14,.0f}")

    rwa_irb = rw * ead
    rwa_sa = 0.35 * ead          # illustrative standardised risk weight
    print(f"\nAgainst a standardised RWA of {rwa_sa:,.0f}:")
    for year in (2027, 2028, 2029, 2030):
        r = irb.apply_output_floor(rwa_irb, rwa_sa, year)
        flag = f"FLOOR BINDS  +{r['uplift']:,.0f}" if r["floor_binding"] else "IRB governs"
        print(f"  {year}  floor {r['floor_pct']:>5.1%}  -> {r['rwa_floored']:>12,.0f}"
              f"   final {r['rwa_final']:>12,.0f}   {flag}")

    # ---------------------------------------------------------------- ECL
    rule("7. IFRS 9 ECL from the SAME parameter set")
    hazard_base = np.full(60, pd_ttc_floored / 12.0)
    lgd_pit = 0.19                      # point-in-time, NOT the downturn figure
    balance = ead * (1 - np.arange(60) / 300.0)

    ecl_by_scenario = {}
    for name, z in [("upside", 1.1), ("base", 0.2), ("downside", -1.0), ("severe", -2.0)]:
        z_path = np.concatenate([np.full(12, z), vasicek.mean_revert(z, 48, speed=0.35)])
        h = np.array([vasicek.ttc_to_pit(pd_ttc_floored / 12.0, zz, R_MORTGAGE)
                      for zz in z_path])
        m = E.marginal_pd_from_survival(E.survival_from_hazard(h))
        E.assert_unbiased_inputs(h, moc_applied=False, downturn_lgd=False)
        ecl_by_scenario[name] = E.ecl(m, lgd_pit, balance, eir=0.045, periods_per_year=12)

    res = E.scenario_weighted_ecl(ecl_by_scenario)
    for name, value in ecl_by_scenario.items():
        print(f"  {name:<9} ECL {value:>12,.0f}   contribution "
              f"{res['contribution'][name]:>12,.0f}")
    print(f"\n  Probability-weighted ECL   {res['ecl_weighted']:>12,.0f}")
    print(f"  Base scenario alone        {res['ecl_base_only']:>12,.0f}")
    print(f"  Non-linearity uplift       {res['non_linearity_uplift']:>12,.0f}"
          f"   ({res['non_linearity_pct']:+.1%})")

    rule("8. Side by side")
    print(f"  IRB RWA (after output floor, 2027)  {irb.apply_output_floor(rwa_irb, rwa_sa, 2027)['rwa_final']:>14,.0f}")
    print(f"  Regulatory expected loss            {float(irb.expected_loss(pd_regulatory, lgd_downturn, ead)):>14,.0f}")
    print(f"  IFRS 9 lifetime ECL                 {res['ecl_weighted']:>14,.0f}")
    print("\n  Same portfolio, same underlying data, different regulatory purposes.")
    print("  The gap between the last two is the EL-versus-provision comparison that")
    print("  drives the CET1 shortfall or excess.")
    print()


if __name__ == "__main__":
    main()
