"""Synthetic loan-month panel with a macro-driven default process.

Purpose: make every module runnable and testable TODAY, before the Freddie Mac
download lands, and permanently thereafter for unit tests.

The schema deliberately mirrors the Freddie Mac standard dataset so the real
loader is a drop-in replacement. The generating process is knowable, which means
tests can assert that estimation recovers the truth - something you can never do
on real data.

Generating process:
  * a systematic factor Z follows an AR(1) with a crisis and a COVID shock
  * monthly default hazard = logistic(base + seasoning + LTV + score + DTI - sqrt(R)*Z)
  * delinquency rolls 0 -> 1 -> 2 -> 3, with cures back to 0
  * prepayment competes as a second exit, rate-sensitive and seasoned
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hcr.config import load

_CFG = load("data")["synthetic"]

DELINQ_CURRENT, DELINQ_90PLUS = 0, 3
# Buckets run beyond 3 so DoD variants above 90 DPD (e.g. strict_120dpd) are
# meaningful. Capping at 3 would make any stricter threshold trivially empty.
DELINQ_MAX = 6
ZB_PREPAID, ZB_THIRD_PARTY, ZB_SHORT_SALE, ZB_REO = "01", "02", "03", "09"


def _macro(periods: pd.PeriodIndex, rng: np.random.Generator) -> pd.DataFrame:
    """AR(1) systematic factor with a 2008-10 crisis and a 2020 COVID shock."""
    n = len(periods)
    z = np.zeros(n)
    phi, sigma = 0.97, 0.18
    for t in range(1, n):
        z[t] = phi * z[t - 1] + rng.normal(0.0, sigma)

    yrs = periods.year.to_numpy()
    mth = periods.month.to_numpy()
    t_idx = (yrs - yrs.min()) * 12 + (mth - 1)

    crisis_c = (2008 - yrs.min()) * 12 + 9
    z -= 2.2 * np.exp(-0.5 * ((t_idx - crisis_c) / 14.0) ** 2)
    covid_c = (2020 - yrs.min()) * 12 + 3
    z -= 1.4 * np.exp(-0.5 * ((t_idx - covid_c) / 5.0) ** 2)
    z = (z - z.mean()) / z.std()

    unemployment = 5.8 - 2.1 * z + rng.normal(0, 0.12, n)
    hpi_growth = 0.028 + 0.055 * z + rng.normal(0, 0.008, n)
    hpi_level = 100.0 * np.cumprod(1.0 + hpi_growth / 12.0)
    return pd.DataFrame({
        "period": periods.astype(str),
        "z": z,
        "unemployment_rate": unemployment,
        "hpi_growth_yoy": hpi_growth,
        "hpi_level": hpi_level,
        "mortgage_rate_30y": 5.6 - 0.9 * z + rng.normal(0, 0.15, n),
    })


def _seasoning(age_months: np.ndarray) -> np.ndarray:
    """Low in year 1, peaking around years 3-5, declining thereafter."""
    a = age_months / 12.0
    return 1.35 * np.exp(-0.5 * ((a - 3.8) / 2.6) ** 2) - 0.55 * np.exp(-a / 0.8)


def generate(n_loans: int | None = None, seed: int | None = None,
             out_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    n_loans = n_loans or _CFG["n_loans"]
    rng = np.random.default_rng(seed if seed is not None else _CFG["random_seed"])

    periods = pd.period_range(_CFG["start_period"], _CFG["end_period"], freq="M")
    macro = _macro(periods, rng)
    z_by_t = macro["z"].to_numpy()
    hpi_by_t = macro["hpi_level"].to_numpy()
    n_periods = len(periods)

    # ---------------- origination ----------------
    orig_t = rng.integers(0, n_periods - 24, size=n_loans)
    credit_score = np.clip(rng.normal(725, 52, n_loans), 520, 830).astype(int)
    orig_ltv = np.clip(rng.normal(76, 14, n_loans), 25, 100).astype(int)
    dti = np.clip(rng.normal(34, 9, n_loans), 5, 65).astype(int)
    orig_upb = np.round(np.clip(rng.lognormal(12.05, 0.48, n_loans), 30_000, 900_000), -3)
    orig_rate = np.clip(4.9 - 0.004 * (credit_score - 725) + rng.normal(0, 0.55, n_loans), 2.0, 11.0)
    term = np.full(n_loans, 360)
    purpose = rng.choice(["P", "C", "N"], n_loans, p=[0.52, 0.31, 0.17])
    occupancy = rng.choice(["O", "I", "S"], n_loans, p=[0.86, 0.10, 0.04])
    state = rng.choice(["CA", "TX", "FL", "NY", "IL", "OH", "PA", "GA", "NC", "MI"], n_loans)

    origination = pd.DataFrame({
        "loan_id": [f"L{i:08d}" for i in range(n_loans)],
        "orig_period": periods[orig_t].astype(str),
        "credit_score": credit_score,
        "orig_ltv": orig_ltv,
        "orig_cltv": np.minimum(orig_ltv + rng.integers(0, 6, n_loans), 105),
        "dti": dti,
        "orig_upb": orig_upb,
        "orig_interest_rate": np.round(orig_rate, 3),
        "orig_term_months": term,
        "loan_purpose": purpose,
        "occupancy_status": occupancy,
        "property_state": state,
        "first_time_buyer": rng.choice(["Y", "N"], n_loans, p=[0.22, 0.78]),
        "num_borrowers": rng.choice([1, 2], n_loans, p=[0.42, 0.58]),
    })

    # risk index, in log-odds space
    lp = (-6.55
          + 0.030 * (720 - credit_score) / 10.0
          + 0.042 * (orig_ltv - 75) / 5.0
          + 0.021 * (dti - 33) / 5.0
          + np.where(occupancy == "I", 0.32, 0.0)
          + np.where(purpose == "C", 0.14, 0.0))
    sqrt_r = np.sqrt(0.15)

    # ---------------- monthly performance ----------------
    rows = []
    monthly_rate = orig_rate / 100.0 / 12.0
    payment = orig_upb * monthly_rate / (1.0 - (1.0 + monthly_rate) ** -term)

    for i in range(n_loans):
        t0 = orig_t[i]
        bal = orig_upb[i]
        delinq = 0
        months_clean = 0
        defaulted_once = False
        hpi0 = hpi_by_t[t0]

        max_age = min(n_periods - t0, 240)
        for age in range(1, max_age):
            t = t0 + age
            bal = max(bal * (1.0 + monthly_rate[i]) - payment[i], 0.0)
            if bal <= 1.0:
                break

            eltv = orig_ltv[i] * (hpi0 / hpi_by_t[t])

            # prepayment: rate incentive + seasoning
            incentive = orig_rate[i] - macro["mortgage_rate_30y"].iloc[t]
            p_prepay = 1.0 / (1.0 + np.exp(-(-5.05 + 0.52 * incentive
                                             + 0.85 * min(age / 24.0, 1.0)
                                             - 0.012 * (eltv - 75))))
            # default hazard
            eta = (lp[i] + _seasoning(np.array([age]))[0]
                   + 0.028 * (eltv - orig_ltv[i]) - sqrt_r * z_by_t[t] * 1.85)
            p_default = 1.0 / (1.0 + np.exp(-eta))

            zb_code = None
            if delinq == 0 and rng.random() < p_prepay:
                zb_code = ZB_PREPAID
            elif rng.random() < p_default:
                delinq = min(delinq + 1, DELINQ_MAX)
                months_clean = 0
            elif delinq > 0:
                if rng.random() < 0.34:                       # cure
                    delinq = 0
                    months_clean = 0
                elif rng.random() < 0.30:                     # roll deeper
                    delinq = min(delinq + 1, DELINQ_MAX)
            else:
                months_clean += 1

            if delinq >= DELINQ_90PLUS:
                defaulted_once = True
                if rng.random() < 0.055:                      # workout completes
                    zb_code = rng.choice([ZB_REO, ZB_SHORT_SALE, ZB_THIRD_PARTY],
                                         p=[0.62, 0.27, 0.11])

            rows.append((origination.loan_id[i], str(periods[t]), age, round(bal, 2),
                         int(delinq), round(float(eltv), 1), zb_code))
            if zb_code is not None:
                break

    performance = pd.DataFrame(rows, columns=[
        "loan_id", "period", "loan_age", "current_upb", "delinq_bucket",
        "estimated_ltv", "zero_balance_code"])

    out = {"origination": origination, "performance": performance, "macro": macro}

    out_dir = Path(out_dir) if out_dir else Path(load("data")["paths"]["raw"]) / "synthetic"
    if not out_dir.is_absolute():
        from hcr.config import REPO_ROOT
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in out.items():
        df.to_parquet(out_dir / f"{name}.parquet", index=False)

    return out


if __name__ == "__main__":
    d = generate()
    perf = d["performance"]
    print(f"loans        {len(d['origination']):>10,}")
    print(f"loan-months  {len(perf):>10,}")
    print(f"ever 90+ DPD {perf.groupby('loan_id').delinq_bucket.max().ge(3).sum():>10,}")
    print(f"periods      {d['macro'].period.min()} .. {d['macro'].period.max()}")
    print("\ndefault rate by year (share of loan-months at 90+ DPD):")
    perf = perf.assign(year=perf.period.str[:4])
    print((perf.groupby("year").delinq_bucket.apply(lambda s: s.ge(3).mean())
           .mul(100).round(2).to_string()))
