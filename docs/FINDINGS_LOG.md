# Findings log

Dated record of what the data showed and what changed as a result. Source material for
the MDD's data, definition-of-default and limitations chapters. Newest entry first.

---

## 2026-09-29 (2) — calibration and the TTC → PIT bridge

**Reproduce:** `python scripts/calibrate.py` (after the sample build and scorecard fit).
Tables and the chart in `outputs/calibration/`. Settings in `config/calibration.yaml` and
`config/moc.yaml`.

### Regulatory PD per grade

Long-run average of each grade's one-year default rates, 1999–2024 (26 years including
2008–09 and COVID), then margin of conservatism, then the UK floor.

| Grade | Long-run average | + MoC C | Floor uplift | **Regulatory PD** |
|---|---|---|---|---|
| 1 | 0.014% | 0.003% | 0.083% | **0.100%** |
| 2 | 0.030% | 0.004% | 0.066% | **0.100%** |
| 3 | 0.065% | 0.006% | 0.029% | **0.100%** |
| 4 | 0.154% | 0.012% | — | **0.167%** |
| 5 | 0.392% | 0.031% | — | **0.423%** |
| 6 | 0.995% | 0.086% | — | **1.081%** |
| 7 | 2.189% | 0.169% | — | **2.358%** |
| 8 | 4.791% | 0.264% | — | **5.055%** |
| 9 | 8.958% | 0.416% | — | **9.374%** |
| 10 | 18.348% | 0.658% | — | **19.006%** |
| 11 | 37.946% | 0.954% | — | **38.900%** |

- **The UK 0.10% floor binds for grades 1–3: 27.8% of observations.** For the best third
  of the book the floor, not the model, sets capital — model improvements there buy nothing.
- **Weighting matters.** Pooling years (total defaults / total observations) gives
  6–35% higher averages than equal weight per year, because crisis years contribute more
  observations to the riskier grades. Equal weight is used - each year is one draw of
  the cycle - and the pooled figures are kept for the MDD.
- Averages were already monotonic across grades; no pooling adjustment was needed.
- **Margin of conservatism.** Category C (bootstrapping whole years, 75th percentile)
  adds 2.5–20% of the estimate, most in the thinnest grades. A1.2 and B.1 are assessed as
  conservative (zero add-on). **A1.1, A2.1 and B.2 are pending quantification** and are
  printed with every calibration run until resolved.

### PIT correlation — the Basel value does not fit the IFRS 9 conversion

If the correlation in the Vasicek transform is right, the economic factor Z implied by
each year's defaults behaves as a standard normal. At the Basel mortgage value of 0.15 it
does not: standard deviation 0.44, mean −0.30. The behavioural scorecard absorbs most of
the cycle through **grade migration** - loans falling behind move to worse grades - so
default rates *within* a grade are far calmer than 0.15 assumes.

| Correlation | Z mean | Z std | Grade-year log-likelihood |
|---|---|---|---|
| **0.031 (estimated: unit-variance Z)** | **−0.04** | **1.00** | **−2,043** |
| 0.15 (Basel capital value) | −0.30 | 0.44 | −2,666 |

The estimate simultaneously gives Z a mean of about zero and standard deviation of one,
and sits on the likelihood plateau (0.01–0.03); the fit deteriorates steeply above it.
**Capital keeps the prescribed 0.15; the IFRS 9 conversion uses 0.031.** Z on a true
standard-normal scale is also what the IFRS 9 macro step will need, to map scenarios
("severe recession ≈ Z of −2") onto PDs.

### The bridge

| | Regulatory PD | IFRS 9 PD | Regulatory ÷ IFRS 9 |
|---|---|---|---|
| Benign years 2000–2006 | | | 1.54–1.84× |
| **2008** | 1.70% | **3.29%** | **0.52×** |
| 2019 (COVID) | 0.69% | 0.83% | 0.83× |
| 2024 | 0.88% | 0.79% | 1.10× |

IFRS 9 PD exceeds regulatory PD in 2007–2012 and 2019, and only marginally (ratios
0.97–0.99) in 2021–23. The relationship reverses in a downturn: regulatory PD is
designed to be stable and conservative; IFRS 9 PD is designed to be unbiased *now*.

**The hybrid rating system, measured.** Across 2000–2024 the portfolio regulatory PD
ranges 0.63%–1.85% (2.9×) — it moves because loans migrate between grades — while the
IFRS 9 PD ranges 0.55%–3.29% (5.9×). A pure through-the-cycle system would show a flat
regulatory line. This is the design question UK hybrid PD models turn on.

**2024 waterfall, portfolio:** regulatory 0.875% → less floor uplift 0.013pp → less MoC C
0.047pp → unbiased long-run average 0.816% → Vasicek adjustment (Z = −0.09) −0.024pp →
**IFRS 9 PD 0.792%**, matching the observed 2024 default rate by construction.

### Capital — placeholder only

With a **placeholder** downturn LGD of 25% (no LGD model yet) the EAD-weighted risk
weight is 20.0%. It demonstrates the pipeline end to end; it is not a result.

### Open items from this step

- Quantify MoC A1.1 (13-month window), A2.1 (modification-as-default variant) and B.2
  (forbearance variant) — each needs a variant development sample.
- Current implied Z uses realised defaults. Forecasting Z from macroeconomic scenarios is
  the IFRS 9 step.

---

## 2026-09-29 — behavioural PD scorecard, first fit

**Reproduce:** `python scripts/build_pd_sample.py` then `python scripts/fit_scorecard.py`.
Tables in `outputs/scorecard/`. Settings in `config/scorecard.yaml`.

### Development sample

Annual December snapshots of performing loans; target = default in the next 12 months.
**5,782,547 snapshots from 1,248,747 loans, 67,399 defaults, one-year default rate
1.166%** — within 0.01pp of the 1.1745% from all monthly observations, so the annual
snapshot design does not distort the default rate. The 2025 vintage contributes nothing
yet: its first December window runs past the data end.

| Split | Definition | Snapshots | Default rate |
|---|---|---|---|
| Development | 1999–2016, 70% of loans | 2,390,630 | 1.41% |
| Out-of-sample | 1999–2016, the other 30% of loans | 1,023,184 | 1.42% |
| Out-of-time | 2017–2024, all loans | 2,368,733 | 0.81% |

The out-of-sample split is by **loan**, via an MD5 hash, so no loan is on both sides and
the split is identical on every machine.

### Driver findings

- **Freddie's estimated current LTV is 100% missing for every snapshot year before
  2017** (84–91% populated after). It would be blank for every development observation,
  so it is excluded. Replacement: current LTV indexed with FHFA state house prices.
- **Binning thresholds.** The textbook 5%-per-bin rule would merge 30- and 60-day arrears
  into "current" and destroy the strongest signal. With millions of observations a 0.5%
  bin holds tens of thousands of loans; bins also need at least 100 defaults.
- **Origination interest rate is partly a proxy for when the loan was written** —
  rank correlation with vintage −0.76 (mean rate ~8% in 2000, under 4% by 2016). It still
  adds 0.007 Gini overall and in 23 of 26 years, including out-of-time, because within a
  year a higher rate reflects risk-based pricing. Kept for now; **to be replaced by the
  spread over the market rate at origination** once the macro series is loaded.

### Driver selection — 24 candidates, 13 selected

| Driver | IV | Outcome |
|---|---|---|
| Worst delinquency, last 12 months | 2.03 | Selected (−0.584) |
| Months delinquent, last 12 months | 1.96 | Dropped — correlation 0.98 with the above |
| Current delinquency | 1.71 | Selected (−0.396) |
| Credit score | 0.81 | Selected (−0.348) |
| Origination rate | 0.65 | Selected (−0.376) — flagged, see above |
| Prior default | 0.36 | Dropped in regression, p = 0.045 — captured by the delinquency history |
| Original term | 0.27 | Selected (−0.295) |
| Original CLTV | 0.24 | Selected (−0.602) |
| Original LTV | 0.23 | Dropped — correlation 0.94 with CLTV |
| DTI | 0.22 | Selected (−0.431) |
| Channel | 0.20 | Selected (−0.124) |
| Months on book | 0.16 | Selected (−0.331) |
| MI cover | 0.12 | Dropped — wrong sign (collinear with CLTV: MI is required above 80%) |
| Modified | 0.10 | Dropped — wrong sign: modification follows arrears already captured; conditional on them it reduces risk |
| Multiple borrowers | 0.10 | Selected (−0.598) |
| State | 0.07 | Selected (−0.850) |
| Loan purpose | 0.07 | Selected (−0.347) |
| Balance outstanding / original | 0.06 | Selected (−0.551) |
| HARP, property type, occupancy, first-time buyer, units, forbearance in last 12m | < 0.02 | Dropped — IV below minimum |

All coefficients negative, as required for a model of default with WOE = ln(good/bad),
and between −0.12 and −0.85. Variance inflation factors 1.0–1.7. The five drivers above
IV 0.5 are flagged for leakage review: the delinquency features were verified as known
at the snapshot date (`tests/test_pd_sample.py`), credit score is from origination, and
the rate is discussed above.

### Discrimination

| Sample | Gini |
|---|---|
| Development | 0.8041 |
| Out-of-sample | 0.8039 |
| **Out-of-time** | **0.8052** |

No overfitting (development equals out-of-sample) and no decay out of time. By snapshot
year the Gini sits between 0.72 and 0.91, with two stories:

- **1999–2000 are weakest (0.72–0.73):** loans with no 12-month history yet, so the
  behavioural drivers have little to say.
- **2019 drops to 0.58 — COVID.** Loans observed in December 2019 defaulted in 2020 for
  reasons none of their characteristics predicted. The model recovers immediately
  (2020: 0.91). This is exactly the failure mode that justifies IFRS 9 forward-looking
  scenarios and post-model adjustments: a behavioural model cannot see an exogenous shock
  coming.

### Grades and raw calibration

Default rates rise monotonically through all 11 master-scale grades, from 0.01% to 40%
(development) and 0.02% to 40% (out-of-time). Effective grades by Herfindahl: 5.7
(development), 5.2 (out-of-time) — the book concentrates in grades 3–6, typical of prime
mortgages.

The raw model PD matches the development default rate exactly (1.41% vs 1.41%, a
property of maximum likelihood). Out-of-time the model predicts 0.89% against 0.81%
observed, and in the middle grades observed rates run at roughly 60–75% of model PD.
That gap is the economic cycle: the model is calibrated to a period that includes 2008,
and 2017–2024 was mostly benign. **Separating that cycle effect out is the job of the
next step — TTC calibration and the Vasicek bridge.**

---

## 2026-09-28 (2) — full sample, all 27 vintages

**Data:** SFLLD Release 47 sample files, vintages 1999–2025. 1,350,000 loans,
74,919,968 loan-months, observation cut-off 2026-03 for every vintage. Reproduce with
`python scripts/vintage_summary.py`; tables below come from `outputs/vintage_summary/`.

### Column mapping

All 31 origination and 35 performance fields match Freddie's official July 2026 header
files, one-to-one and in order.

### Loader checks — passed on every file

| Check | Result |
|---|---|
| Orphans | 5 — loans in the 2025 origination file with no performance yet |
| Recovery sign convention | 40,490 non-zero raw recoveries, 89 positive (0.22%; worst vintage 0.70%) |
| Total expenses vs components | 19,544 rows, all reconcile exactly |
| Delinquency status | 100,736 `RA` rows; 13,627 `XX` rows, treated as missing |
| Loan-age resets | 38,544 loans — confirms calendar-month keying was necessary |
| Missing months | 195 loans |
| Missing values | FICO 0.16%, LTV 0.003%, VantageScore 4.0 100% |
| **DTI missing** | **7.1% overall, 29–35% in the 2010–2013 vintages** — HARP refinances disclose DTI as not available. Missingness is structural and informative: give it its own WOE bin, and consider the HARP indicator as a driver. |

### Zero-balance codes — ever 90+ DPD at or before exit

| Code | Loans | Ever 90+ DPD |
|---|---|---|
| 01 prepaid | 976,210 | 2.4% |
| 02 third party sale | 3,390 | 100% |
| 03 short sale / charge-off | 4,590 | 96.9% |
| 09 REO disposition | 11,885 | 100% |
| 15 whole loan sale | 1,291 | 100% — UTP judgement immaterial across the full sample |
| 16 RPL securitisation | 5,925 | 96.4% |
| 96 defect repurchase | 4,117 | 45.6% |

### Cumulative default rate by vintage

| Vintage | Default rate | | Vintage | Default rate | | Vintage | Default rate |
|---|---|---|---|---|---|---|---|
| 1999 | 3.45% | | 2008 | 9.22% | | 2017 | 3.43% |
| 2000 | 3.49% | | 2009 | 2.93% | | 2018 | 2.97% |
| 2001 | 3.22% | | 2010 | 2.90% | | 2019 | 2.76% |
| 2002 | 3.37% | | 2011 | 2.50% | | 2020 | 1.42% |
| 2003 | 4.03% | | 2012 | 2.51% | | 2021 | 1.69% |
| 2004 | 7.01% | | 2013 | 2.79% | | 2022 | 3.10% |
| 2005 | 10.94% | | 2014 | 2.79% | | 2023 | 2.18% |
| 2006 | 14.34% | | 2015 | 2.55% | | 2024 | 1.05% |
| **2007** | **16.37%** | | 2016 | 2.75% | | 2025 | 0.11% |

Recent vintages are not comparable to old ones - a 2025 loan has had at most 15 months
to default. Compare vintages at equal seasoning before drawing conclusions.

### One-year default rate by observation year — the cycle

| Year | Rate | | Year | Rate | | Year | Rate |
|---|---|---|---|---|---|---|---|
| 1999 | 0.30% | | 2008 | 2.57% | | 2017 | 0.82% |
| 2000 | 0.69% | | **2009** | **3.32%** | | 2018 | 0.61% |
| 2001 | 0.92% | | 2010 | 2.37% | | 2019 | 0.62% |
| 2002 | 1.00% | | 2011 | 2.04% | | 2020 | 1.11% |
| 2003 | 0.95% | | 2012 | 1.64% | | 2021 | 1.25% |
| 2004 | 0.77% | | 2013 | 1.26% | | 2022 | 0.67% |
| 2005 | 0.73% | | 2014 | 0.95% | | 2023 | 0.69% |
| 2006 | 0.70% | | 2015 | 0.77% | | 2024 | 0.81% |
| 2007 | 1.13% | | 2016 | 0.72% | | 2025 | 0.79% |

**Long-run average one-year default rate: 1.17% pooled, 1.12% equal weight per year.**
The weighting choice is an MDD judgement: equal weighting treats each year as one draw
of the cycle; pooling weights years by portfolio size. Peak 3.32% (2009) against ~0.6% in
benign years (2018–19): a cycle amplitude of about 5.4x. 1999 is excluded as a trough
because it contains only unseasoned 1999 originations.

### Post-forbearance defaults — a judgement for the MDD

In the 2011–2019 vintages roughly two-thirds of defaulted loans were at some point in
forbearance. Because the DPD limb is suspended during a plan, these defaults trigger
when a plan **ends** with the loan still 90+ DPD. On the 2016–2019 vintages:

| Default events | Count | Cured within 6 months | Payment deferral booked around the default |
|---|---|---|---|
| Within 3 months after a forbearance plan ended | 4,019 | 46.4% | 35.2% |
| Other | 3,769 | 38.8% | 5.4% |

1,330 post-forbearance events cure within 6 months **with** a deferral booked — consistent
with a gap between the plan ending and Freddie booking the deferral. Treating those as
non-defaults would lower cumulative default rates by about 0.5pp:

| Vintage | As now | Excluding those 1,330 events |
|---|---|---|
| 2016 | 2.75% | 2.31% |
| 2017 | 3.43% | 2.83% |
| 2018 | 2.97% | 2.46% |
| 2019 | 2.76% | 2.16% |

Two defensible positions:

- **Keep as defaults (current baseline).** Conservative. Also arguably what EBA/GL/2016/07
  requires anyway: an interest-free deferral of arrears to maturity is a concession, and
  if the NPV loss exceeds 1% it is a distressed restructuring - itself a default trigger.
- **Grace period after a plan ends.** Treats the deferral as returning the loan to its
  revised schedule, as with qualifying moratoria. Lower, less conservative, and needs its
  own justification.

Recommendation: keep the baseline for the regulatory definition, add a post-forbearance
grace-period variant as a sensitivity, and carry the difference as margin of conservatism.

### Other observations

- **Re-defaults outnumber first defaults from 2014 onwards** in the pooled data - the
  prior-default segment for the PD model is confirmed as necessary.
- **The 2022 vintage (3.10%) is running well above 2020–21** at similar or shorter
  seasoning. Worth checking against origination characteristics before modelling.

---

## 2026-09-28 — first run on real data

**Data:** Freddie Mac SFLLD Release 47 (July 2026 format), sample files for vintages
2006, 2007, 2018. 150,000 loans, 8,275,124 loan-months, reporting periods 2006-01 to
2026-03.

### Loader checks — all passed

| Check | Result |
|---|---|
| Column counts | 31 origination / 35 performance, every file |
| Loan identifiers | All match `PYYQnXXXXXXX`; no orphans either way |
| Recovery sign convention | 16,518 non-zero raw recoveries, 25 positive (0.15%) — Release 47 convention confirmed |
| Zero-balance codes | Only 01, 02, 03, 09, 15, 16, 96 present |
| Total expenses vs components | 7,886 rows, all reconcile exactly — summing both would double count |
| Delinquency status | 40,630 `RA` rows; no `XX` rows |
| Missing values | Credit score 0.07%, LTV 0.01%, DTI 1.8%; **VantageScore 4.0 100% missing** for these vintages — not usable as a driver |

### Zero-balance codes: was the loan ever 90+ DPD at or before exit?

| Code | Loans | Ever 90+ DPD | Consequence |
|---|---|---|---|
| 01 prepaid | 126,027 | 3.8% | Censored |
| 02 third party sale | 1,115 | 100% | Default |
| 03 short sale / charge-off | 2,217 | 96.8% | Default — UTP adds the other 3.2% |
| 09 REO disposition | 4,816 | 100% | Default |
| 15 whole loan sale | 476 | **100%** | UTP judgement is **immaterial** — the DPD limb catches every one |
| 16 RPL securitisation | 2,154 | 96.5% | Censored at exit; their earlier default episodes are counted |
| 96 defect repurchase | 860 | 73.6% | The borrower defaults are counted via DPD; UTP correctly adds none |

### Defects found in the panel / DoD code — all fixed, each with a regression test

1. **Loan age resets on modification** (e.g. 48 → 6 with a 6.375% → 2.0% rate cut and
   term extension). 6,693 loans affected, essentially every modified loan. The panel was
   ordered by loan age, which would have scrambled the history of the most distressed
   loans. Re-keyed on calendar month; `months_on_book` added as the non-resetting
   seasoning variable.

2. **Observation rule understated the one-year default rate by 43%.** Requiring 12 rows
   ahead dropped every month before a prepayment (1.44m observations) and every month
   before a fast default-and-exit (~105k observations, all defaults: REO, short sale,
   third party sale, NPL sale). Replaced with the cohort rule: window must end by the
   last data month, and the loan is either observed through it or exits inside it.

   | Rule | Observations | Defaults | One-year default rate |
   |---|---|---|---|
   | Old — 12 rows ahead | 6,001,813 | 109,059 | 1.82% |
   | New — cohort rule | 7,585,586 | 241,871 | **3.19%** |

3. **Forbearance was counted as default** although the config said otherwise — the
   setting was never implemented. Arrears keep accruing under a Freddie forbearance plan;
   85% of 2020 default events were in forbearance. The earlier config also had the
   direction wrong (claimed forbearance *suppressed* defaults). Now the DPD limb is
   suspended while a plan is active; UTP still applies.
   2018 vintage cumulative default: **5.42% → 2.97%**. Loans never in forbearance: 0.86%.

4. **Probation after distressed restructuring was 3 months; EBA/GL/2016/07 requires
   12.** Applied to any loan modified at or before the cure. Re-default events in 2016
   fell from 484 to 365.

5. Also fixed: the `continuation` re-default variant measured from the latest
   post-probation month (always ~1) and so merged every re-default regardless of the
   window; and the data-end rule admitted exits but not survivors near the end of the
   data, making the last year a sample of loans that left.

### Results after fixes (baseline definition of default)

| Vintage | Loans | Ever defaulted | Cumulative default rate |
|---|---|---|---|
| 2006 | 50,000 | 7,168 | 14.3% |
| 2007 | 50,000 | 8,185 | 16.4% |
| 2018 | 50,000 | 1,487 | 3.0% |

One-year default rate peaks at 5.9% (2009 observation year). 62 loans have a single
missing reporting month; the calendar-month RANGE window handles them.

### Open items

- **Post-forbearance defaults.** 2021 shows 577 first defaults, mostly loans still 90+
  DPD when their plan ended. Some may be administrative lag before a payment deferral was
  booked. Quantify before the Category B MoC assessment.
- **Re-defaults exceed first defaults from 2014 to 2018** even with 12-month probation —
  legacy modified loans. The PD model needs a prior-default / restructured segment or
  driver; a single population model would blend two very different risk profiles.
