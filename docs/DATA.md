# Data acquisition

Raw data is never committed. This file is the reproduction recipe.

Everything in the repo runs on synthetic data until you complete step 1:

```bash
python -m hcr.data.synthetic
```

---

## 1. Freddie Mac Single-Family Loan-Level Dataset — the mortgage book

**You must do this yourself.** It requires your own registration and licence acceptance.

1. Register (free) via `freddiemac.com/research/datasets/sf-loanlevel-dataset` →
   *Access Historical Data*. This opens Clarity Data Intelligence.
2. Use the **SFLLD Data** tab, top right. **Not CRT Data** — that is credit risk transfer
   deal disclosure, organised by securitisation and starting around 2013, so it has no
   2008–09 downturn.
3. Do **not** click *Full Standard Dataset* — that is all ~49 million loans at once.
4. In the *Standard Dataset Download by Year* table, use the **Sample File** column.
   Each `sample_YYYY.zip` is a 50,000-loan sample from that origination year.
5. Starter set: `sample_2006.zip`, `sample_2007.zip`, `sample_2018.zip`.
6. Unzip into `data/raw/freddie/`, then run:

   ```bash
   python -m hcr.data.freddie
   ```

   It prints the files it found, a data quality report and default rates by year.
7. Once that works, download every sample year from 1999 to 2025 — about 1.35 million
   loans across the full cycle, which is enough for the whole project.

### Documentation — get the July 2026 versions

From the public SFLLD page, under **Recent Disclosure Changes**, download the
**General User Guide**, **File Layout** and **File Headers**, all *Effective July 2026*,
into `docs/freddie/`. Ignore the *Pre-July 2026* versions.

### Release 47 (July 2026) changed the format — what the loader handles

Source: *SFLLD Disclosure Changes, Effective July 2026*, v1.2. Every item below would
silently corrupt results if handled naively, and each has a test in
`tests/test_freddie_loader.py`.

| Change | Risk if ignored | Loader behaviour |
|---|---|---|
| Files renamed to `orig_YYYYQn.txt` / `perf_YYYYQn.txt` | Nothing found, or old files loaded | Pattern discovery; lists folder contents when nothing matches |
| 31 origination / 35 performance fields; fields added and moved | Columns shifted by one — every downstream field wrong | Column-count guard rejects pre-July-2026 files by name |
| **Recoveries now NEGATIVE, costs POSITIVE** | Tutorial code that adds recoveries yields **LGD > 100%** | Normalised into `recovery_*` ≥ 0 and `cost_*` ≥ 0; raises if raw signs look like the old convention |
| Delinquency status `'00'..'99'`, `'RA'`, `'XX'` | `RA` lost; `XX` read as current | `RA` → bucket 99 (triggers default); `XX` → NULL, counted |
| Sentinels: credit score 9999, LTV/CLTV/DTI/MI% 999, etc. | A 9999 credit score lands in the best WOE bin | Mapped to NULL, missingness reported |
| *Total Expenses* is the sum of four components | Adding total and components double counts costs | Reconciliation in the quality report |
| Zero-balance codes 01/02/03/09/15/16/96 | Wrong default definition | Verified list in config; unknown codes raise |

### Zero-balance codes — a correction

The first draft of `config/default_definition.yaml` carried an unverified candidate list:
`02, 03, 09, 15, 96, 97, 98`. Checked against the July 2026 disclosure, it was wrong in
two ways: **97 and 98 do not exist**, and **96 is a repurchase for an underwriting or
servicing defect "prior to credit event"** — a non-credit exit. Counting it as default
would have overstated the default rate.

The verified classification:

| Code | Meaning | Treatment |
|---|---|---|
| 01 | Prepaid or matured | Censored |
| 02 | Third party sale | Default |
| 03 | Short sale or charge-off | Default |
| 09 | REO disposition | Default |
| 15 | Whole loan sale | Default — **judgement**, see below |
| 16 | Reperforming loan securitisation | Censored |
| 96 | Defect repurchase prior to credit event | Censored |

Code 15 is a judgement: Freddie's whole loan sales are predominantly non-performing loan
sales, and EBA/GL/2016/07 treats a sale at a material credit-related loss as unlikeliness
to pay. Check it on real data — if nearly every code-15 exit was already 90+ DPD, the DPD
limb catches them regardless and the choice is immaterial. The `exclude_whole_loan_sales`
DoD variant quantifies it. The config validator now refuses to load any code marked
`non_default_exit` as a UTP trigger.

### Why this dataset

It is the only free source that provides all four of:

1. a loan-month panel long enough for lifetime term structures
2. monthly delinquency status, so 90 DPD is derivable
3. zero-balance and disposition codes, giving real credit events rather than proxies
4. **net sale proceeds, MI and non-MI recoveries, expenses and legal costs** — actual
   cash flows, so workout LGD is estimable rather than assumed

Plus it spans 2008–09 and COVID, which you need for downturn LGD and TTC calibration.

### Size and representativeness

Freddie's own 50,000-loan-per-year samples replace any sampling of ours. Record in the
MDD how Freddie draws them (the General User Guide describes it) and run a
representativeness analysis comparing sample and full-population characteristics by
vintage — that analysis is a required IRB artefact, so the constraint becomes a
deliverable.

---

## 2. SBA 7(a) FOIA data — the SME book

No registration. From `data.sba.gov`, download the FOIA 7(a) dataset to
`data/raw/sba/foia_7a.csv`, then set `sources.sba.enabled: true`.

Useful fields: `GrossApproval`, `SBAGuaranteedApproval`, `ApprovalDate`, `TermInMonths`,
`NaicsCode`, `BusinessAge`, `BusinessType`, `JobsSupported`, `RevolverStatus`,
`LoanStatus` (PIF / CHGOFF / CANCLD), `ChargeOffDate`, `GrossChargeOffAmount`.

**Known limitations — state these in the MDD rather than hoping nobody notices:**

- no monthly balance path, so EAD must come from the contractual amortisation schedule
- the guarantee structure makes "whose loss" ambiguous. Define it explicitly — lender's
  net loss, SBA's loss, or total — and be consistent
- approval-only view, so there is survivorship to think about

Each of these is a Category A margin-of-conservatism item. Quantifying them is better
than having clean data, because it demonstrates the judgement IRB estimation requires.

---

## 3. FRED and FHFA — macro

Get a free FRED API key at `fred.stlouisfed.org/docs/api/api_key.html`. Store it as an
environment variable, never in the repo:

```bash
export FRED_API_KEY="..."
```

Series are listed in `config/data.yaml`: `UNRATE`, `GDPC1`, `MORTGAGE30US`, `CPIAUCSL`,
`FEDFUNDS`, plus state-level unemployment. FHFA house price indices download directly
from `fhfa.gov/hpi` with no key — you need the state-level series for updated LTV, which
drives point-in-time LGD.

---

## On using US data for a UK project

Put this in the MDD's limitations section, and have it ready verbally:

> The methodology is jurisdiction-agnostic; the calibration is not. US data was selected
> because it is the only publicly available source providing loan-level recovery cash
> flows together with a full downturn cycle, both prerequisites for workout LGD and
> downturn estimation. Where UK rules differ materially — parameter floors, exposure
> class definitions, standardised risk weights — the UK treatment is applied. A UK
> implementation would require re-estimation on domestic data but no change to the
> framework.

This is a stronger position than a clean UK synthetic dataset, because synthetic data
cannot produce a defensible LGD and any experienced interviewer knows it.
