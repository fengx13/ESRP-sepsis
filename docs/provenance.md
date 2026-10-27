# Provenance: how the published external-validation numbers were produced

## 1. Data releases

The published external validation combined several PhysioNet releases. The code reads each table from the release listed here; this is what reproduces the published numbers exactly.

| Site | Part | Release | Tables |
|---|---|---|---|
| BIDMC | Cohort, triage, ESRP predictors | MIMIC-IV-ED v1.0; MIMIC-IV v1.0 | `ed/edstays`, `ed/triage`; `core/patients`, `core/admissions`, `hosp/diagnoses_icd` |
| BIDMC | Outcomes (ICD-coded sepsis, Sepsis-3, septic shock) | MIMIC-IV-ED v2.2; MIMIC-IV v3.1 | `ed/diagnosis`, `ed/pyxis`, `ed/vitalsign`; `hosp/diagnoses_icd`, `hosp/labevents`, `hosp/microbiologyevents`, `hosp/emar`, `hosp/prescriptions`, `icu/inputevents` |
| Stanford | Everything | MC-MED v1.0.0 (identical to v1.0.1 except for re-generated MRNs) | `visits`, `pmh`, `orders`, `labs`, `numerics` |

The BIDMC cohort is the 448,804 adult stays of MIMIC-IV-ED v1.0. Its outcomes were taken from later releases, joined on `stay_id`, `hadm_id` and `subject_id`, which are stable across releases. Using the v1.0 diagnosis tables instead gives 12,024 ICD-coded sepsis stays rather than 11,421.

The Stanford analysis ran on MC-MED v1.0.0, the version provided to reviewers before public release. The public release, v1.0.1, which the manuscript cites, differs only in that "a small number of de-identified MRNs were re-generated" (v1.0.1 release notes). MRNs are used here only to link a patient's earlier visits and problem-list entries. SHA-256 checksums of the v1.0.0 files used:

| File | SHA-256 |
|---|---|
| `visits.csv` | `ae0027dc0c1dab1b8010e1d28fa84f6635c0d8e8dd14e5631b77940488efddf2` |
| `pmh.csv` | `c5d39914c19c9666aab7796107506e7fc3a7b6c5657ef7036ac72fbe6efa2e86` |
| `orders.csv` | `d4521018baac4bb85d4a15098bb176a2aeaace81e042abbad0d27a1f036822d9` |
| `labs.csv` | `254ab541f9dc61c175c177dc8d53b9839a37b7f8cb6ef64a200574ba32c31611` |

## 2. BIDMC processing

**Cohort.** All MIMIC-IV-ED v1.0 stays with age ≥18 years, where age = year of arrival − `anchor_year` + `anchor_age` (448,804 stays). No other exclusion.

**Triage vital signs.** As in the MIMIC-IV-ED benchmark (Xie et al., Sci Data 2022): temperature converted from Fahrenheit to Celsius; a value outside the outlier bounds set to missing and a value between a valid and an outlier bound clipped to the valid bound (temperature 14·2/26/45/47 °C; heart rate 0/0/350/390; respiratory rate 0/0/300/330; SpO2 0/0/100/150; systolic and diastolic pressure 0/0/375/375; acuity 1/1/5/5); missing values then filled with the median of the 448,804 stays (temperature 36·67 °C, heart rate 84, respiratory rate 18, SpO2 99, systolic 133, diastolic 77, acuity 3).

**Fever or chills.** The chief complaint contains "fever" or "chill", case-insensitive.

**Hospitalisations in the previous 365 days.** Admissions in `core/admissions` with 0 < arrival − `admittime` ≤ 365 days; observation admissions included.

**Comorbidity.** Charlson and Elixhauser categories (Quan 2005 code lists as implemented in the benchmark's `medcodes` package) from `hosp/diagnoses_icd` of admissions discharged in the 1,780 days before arrival.

**Outcomes.**

- ICD-coded sepsis is a sepsis code in any position of the ED diagnoses (`stay_id`) or the hospital diagnoses (`hadm_id`).
- Suspected infection is a culture and a systemic antibiotic paired within 72 h (culture first) or 24 h (antibiotic first). Antibiotics come from `emar` (doses given), `prescriptions` for patients absent from `emar`, and ED `pyxis`. Cultures are any `microbiologyevents` specimen. The earliest suspicion between 24 h before arrival and ED departure counts.
- SOFA uses the worst platelet count, bilirubin, creatinine and lactate (`labevents`), the lowest MAP from ED vital signs (artefacts outside 20–200 mm Hg removed), and any vasopressor in `icu/inputevents`, all from 48 h before to 24 h after suspicion.
- Sepsis-3, septic shock and the composite outcome follow eTable 1, Panel C.

## 3. Stanford processing

**Cohort.** All 118,385 MC-MED visits (all aged ≥18 years).

**Triage vital signs.** `Triage_*` columns of `visits.csv` (temperature already in Celsius), acuity from the leading digit of `Triage_acuity`; no outlier step; missing values filled with the median of the 118,385 visits.

**Fever or chills.** The chief complaint contains "fever", case-insensitive; chills alone not counted (485 visits).

**Hospitalisations in the previous 365 days.** Earlier ED visits of the same MRN arriving in the 365 days before, with `ED_dispo` of Inpatient, ICU or Observation.

**Comorbidity.** The same code lists, applied to every `pmh.csv` problem-list entry noted before arrival, without a time limit; MC-MED has no discharge diagnoses for earlier admissions.

**Outcomes.**

- ICD-coded sepsis is a sepsis code in the visit's `Dx_ICD9` or `Dx_ICD10`.
- Suspected infection uses the same pairing rule, applied to the visit's own orders (`Microbiology` and `Microbiology Culture` order types; antibiotics by order name, timed at first administration, else at order).
- SOFA uses `labs.csv` (platelets, bilirubin, creatinine, lactate) and `numerics.csv` (lowest charted MAP, else (SBP + 2 DBP)/3, artefacts removed), with vasopressors identified as continuous intravenous infusions by order name.

## 4. Site-specific definitions and switches

Three variables were derived at the external sites from what each source records (eTable 1, Panel A, of the appendix). The code uses these definitions by default, which gives the published numbers; each can be switched to the UMN definition.

| Item | UMN definition | BIDMC | Stanford | AUROC with the UMN definition | Switch |
|---|---|---|---|---|---|
| Hospitalisations | Excluding observation stays | Observation included | Observation included | BIDMC 0·784 → 0·789; Stanford 0·822 → 0·823 | `exclude_observation` |
| Comorbidity window | Five years | 1,780 days (the benchmark's 356 × 5) | All earlier problem-list entries | Stanford 0·822 → 0·823 | `comorbidity_lookback_days` |
| Fever or chills | "fever" or "chills" | "fever" or "chill" | "fever" only (485 chills-only visits not scored) | Stanford 0·822 → 0·823 | `fever_pattern` |

With both Stanford switches aligned, the AUROC is 0·824. With a five-year window, the Stanford baseline SOFA also changes: Sepsis-3 3,925 and composite 4,335, instead of 3,809 and 4,228.

**Outcome search restricted to MIMIC-IV-ED v2.2 patients.** The published run searched `emar`, `prescriptions` and `microbiologyevents` only for patients who also appear in MIMIC-IV-ED v2.2, because those extracts were cached from the v2.2 cohort. Two v1.0 stays of patients absent from v2.2 therefore lost their suspected infection, and one of them its Sepsis-3 label. `bidmc.add_outcomes(match_published=True)` (the default) reproduces this; `--all-patients` searches every patient and gives Sepsis-3 17,306 and composite 23,650, instead of 17,305 and 23,649.

## 5. From raw files to the published numbers

**Step 1: build the cohorts** (one row per ED encounter, with predictors, ESRP score, outcomes and comparator inputs):

```bash
python -m esrp.cohorts.bidmc --ed-v1 <mimic-iv-ed-1.0> --mimic-v1 <mimic-iv-1.0> \
    --ed-v22 <mimic-iv-ed-2.2> --mimic-v31 <mimic-iv-3.1> --cache-dir <cache> \
    --out results/bidmc_cohort.csv.gz
python -m esrp.cohorts.stanford --mcmed <mc-med-1.0.0> --out results/stanford_cohort.csv.gz
```

**Step 2: compute the tables and figures:**

```bash
B=results/bidmc_cohort.csv.gz; S=results/stanford_cohort.csv.gz
python notebooks/table3_external.py     $B $S
python notebooks/appendix_external.py   $B $S
python notebooks/appendix_refitted.py   $B BIDMC    --cache <cache>
python notebooks/appendix_refitted.py   $S Stanford --cache <cache>
python notebooks/appendix_subgroups.py  $B $S
python notebooks/figures_external.py    $B $S
```

| Published output (BIDMC and Stanford columns) | Produced by |
|---|---|
| Cohort sizes and outcome counts (Figure 1, Table 3) | `esrp.cohorts.bidmc`, `esrp.cohorts.stanford` |
| Table 3, Panels A and B | `table3_external.py` |
| eTable 2, Panel A, fixed and frozen models | `appendix_external.py` |
| eTable 2, Panel A, models refitted within the cohort | `appendix_refitted.py` |
| eTable 2, Panel B; eTable 5, Panel A; eTable 7, Panel A; eTable 8 | `appendix_external.py` |
| eTable 9, Panels C and D | `appendix_subgroups.py` |
| Figure 2, Panels B and C; eFigure 2; eFigure 3, Panels B and C | `figures_external.py` |

Each script prints its rows in the format of the manuscript and writes a CSV to `results/`. The two end-to-end tests (`tests/test_bidmc_cohort.py`, `tests/test_stanford_cohort.py`) check the cohort counts and the ESRP AUROC and AUPRC against the published values.

**Not reproduced here.** Everything from the UMN cohorts, whose data cannot be shared; eTable 2, Panel C (the Epic Sepsis Model is recorded only at UMN); Table 1; eTables 3, 5B–E and 10; and eFigure 4, which need treatment and mortality extracts not included in this package.
