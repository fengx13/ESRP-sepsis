# ESRP — Emergency Sepsis Risk Prediction score

## Overview

The Emergency Sepsis Risk Prediction (ESRP) score is an integer score computed from nine variables recorded at emergency department (ED) triage, without laboratory tests. It places adult ED patients in four ordered tiers of sepsis risk. It was derived in 2·5 million encounters at M Health Fairview (University of Minnesota, UMN) and validated without re-estimation at UMN and in two external cohorts: MIMIC-IV-ED (Beth Israel Deaconess Medical Center, BIDMC; 448,804 encounters) and MC-MED (Stanford Health Care; 118,385 encounters).

This repository contains the full specification of the score, every variable and outcome definition used in the paper, the comparator scores, and code that rebuilds both external cohorts from the raw PhysioNet files and reproduces every external-validation number in the paper. It contains no patient data. The UMN data cannot be shared.

Manuscript: *Development and external validation of an interpretable machine learning score for early sepsis identification in the emergency department: a multicenter retrospective cohort study.* eClinicalMedicine (under review).

**Not for clinical use.**

## The ESRP score

| Variable | Points |
|---|---|
| Age, years | 18–29: 0 · 30–69: 9 · ≥70: 14 |
| Chief complaint of fever or chills | 20 |
| Heart rate, beats/min | <90: 0 · 90–109: 3 · ≥110: 6 |
| Systolic blood pressure, mm Hg | <90: 14 · 90–179: 3 · ≥180: 0 |
| Diastolic blood pressure, mm Hg | <60: 11 · 60–89: 6 · ≥90: 0 |
| Respiratory rate, breaths/min | <18: 0 · 18–21: 3 · ≥22: 9 |
| Oxygen saturation, % | <94: 11 · 94–98: 3 · ≥99: 0 |
| Two or more hospitalisations in the previous 365 days | 6 |
| Coded fluid and electrolyte disorder | 9 |

The total ranges from 0 to 100. Risk tiers: low 0–20, moderate 21–30, high 31–44, very high ≥45. Predicted probability = 1 / (1 + exp[−(−7·2541 + 0·111705 × score)]), fitted in the derivation cohort. The mapping under-predicts absolute risk outside UMN, so an adopting site should re-estimate the intercept locally (eTable 8).

```python
from esrp.score import esrp_score, risk_tier, predicted_probability
s = esrp_score({"age": 64, "heart_rate": 104, "systolic_bp": 112, "diastolic_bp": 68,
                "oxygen_saturation": 95, "respiratory_rate": 20, "fever_or_chills": 0,
                "hospitalisations_365d_ge2": 0, "fluid_electrolyte_disorder": 0})
s, risk_tier(s), predicted_probability(s)   # 27, 'moderate', 0.0142
```

An online calculator is in `docs/index.html` (served by GitHub Pages).

## Installation

```bash
git clone https://github.com/fengx13/ESRP-sepsis.git
cd ESRP-sepsis
pip install -e ".[reproduce,test]"
```

The score itself needs only numpy, pandas and scipy (`pip install -e .`). Reproducing the paper also needs scikit-learn and matplotlib. Tested with Python 3.10, numpy 2.2, pandas 2.3, scipy 1.15, scikit-learn 1.7 and matplotlib 3.10.

## Data access

All external-validation data are on PhysioNet under credentialed access: complete the required training, sign each data use agreement, then download.

| Dataset | Version | Used for |
|---|---|---|
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/1.0/) | 1.0 | BIDMC cohort and triage data |
| [MIMIC-IV](https://physionet.org/content/mimiciv/1.0/) | 1.0 | Patients, admissions, prior diagnoses |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) | 2.2 | ED diagnoses, medication dispensing, vital signs (outcomes) |
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | 3.1 | Hospital diagnoses, laboratory, microbiology, medications, ICU infusions (outcomes) |
| [MC-MED](https://physionet.org/content/mc-med/1.0.1/) | 1.0.1 | Stanford cohort |

`docs/provenance.md`, section 1, lists the tables read from each release and explains why the BIDMC outcomes come from later releases. Keep the downloads outside the repository; nothing derived from them is committed (`.gitignore`).

## Reproducing the external validation

Two steps: build each cohort from the raw files, then compute the tables and figures from the cohorts.

```bash
python -m esrp.cohorts.bidmc --ed-v1 <mimic-iv-ed-1.0> --mimic-v1 <mimic-iv-1.0> \
    --ed-v22 <mimic-iv-ed-2.2> --mimic-v31 <mimic-iv-3.1> --cache-dir <cache> \
    --out results/bidmc_cohort.csv.gz
python -m esrp.cohorts.stanford --mcmed <mc-med> --out results/stanford_cohort.csv.gz

python notebooks/table3_external.py results/bidmc_cohort.csv.gz results/stanford_cohort.csv.gz
```

The BIDMC build reads `labevents` (2·6 GB compressed) once and keeps the filtered extracts in `--cache-dir`. Later runs take about a minute; Stanford takes under a minute. The remaining scripts (appendix tables, refitted models, subgroups, figure data) and the published output each one produces are listed in `docs/provenance.md`, section 5.

To check a build against the published numbers:

```bash
ESRP_MCMED=<mc-med> ESRP_MIMIC_ROOT=<folder with the four MIMIC folders> pytest
```

Without these variables, only the unit tests run.

## Repository layout

| Path | Content |
|---|---|
| `esrp/score.py` | The ESRP score: points, tiers, probability mapping |
| `esrp/definitions.py` | ICD-coded sepsis, sepsis mimics, SOFA components, baseline SOFA, Sepsis-3, septic shock, composite outcome |
| `esrp/comorbidity.py` | The Charlson and Elixhauser categories the study uses |
| `esrp/comparators.py` | ESI, qSOFA, NEWS2, MEWS, REMS, SIRS, CART and the frozen transfer models |
| `esrp/evaluation.py` | AUROC with DeLong intervals, AUPRC, bootstrap, Wilson intervals, operating points, calibration, risk tiers, net benefit |
| `esrp/cohorts/` | Builds the BIDMC and Stanford cohorts from the raw PhysioNet files |
| `notebooks/` | Scripts that reproduce the external-validation tables and figures |
| `docs/` | Online calculator (`index.html`) and data provenance (`provenance.md`) |
| `tests/` | Unit tests, and end-to-end checks against the published numbers |
| `results/` | Written by the scripts; not tracked |

## Definitions and provenance

Every variable and outcome is defined in code, with the corresponding appendix table cited in each docstring (eTable 1). `docs/provenance.md` records the data release behind each number, how each site was processed, where the external sites differ from the UMN definitions (and the switch that applies the UMN definition), and which script produces each published table and figure.

## Citation

The citation will be added on publication. Please also cite the datasets: MIMIC-IV-ED, MIMIC-IV and MC-MED, as listed on their PhysioNet pages.

## License, acknowledgements and contact

MIT License (see `LICENSE`). The Charlson and Elixhauser code lists follow Quan et al., Med Care 2005, as implemented in the MIMIC-IV-ED benchmark (Xie et al., Sci Data 2022); the antibiotic list and SOFA conventions follow the MIT-LCP mimic-code repository.

Yanwei (Yana) Jin, Xiaodong (Sheldon) Huang, Yinzhao Wang, David A. Wacker, Michael A. Puskarich, Feng Xie · Dr Feng Xie's Lab, University of Minnesota. Questions and problems: please open an issue on GitHub.
