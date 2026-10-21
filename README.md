# ESRP — Emergency Sepsis Risk Prediction score

An integer score, computed from nine triage variables, that places adult emergency-department (ED) patients in four ordered risk tiers for sepsis. Derived with AutoScore in 3·2 million ED encounters at M Health Fairview (University of Minnesota) and validated externally in MIMIC-IV-ED (Beth Israel Deaconess Medical Center) and MC-MED (Stanford Health Care).

Manuscript: *Development and external validation of an interpretable machine learning score for early sepsis identification in the emergency department: a multicenter retrospective cohort study* (eClinicalMedicine, under revision).

**What is here.** The full specification of the score (variables, cut-points, points, tier boundaries, probability mapping), every outcome and variable definition used in the paper, the comparator scores, and the code that rebuilds the two external validation cohorts from the raw PhysioNet files and reproduces the external validation results.

**What is not here.** No patient data. MIMIC-IV-ED and MC-MED are available on PhysioNet under credentialed access; the UMN data cannot be shared. Scripts write to a data directory outside the repository.

**Not for clinical use.**

## Layout

| Path | Content |
|---|---|
| `esrp/score.py` | The ESRP score: nine variables, cut-points, points, tiers, probability mapping |
| `esrp/definitions/` | Outcome and variable definitions: Sepsis-3 (SOFA, suspected infection), ICD-coded sepsis, sepsis mimics, comorbidity indices |
| `esrp/comparators.py` | qSOFA, NEWS2, SIRS, MEWS, REMS, CART, qSepsis variable set |
| `esrp/cohorts/` | Build the MIMIC-IV-ED and MC-MED cohorts from raw PhysioNet files |
| `esrp/evaluation.py` | AUROC (DeLong), AUPRC, Wilson intervals, calibration, decision curves, risk tiers |
| `notebooks/` | Step-by-step reproduction of the external validation |
| `docs/` | Online calculator (GitHub Pages) and definition tables |
| `tests/` | Unit tests on synthetic rows |
| `results/` | Written by the notebooks; not tracked |

## Team

Yanwei (Yana) Jin, Xiaodong (Sheldon) Huang, Yinzhao Wang, David A. Wacker, Michael A. Puskarich, Feng Xie · Dr. Feng Xie's Lab, University of Minnesota.
