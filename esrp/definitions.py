"""Outcome and variable definitions used in the ESRP study, as pure functions.

Everything here is the operational definition used for the published external
validation (BIDMC = MIMIC-IV-ED, Stanford = MC-MED); see eTable 1 of the appendix.
ICD codes are matched on dot-free, upper-case prefixes so that MIMIC (no dots)
and MC-MED (dots) are handled identically.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ICD-coded sepsis (eTable 1, Panel B). Any diagnosis position, ED or hospital.
# ---------------------------------------------------------------------------
SEPSIS_ICD9 = ("038", "99591", "99592", "78552", "6702")
SEPSIS_ICD10 = ("A40", "A41", "R6520", "R6521", "R572", "T8144", "O85", "O0337", "O0387", "O0487")

# Sepsis mimics (eTable 1, Panel B): five non-infectious conditions with overlapping presentations.
MIMIC_ICD = {
    "dka":               (("2501",), ("E081", "E091", "E101", "E111", "E131")),
    "pancreatitis":      (("5770",), ("K85",)),
    "cardiogenic_shock": (("78551",), ("R570",)),
    "pulmonary_embolism": (("41511", "41512", "41513", "41519", "4162"), ("I26",)),
    "copd_asthma":       (("49121", "49122", "493"), ("J441", "J45")),
}


def normalise_icd(codes: pd.Series) -> pd.Series:
    """Upper-case, strip dots and whitespace."""
    return codes.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()


def icd_matches(codes: pd.Series, version: pd.Series | None, prefixes9, prefixes10) -> pd.Series:
    """Boolean mask: code starts with any prefix of its ICD version.

    If ``version`` is None the code is matched against both lists (MC-MED stores
    ICD-9 and ICD-10 in separate columns, so pass each column with its own prefixes).
    """
    c = normalise_icd(codes)
    if version is None:
        return c.str.startswith(prefixes9) | c.str.startswith(prefixes10)
    v = pd.to_numeric(version, errors="coerce")
    return ((v == 9) & c.str.startswith(prefixes9)) | ((v == 10) & c.str.startswith(prefixes10))


def is_sepsis_icd(codes: pd.Series, version: pd.Series | None = None) -> pd.Series:
    return icd_matches(codes, version, SEPSIS_ICD9, SEPSIS_ICD10)


def mimic_flags(codes: pd.Series, version: pd.Series | None = None) -> pd.DataFrame:
    """One boolean column per mimic category, aligned with ``codes``."""
    return pd.DataFrame({k: icd_matches(codes, version, p9, p10) for k, (p9, p10) in MIMIC_ICD.items()})


# ---------------------------------------------------------------------------
# Sepsis-3 component (eTable 1, Panel C).
# Suspected infection: a body-fluid culture ordered within 72 h before an antibiotic,
# or an antibiotic given within 24 h before a culture; time of suspicion = the earlier of the pair.
# SOFA window: 48 h before to 24 h after the time of suspicion (Seymour 2016).
# ---------------------------------------------------------------------------
CULTURE_BEFORE_ANTIBIOTIC_H = 72
ANTIBIOTIC_BEFORE_CULTURE_H = 24
SOFA_WINDOW_H = (-48, 24)
DELTA_SOFA_THRESHOLD = 2

# Organ systems scored at the external sites (laboratory results available) and at UMN (vital signs and medications only).
SOFA_SYSTEMS_EXTERNAL = ("liver", "renal", "coagulation", "cardiovascular")
SOFA_SYSTEMS_UMN = ("cardiovascular", "cns", "respiratory")


def sofa_coagulation(platelets_min) -> np.ndarray:
    """Platelets ×10³/µL: <20 → 4, <50 → 3, <100 → 2, <150 → 1."""
    p = np.asarray(platelets_min, dtype=float)
    return np.select([p < 20, p < 50, p < 100, p < 150], [4, 3, 2, 1], 0)


def sofa_liver(bilirubin_max) -> np.ndarray:
    """Bilirubin mg/dL: ≥12 → 4, ≥6 → 3, ≥2 → 2, ≥1.2 → 1."""
    b = np.asarray(bilirubin_max, dtype=float)
    return np.select([b >= 12, b >= 6, b >= 2, b >= 1.2], [4, 3, 2, 1], 0)


def sofa_renal(creatinine_max) -> np.ndarray:
    """Creatinine mg/dL: ≥5 → 4, ≥3.5 → 3, ≥2 → 2, ≥1.2 → 1."""
    c = np.asarray(creatinine_max, dtype=float)
    return np.select([c >= 5, c >= 3.5, c >= 2, c >= 1.2], [4, 3, 2, 1], 0)


def sofa_cardiovascular(map_min, on_vasopressor) -> np.ndarray:
    """Any vasopressor → 3 (dose not available in the extracts); else MAP <70 mm Hg → 1."""
    m = np.asarray(map_min, dtype=float)
    v = np.asarray(on_vasopressor, dtype=float)
    return np.where(v == 1, 3, np.where(m < 70, 1, 0))


def baseline_sofa(comorbidity: pd.DataFrame) -> pd.DataFrame:
    """Baseline organ dysfunction from comorbidity coding in the previous five years (eTable 1, Panel C).

    Expects Charlson/Elixhauser indicator columns (Quan 2005 mapping) named
    cci_Liver1, cci_Liver2, cci_Renal, eci_Renal, eci_Coagulopathy, cci_CHF, eci_CHF;
    missing columns count as 0.
    """
    def col(n):
        return comorbidity[n].fillna(0) if n in comorbidity.columns else pd.Series(0, index=comorbidity.index)
    b = pd.DataFrame(index=comorbidity.index)
    b["liver"] = np.where(col("cci_Liver2") == 1, 2, np.where(col("cci_Liver1") == 1, 1, 0))
    b["renal"] = np.where((col("cci_Renal") == 1) | (col("eci_Renal") == 1), 2, 0)
    b["coagulation"] = np.where(col("eci_Coagulopathy") == 1, 1, 0)
    b["cardiovascular"] = np.where((col("cci_CHF") == 1) | (col("eci_CHF") == 1), 1, 0)
    return b


def sepsis3(suspected_infection, sofa_abs: pd.DataFrame, baseline: pd.DataFrame,
            systems=SOFA_SYSTEMS_EXTERNAL) -> pd.Series:
    """Sepsis-3: suspected infection and a rise of ≥2 in SOFA over baseline, summed over ``systems``.

    The rise is taken per system as max(0, absolute − baseline), then summed.
    """
    delta = sum(np.maximum(0, sofa_abs[s].fillna(0).values - baseline[s].fillna(0).values) for s in systems)
    return pd.Series(np.asarray(suspected_infection, dtype=bool) & (delta >= DELTA_SOFA_THRESHOLD), index=sofa_abs.index)


def septic_shock(sepsis3_flag, on_vasopressor, lactate_max) -> pd.Series:
    """Sepsis-3 plus vasopressor requirement with lactate above 2 mmol/L."""
    return (pd.Series(sepsis3_flag).astype(bool)
            & (pd.Series(on_vasopressor).fillna(0).astype(int) == 1)
            & (pd.Series(lactate_max).fillna(0) > 2.0))


def composite_outcome(sepsis_icd, sepsis3_flag) -> pd.Series:
    """The study outcome: ICD-coded sepsis or Sepsis-3 criteria during the index encounter."""
    return pd.Series(sepsis_icd).astype(bool) | pd.Series(sepsis3_flag).astype(bool)
