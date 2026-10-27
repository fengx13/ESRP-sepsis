"""Comparator scores and the frozen transfer models used in the ESRP study.

All bedside scores are computed from the first triage values. Components that a
site did not record are omitted and the omission is stated in the manuscript:
qSOFA has no mentation item at BIDMC or Stanford; SIRS has no white-cell item at
UMN; NEWS2 at BIDMC and Stanford was computed from four components (respiratory
rate, temperature, systolic pressure, heart rate), as implemented in the
MIMIC-IV-ED benchmark. Temperatures are in °C.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _f(x):
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)


def esi_reversed(triage_acuity):
    """Emergency Severity Index with the sign reversed, so that higher means sicker."""
    return -_f(triage_acuity)


def qsofa(resp_rate, systolic_bp, gcs=None):
    """qSOFA: respiratory rate ≥22, systolic pressure ≤100, altered mentation (GCS <15) when recorded."""
    rr, sbp = _f(resp_rate), _f(systolic_bp)
    s = (rr >= 22).astype(float) + (sbp <= 100).astype(float)
    if gcs is not None:
        s += (_f(gcs) < 15).astype(float)
    return s


def sirs(temp_c, heart_rate, resp_rate, wbc=None):
    """SIRS criteria: temperature >38 or <36 °C, heart rate >90, respiratory rate >20,
    white-cell count >12 or <4 ×10³/µL when available (missing WBC scores 0)."""
    t, hr, rr = _f(temp_c), _f(heart_rate), _f(resp_rate)
    s = ((t > 38) | (t < 36)).astype(float) + (hr > 90).astype(float) + (rr > 20).astype(float)
    if wbc is not None:
        w = _f(wbc)
        s += np.where(np.isnan(w), 0.0, ((w > 12) | (w < 4)).astype(float))
    return s


NEWS2_COMPONENTS_ALL = ("resp_rate", "spo2", "temp", "systolic_bp", "heart_rate", "consciousness")
NEWS2_COMPONENTS_BENCHMARK = ("resp_rate", "temp", "systolic_bp", "heart_rate")


def news2(resp_rate, temp_c, systolic_bp, heart_rate, spo2=None, altered_consciousness=None,
          components=NEWS2_COMPONENTS_ALL):
    """NEWS2 (Royal College of Physicians 2017, SpO2 scale 1, no supplemental-oxygen item).

    Pass ``components=NEWS2_COMPONENTS_BENCHMARK`` to reproduce the four-component
    version used for BIDMC and Stanford in the published tables.
    """
    rr, t, sbp, hr = _f(resp_rate), _f(temp_c), _f(systolic_bp), _f(heart_rate)
    s = np.zeros(len(rr))
    if "resp_rate" in components:
        s += np.select([rr <= 8, rr <= 11, rr <= 20, rr <= 24], [3, 1, 0, 2], 3)
    if "temp" in components:
        s += np.select([t <= 35, t <= 36, t <= 38, t <= 39], [3, 1, 0, 1], 2)
    if "systolic_bp" in components:
        s += np.select([sbp <= 90, sbp <= 100, sbp <= 110, sbp <= 219], [3, 2, 1, 0], 3)
    if "heart_rate" in components:
        s += np.select([hr <= 40, hr <= 50, hr <= 90, hr <= 110, hr <= 130], [3, 1, 0, 1, 2], 3)
    if "spo2" in components and spo2 is not None:
        o = _f(spo2)
        s += np.select([o <= 91, o <= 93, o <= 95], [3, 2, 1], 0)
    if "consciousness" in components and altered_consciousness is not None:
        s += 3 * _f(altered_consciousness)
    return s


def mews(systolic_bp, heart_rate, resp_rate, temp_c):
    """Modified Early Warning Score (Subbe 2001) without the AVPU item, which triage does not record."""
    sbp, hr, rr, t = _f(systolic_bp), _f(heart_rate), _f(resp_rate), _f(temp_c)
    return (np.select([sbp <= 70, sbp <= 80, sbp <= 100, sbp <= 199], [3, 2, 1, 0], 2)
            + np.select([hr <= 40, hr <= 50, hr <= 100, hr <= 110, hr <= 129], [2, 1, 0, 1, 2], 3)
            + np.select([rr < 9, rr <= 14, rr <= 20, rr <= 29], [2, 0, 1, 2], 3)
            + np.select([t < 35, t < 38.5], [2, 0], 2))


def mean_arterial_pressure(systolic_bp, diastolic_bp):
    return _f(systolic_bp) / 3 + 2 * _f(diastolic_bp) / 3


def rems(age, mean_ap, heart_rate, resp_rate, spo2):
    """Rapid Emergency Medicine Score (Olsson 2004), with the Glasgow Coma Scale item omitted.

    Bins are applied to continuous values (the MIMIC-IV-ED benchmark implementation
    uses integer bin edges, which leaves gaps for a non-integer mean arterial pressure).
    """
    a, m, hr, rr, o = _f(age), _f(mean_ap), _f(heart_rate), _f(resp_rate), _f(spo2)
    return (np.select([a < 45, a < 55, a < 65, a < 75], [0, 2, 3, 5], 6)
            + np.select([m > 159, m >= 130, m >= 110, m >= 70, m >= 50], [4, 3, 2, 0, 2], 4)
            + np.select([hr > 179, hr >= 140, hr >= 110, hr >= 70, hr >= 55, hr >= 40], [4, 3, 2, 0, 2, 3], 4)
            + np.select([rr > 49, rr >= 35, rr >= 25, rr >= 12, rr >= 10, rr >= 6], [4, 3, 1, 0, 1, 2], 4)
            + np.select([o < 75, o <= 85, o <= 89], [4, 3, 1], 0))


def cart(age, resp_rate, heart_rate, diastolic_bp):
    """Cardiac Arrest Risk Triage score (Churpek 2012)."""
    a, rr, hr, dbp = _f(age), _f(resp_rate), _f(heart_rate), _f(diastolic_bp)
    return (np.select([a < 55, a < 70], [0, 4], 9)
            + np.select([rr < 21, rr <= 23, rr <= 25, rr <= 29], [0, 8, 12, 15], 22)
            + np.select([hr < 110, hr <= 139], [0, 4], 13)
            + np.select([dbp > 49, dbp >= 40, dbp >= 35], [0, 4, 6], 13))


# ---------------------------------------------------------------------------
# Frozen transfer models (eTable 2, Panel D): logistic models fitted once in the
# UMN derivation cohort and applied unchanged. Coefficients on the raw scale.
# Continuous variables are imputed at the derivation-cohort medians before use.
# ---------------------------------------------------------------------------
TRANSFER_MEDIANS = {"age": 47.4, "resp_rate": 18.0, "heart_rate": 80.0, "temp_c": 36.72, "abdominal_pain": 0.0,
                    "fever_or_chills": 0.0, "male": 0.0, "triage_acuity": 3.0, "spo2": 98.0, "systolic_bp": 129.0,
                    "diastolic_bp": 78.0, "hospitalisations_365d_ge2": 0.0, "fluid_electrolyte_disorder": 0.0}
TRANSFER_MODELS = {
    "qsepsis_10": (6.7749614, {"age": .0360216, "resp_rate": .0300349, "heart_rate": .0142306, "temp_c": -.1402717,
                               "abdominal_pain": .4412816, "fever_or_chills": 2.4251087, "male": .2013179,
                               "triage_acuity": -1.0555912, "spo2": -.0474856, "systolic_bp": -.0150395}),
    "qsepsis_without_acuity": (5.8703406, {"age": .0406189, "resp_rate": .0369764, "heart_rate": .0171458,
                                           "temp_c": -.1827653, "abdominal_pain": .3249765, "fever_or_chills": 2.4382979,
                                           "male": .2717385, "spo2": -.0539922, "systolic_bp": -.0187579}),
    "esrp_variables_continuous": (-0.3160914, {"age": .0283663, "systolic_bp": -.0045296, "diastolic_bp": -.0294096,
                                               "heart_rate": .0171044, "spo2": -.0514598, "resp_rate": .0355367,
                                               "fever_or_chills": 2.2431767, "hospitalisations_365d_ge2": .5165796,
                                               "fluid_electrolyte_disorder": .9525290}),
}


def transfer_model(df: pd.DataFrame, name: str, columns: dict | None = None) -> np.ndarray:
    """Predicted probability from a frozen transfer model.

    ``columns`` maps the variable names above to the column names in ``df``.
    Missing values are imputed at the derivation-cohort medians.
    """
    intercept, coefs = TRANSFER_MODELS[name]
    col = {k: k for k in coefs}
    if columns:
        col.update(columns)
    lp = np.full(len(df), intercept, dtype=float)
    for var, beta in coefs.items():
        x = pd.to_numeric(df[col[var]], errors="coerce").fillna(TRANSFER_MEDIANS[var]).to_numpy(dtype=float)
        lp += beta * x
    return 1.0 / (1.0 + np.exp(-lp))
