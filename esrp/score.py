"""The ESRP score, exactly as published (Table 2 of the manuscript).

Nine variables recorded at ED triage, each scored from a fixed table and summed.
No laboratory test is used. The total ranges from 0 to 100 and places the
encounter in one of four ordered risk tiers.

Intervals follow Table 2: a half-open interval [a, b) includes a and excludes b.
Missing values are not imputed here; pass complete rows, or impute first with
``impute()`` using the derivation-cohort medians.
"""
from __future__ import annotations
import math
from typing import Mapping

# ---- Table 2: variable, intervals and points ------------------------------
# Each entry: (lower bound inclusive or None, upper bound exclusive or None, points)
POINTS = {
    "heart_rate":        [(None, 90, 0), (90, 110, 3), (110, None, 6)],       # beats/min
    "systolic_bp":       [(None, 90, 14), (90, 180, 3), (180, None, 0)],      # mm Hg
    "age":               [(18, 30, 0), (30, 70, 9), (70, None, 14)],          # years
    "diastolic_bp":      [(None, 60, 11), (60, 90, 6), (90, None, 0)],        # mm Hg
    "oxygen_saturation": [(None, 94, 11), (94, 99, 3), (99, None, 0)],        # %
    "respiratory_rate":  [(None, 18, 0), (18, 22, 3), (22, None, 9)],         # breaths/min
}
BINARY_POINTS = {
    "fever_or_chills":           20,  # chief complaint of fever or chills
    "hospitalisations_365d_ge2":  6,  # two or more hospitalisations in the previous 365 days
    "fluid_electrolyte_disorder": 9,  # Elixhauser fluid and electrolyte disorder, coded in the previous five years
}
VARIABLES = tuple(POINTS) + tuple(BINARY_POINTS)
MAX_SCORE = 100

# ---- Risk tiers (fixed in the UMN internal validation cohort) --------------
THRESHOLDS = (21, 31, 45)                      # lowest score of each tier above "low"
TIERS = (("low", 0, 20), ("moderate", 21, 30), ("high", 31, 44), ("very high", 45, MAX_SCORE))

# ---- Probability mapping fitted in the derivation cohort (Table 2 footnote) -
INTERCEPT, SLOPE = -7.2541, 0.111705


def _points_continuous(name: str, value: float) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError(f"{name} is missing; impute before scoring")
    for lo, hi, pts in POINTS[name]:
        if (lo is None or value >= lo) and (hi is None or value < hi):
            return pts
    raise ValueError(f"{name}={value!r} outside the scoring table")


def esrp_points(row: Mapping[str, object]) -> dict[str, int]:
    """Points for each of the nine variables, keyed by variable name."""
    out = {name: _points_continuous(name, float(row[name])) for name in POINTS}
    for name, pts in BINARY_POINTS.items():
        v = row[name]
        if v is None or (isinstance(v, float) and math.isnan(v)):
            raise ValueError(f"{name} is missing")
        out[name] = pts if bool(v) else 0
    return out


def esrp_score(row: Mapping[str, object]) -> int:
    """Total ESRP score (0 to 100) for one encounter."""
    return sum(esrp_points(row).values())


def risk_tier(score: int) -> str:
    """Risk tier for a total score: low (<21), moderate (21-30), high (31-44), very high (>=45)."""
    if not 0 <= score <= MAX_SCORE:
        raise ValueError(f"score {score} outside 0..{MAX_SCORE}")
    for name, lo, hi in TIERS:
        if lo <= score <= hi:
            return name
    raise AssertionError  # unreachable


def predicted_probability(score: int, intercept: float = INTERCEPT, slope: float = SLOPE) -> float:
    """Predicted probability of sepsis from the logistic mapping fitted in the derivation cohort.

    The mapping under-predicts absolute risk outside the developing institution;
    an adopting site should re-estimate ``intercept`` locally (eTable 8).
    """
    return 1.0 / (1.0 + math.exp(-(intercept + slope * score)))


def score_frame(df, columns: Mapping[str, str] | None = None):
    """Vectorised scoring of a pandas DataFrame.

    ``columns`` maps the nine variable names used here to the column names in ``df``
    (default: identical names). Returns an integer Series aligned with ``df``.
    """
    import numpy as np
    import pandas as pd
    col = {v: v for v in VARIABLES}
    if columns:
        col.update(columns)
    total = pd.Series(0, index=df.index, dtype="int64")
    for name, table in POINTS.items():
        x = pd.to_numeric(df[col[name]], errors="coerce")
        if x.isna().any():
            raise ValueError(f"{col[name]} has {int(x.isna().sum())} missing values; impute before scoring")
        pts = pd.Series(np.nan, index=df.index)
        for lo, hi, p in table:
            m = pd.Series(True, index=df.index)
            if lo is not None:
                m &= x >= lo
            if hi is not None:
                m &= x < hi
            pts[m] = p
        if pts.isna().any():
            raise ValueError(f"{col[name]} has values outside the scoring table")
        total += pts.astype("int64")
    for name, p in BINARY_POINTS.items():
        v = pd.to_numeric(df[col[name]], errors="coerce")
        if v.isna().any():
            raise ValueError(f"{col[name]} has missing values")
        total += (v != 0).astype("int64") * p
    return total
