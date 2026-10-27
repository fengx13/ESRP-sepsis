"""Stanford external-validation cohort, built from the raw MC-MED v1.0.0 files.

    python -m esrp.cohorts.stanford --mcmed /path/to/mc-med-1.0.0 --out results/stanford_cohort.csv.gz

One row per ED visit (118,385 visits, all aged 18 or over). Reads ``visits.csv``,
``pmh.csv``, ``orders.csv``, ``labs.csv`` and ``numerics.csv``; the last is 2.7 GB
and is streamed, keeping only the visits with suspected infection.

Operational choices that reproduce the published Stanford numbers, and where
they depart from the wording of the manuscript, are documented in
``docs/provenance.md``. Three are switchable here:

* ``fever_pattern``: the published Stanford analysis flagged a chief complaint
  containing "fever"; chills alone (485 visits) were not flagged.
* ``comorbidity_lookback_days``: the published Stanford analysis used every
  problem-list entry noted before the arrival (``None``), not a five-year window.
* ``exclude_observation``: the published analysis counted earlier visits ending in
  observation as hospitalisations; True applies the UMN definition.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .. import comorbidity as cm
from ..definitions import (is_sepsis_icd, sofa_coagulation, sofa_liver, sofa_renal,
                           sofa_cardiovascular, baseline_sofa, sepsis3, septic_shock,
                           composite_outcome)
from ..score import score_frame
from . import _sepsis3_terms as T

ADMITTED = ("Inpatient", "ICU", "Observation")
VITALS = {"Triage_HR": "heart_rate", "Triage_SBP": "systolic_bp", "Triage_DBP": "diastolic_bp",
          "Triage_SpO2": "oxygen_saturation", "Triage_RR": "respiratory_rate",
          "Triage_Temp": "temperature"}


def read_times(s: pd.Series) -> pd.Series:
    """MC-MED timestamps are ISO-8601 with a Z, shifted into the 2200s (past pandas'
    nanosecond limit), so they are parsed at second resolution."""
    txt = s.astype("string").str.rstrip("Z")
    out = np.full(len(txt), np.datetime64("NaT"), dtype="datetime64[s]")
    ok = txt.notna().to_numpy()
    out[ok] = txt[ok].to_numpy(dtype="U19").astype("datetime64[s]")
    return pd.Series(out, index=s.index)


def _prior_count(pid, t, ev_pid, ev_t, days):
    """For each query (pid, t): number of events of the same patient in [t - days, t)."""
    # Encode (patient, time) as one sortable float: rank * span + seconds.
    ranks = {p: i for i, p in enumerate(np.unique(np.r_[np.asarray(pid), np.asarray(ev_pid)]))}
    base = np.datetime64("2100-01-01T00:00:00", "s")
    span = 400 * 366 * 86400.0
    key = lambda p, tt: (pd.Series(p).map(ranks).to_numpy("float64") * span
                         + (np.asarray(tt, "datetime64[s]") - base).astype("float64"))
    ek = np.sort(key(ev_pid, ev_t))
    qk = key(pid, t)
    hi = np.searchsorted(ek, qk, side="left")
    lo = np.searchsorted(ek, qk - days * 86400.0, side="left")
    return (hi - lo).astype("int32")


# ---------------------------------------------------------------------------
def load_visits(mcmed: Path) -> pd.DataFrame:
    v = pd.read_csv(mcmed / "visits.csv", low_memory=False)
    d = pd.DataFrame({"subject_id": v.MRN, "stay_id": v.CSN, "age": v.Age, "gender": v.Gender})
    d["intime"] = read_times(v.Arrival_time)
    for src, dst in VITALS.items():
        d[dst] = pd.to_numeric(v[src], errors="coerce")
    d["triage_acuity"] = pd.to_numeric(v.Triage_acuity.astype("string").str.split("-").str[0],
                                       errors="coerce")
    d["chief_complaint"] = v.CC
    d["arrival_ems"] = v.Means_of_arrival.eq("EMS").astype("int8")
    d["ed_disposition"] = v.ED_dispo
    d["dx_icd9"], d["dx_icd10"], d["dx_name"] = v.Dx_ICD9, v.Dx_ICD10, v.Dx_name
    return d


def add_predictors(d: pd.DataFrame, mcmed: Path, fever_pattern: str = r"fever",
                   comorbidity_lookback_days: float | None = None,
                   exclude_observation: bool = False) -> pd.DataFrame:
    d = d.copy()
    # Chief complaint of fever (or chills; see module docstring).
    d["fever_or_chills"] = d.chief_complaint.fillna("").str.contains(
        fever_pattern, case=False, regex=True).astype("int8")

    # Hospitalisations in the previous 365 days: earlier ED visits of the same
    # patient, arriving in [t - 365 d, t), whose ED disposition was an admission.
    # The published analysis counted observation stays; exclude_observation=True applies
    # the UMN definition (inpatient or ICU only).
    adm = d[d.ed_disposition.isin([a for a in ADMITTED if not (exclude_observation and a == "Observation")])]
    d["hospitalisations_365d"] = _prior_count(d.subject_id, d.intime, adm.subject_id, adm.intime, 365)
    d["hospitalisations_365d_ge2"] = (d.hospitalisations_365d >= 2).astype("int8")

    # Comorbidity from the problem list (pmh.csv), noted before arrival.
    p = pd.read_csv(mcmed / "pmh.csv", usecols=["MRN", "Noted_date", "CodeType", "Code"])
    f = cm.flags(p.Code, np.where(p.CodeType.eq("Dx9"), 9, 10))
    p = pd.concat([p[["MRN"]], read_times(p.Noted_date).rename("noted"), f], axis=1)
    p = p[f.any(axis=1).to_numpy() & p.noted.notna().to_numpy()]
    j = d[["subject_id", "stay_id", "intime"]].merge(p, left_on="subject_id", right_on="MRN")
    gap = (j.intime - j.noted).dt.total_seconds()
    keep = gap > 0
    if comorbidity_lookback_days is not None:
        keep &= gap <= comorbidity_lookback_days * 86400
    got = j[keep].groupby("stay_id")[list(cm.CATEGORIES)].max()
    for c in cm.CATEGORIES:
        d[c] = d.stay_id.map(got[c]).fillna(0).astype("int8")
    d["fluid_electrolyte_disorder"] = d.eci_FluidsLytes

    # Missing triage vital signs and acuity: median of this cohort (acuity: 527 visits, median 3).
    d["triage_acuity_imputed"] = d.triage_acuity.isna().astype("int8")
    d["triage_acuity"] = d.triage_acuity.fillna(d.triage_acuity.median())
    for c in VITALS.values():
        d[f"{c}_imputed"] = d[c].isna().astype("int8")
        d[c] = d[c].fillna(d[c].median())
    return d


# ---------------------------------------------------------------------------
def _orders(mcmed: Path) -> pd.DataFrame:
    o = pd.read_csv(mcmed / "orders.csv",
                    usecols=["CSN", "Order_type", "Procedure_name", "Order_time", "First_admin_time"],
                    dtype={"Order_time": "string", "First_admin_time": "string"})
    o["Procedure_name"] = o.Procedure_name.fillna("")
    return o


def suspected_infection(o: pd.DataFrame) -> pd.Series:
    """Time of suspicion per visit: culture then antibiotic within 72 h, or antibiotic then
    culture within 24 h; the earlier of the pair, earliest pair per visit."""
    cx = o[o.Order_type.isin(T.MCMED_CULTURE_ORDER_TYPES)].copy()
    cx["cx_time"] = read_times(cx.Order_time)
    cx = cx.dropna(subset=["cx_time"])[["CSN", "cx_time"]]
    med = o[o.Order_type.eq("Medications")]
    names = pd.Series(med.Procedure_name.unique())
    names = names[[T.name_matches(x) for x in names]]
    names = set(names[~names.str.contains(T.ANTIBIOTIC_EXCLUDE_NAME, flags=re.I)])
    ab = med[med.Procedure_name.isin(names)].copy()
    ab["abx_time"] = read_times(ab.First_admin_time).fillna(read_times(ab.Order_time))
    ab = ab.dropna(subset=["abx_time"])[["CSN", "abx_time"]]
    j = cx.merge(ab, on="CSN")
    gap = (j.abx_time - j.cx_time).dt.total_seconds() / 3600
    j = j[gap.between(0, T.ABX_AFTER_CULTURE_H) | gap.between(-T.CULTURE_AFTER_ABX_H, 0)]
    j = j.assign(t=j[["cx_time", "abx_time"]].min(axis=1))
    return j.groupby("CSN")["t"].min()


def vasopressor_visits(o: pd.DataFrame) -> set:
    med = o[o.Order_type.eq("Medications")]
    u = pd.Series(med.Procedure_name.unique())
    u = u[u.str.contains(T.MCMED_VASOPRESSOR_NAME, case=False)
          & ~u.str.contains(T.MCMED_VASOPRESSOR_EXCLUDE, case=False)
          & u.str.contains(T.MCMED_VASOPRESSOR_REQUIRE, case=False)]
    return set(med.CSN[med.Procedure_name.isin(set(u))])


def worst_labs(mcmed: Path, win: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for b in pd.read_csv(mcmed / "labs.csv", chunksize=2_000_000,
                         usecols=["CSN", "Component_name", "Component_value", "Result_time"],
                         dtype={"Component_value": "string", "Result_time": "string"}):
        parts.append(b[b.Component_name.isin(T.MCMED_LAB_ITEMS) & b.CSN.isin(win.index)])
    d = pd.concat(parts, ignore_index=True)
    d["label"] = d.Component_name.map(T.MCMED_LAB_ITEMS)
    d["value"] = pd.to_numeric(d.Component_value.str.replace(r"[<>=]", "", regex=True), errors="coerce")
    d["t"] = read_times(d.Result_time)
    d = d.dropna(subset=["value", "t"]).merge(win, left_on="CSN", right_index=True)
    d = d[(d.t >= d.lo) & (d.t <= d.hi)]
    g = d.groupby(["CSN", "label"])["value"]
    out = pd.DataFrame({"platelets_min": g.min().xs("platelets", level="label"),
                        "bilirubin_max": g.max().xs("bilirubin", level="label"),
                        "creatinine_max": g.max().xs("creatinine", level="label"),
                        "lactate_max": g.max().xs("lactate", level="label")})
    return out


def worst_map(mcmed: Path, win: pd.DataFrame, numerics: Path | None = None) -> pd.Series:
    """Lowest MAP in the SOFA window: charted MAP, else (SBP + 2 DBP)/3 from paired readings.

    ``numerics.csv`` (2.7 GB) is read in chunks; ``numerics`` may point to a pre-filtered copy."""
    keep, csns = [], set(win.index)
    for b in pd.read_csv(numerics or (mcmed / "numerics.csv"), chunksize=4_000_000,
                         usecols=["CSN", "Measure", "Value", "Time"], dtype={"Time": "string"}):
        b = b[b.Measure.isin(T.PLAUSIBLE) & b.CSN.isin(csns)]
        if len(b):
            keep.append(b)
    d = pd.concat(keep, ignore_index=True)
    d["t"] = read_times(d.Time)
    lo = d.Measure.map({k: v[0] for k, v in T.PLAUSIBLE.items()})
    hi = d.Measure.map({k: v[1] for k, v in T.PLAUSIBLE.items()})
    d = d[d.Value.between(lo, hi)].merge(win, left_on="CSN", right_index=True)
    d = d[(d.t >= d.lo) & (d.t <= d.hi)]
    mp = d[d.Measure.eq("MAP")].groupby("CSN")["Value"].min()
    sd = d[d.Measure.isin(["SBP", "DBP"])].pivot_table(index=["CSN", "t"], columns="Measure", values="Value")
    if not sd.empty:
        derived = (sd["SBP"] / 3 + sd["DBP"] * 2 / 3).groupby("CSN").min()
        mp = mp.reindex(mp.index.union(derived.index)).fillna(derived)
    return mp


def first_wbc(d: pd.DataFrame, mcmed: Path) -> pd.Series:
    """First white-cell count ordered from arrival to 6 h after, for SIRS."""
    parts = []
    for b in pd.read_csv(mcmed / "labs.csv", chunksize=2_000_000,
                         usecols=["CSN", "Component_name", "Component_value", "Order_time"],
                         dtype={"Component_value": "string", "Order_time": "string"}):
        parts.append(b[b.Component_name.eq(T.MCMED_WBC_COMPONENT)])
    w = pd.concat(parts, ignore_index=True)
    w["value"] = pd.to_numeric(w.Component_value.str.replace(r"[<>=]", "", regex=True), errors="coerce")
    w["t"] = read_times(w.Order_time)
    j = w.dropna(subset=["value", "t"]).merge(d[["stay_id", "intime"]], left_on="CSN", right_on="stay_id")
    h = (j.t - j.intime).dt.total_seconds() / 3600
    lo, hi = T.WBC_WINDOW_H["mcmed"]
    j = j[(h >= lo) & (h <= hi)].assign(h=h).sort_values("h").groupby("stay_id").first()
    return d.stay_id.map(j.value)


def add_outcomes(d: pd.DataFrame, mcmed: Path) -> pd.DataFrame:
    d = d.copy()
    # ICD-coded sepsis: the visit's ED diagnosis codes (MC-MED has no hospital discharge codes).
    hit = pd.Series(False, index=d.index)
    for col in ("dx_icd9", "dx_icd10"):
        e = d[col].dropna().astype(str).str.split(",").explode().str.strip()
        e = e[e.ne("")]
        hit |= is_sepsis_icd(e).groupby(level=0).any().reindex(d.index, fill_value=False).astype(bool)
    d["sepsis_icd"] = hit.astype("int8")

    o = _orders(mcmed)
    t = suspected_infection(o)
    d["time_suspicion"] = d.stay_id.map(t)
    d["suspected_infection"] = d.time_suspicion.notna().astype("int8")
    win = pd.DataFrame({"lo": t - np.timedelta64(T.SOFA_WINDOW_BEFORE_H, "h"),
                        "hi": t + np.timedelta64(T.SOFA_WINDOW_AFTER_H, "h")})
    labs = worst_labs(mcmed, win)
    for c in labs.columns:
        d[c] = d.stay_id.map(labs[c])
    d["map_min"] = d.stay_id.map(worst_map(mcmed, win))
    d["on_vasopressor"] = (d.stay_id.isin(vasopressor_visits(o)) & d.suspected_infection.astype(bool)).astype("int8")

    sofa = pd.DataFrame({"liver": sofa_liver(d.bilirubin_max), "renal": sofa_renal(d.creatinine_max),
                         "coagulation": sofa_coagulation(d.platelets_min),
                         "cardiovascular": sofa_cardiovascular(d.map_min.fillna(999), d.on_vasopressor)},
                        index=d.index)
    for c in sofa:
        d[f"sofa_{c}"] = sofa[c]
    base = baseline_sofa(d)
    d["sepsis3"] = sepsis3(d.suspected_infection, sofa, base).astype("int8")
    d["septic_shock"] = septic_shock(d.sepsis3, d.on_vasopressor, d.lactate_max).astype("int8")
    d["composite"] = composite_outcome(d.sepsis_icd, d.sepsis3).astype("int8")
    d["wbc_6h"] = first_wbc(d, mcmed)
    return d


def build(mcmed, fever_pattern: str = r"fever", comorbidity_lookback_days: float | None = None,
          outcomes: bool = True, exclude_observation: bool = False) -> pd.DataFrame:
    mcmed = Path(mcmed)
    d = load_visits(mcmed)
    d = d[d.age >= 18].reset_index(drop=True)
    d = add_predictors(d, mcmed, fever_pattern, comorbidity_lookback_days, exclude_observation)
    d["esrp"] = score_frame(d)
    if outcomes:
        d = add_outcomes(d, mcmed)
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcmed", required=True, help="folder holding the MC-MED v1.0.0 CSV files")
    p.add_argument("--out", default="results/stanford_cohort.csv.gz")
    p.add_argument("--fever-pattern", default=r"fever")
    p.add_argument("--lookback-days", type=float, default=None)
    p.add_argument("--exclude-observation", action="store_true")
    a = p.parse_args(argv)
    d = build(a.mcmed, a.fever_pattern, a.lookback_days, exclude_observation=a.exclude_observation)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["time_suspicion", "intime"]).to_csv(a.out, index=False)
    print(f"{len(d):,} visits; ICD {int(d.sepsis_icd.sum()):,}; Sepsis-3 {int(d.sepsis3.sum()):,}; "
          f"composite {int(d.composite.sum()):,}; written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
