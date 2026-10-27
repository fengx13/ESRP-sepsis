"""BIDMC external-validation cohort, built from the raw MIMIC-IV files.

    python -m esrp.cohorts.bidmc --ed-v1 /path/mimic-iv-ed-1.0 --mimic-v1 /path/mimic-iv-1.0 \\
        --ed-v22 /path/mimic-iv-ed-2.2 --mimic-v31 /path/mimic-iv-3.1 --out results/bidmc_cohort.csv.gz

Which release each part comes from (this is what reproduces the published numbers):

* Cohort, triage and the ESRP predictors: MIMIC-IV-ED v1.0 with MIMIC-IV v1.0
  (``core/patients``, ``core/admissions``, ``hosp/diagnoses_icd``), processed as in
  the MIMIC-IV-ED benchmark (Xie et al., Sci Data 2022): adults only (448,804
  stays), temperature converted to Celsius, implausible vital signs removed or
  clipped, then missing values filled with the cohort median.
* Outcomes: ED diagnoses, ``pyxis`` and ``vitalsign`` from MIMIC-IV-ED v2.2;
  hospital diagnoses, ``labevents``, ``microbiologyevents``, ``emar``,
  ``prescriptions`` and ``icu/inputevents`` from MIMIC-IV v3.1, joined to the
  v1.0 stays by ``stay_id``, ``hadm_id`` and ``subject_id``.

Large tables are read in chunks. ``cache_dir`` keeps each filtered extract so an
interrupted run resumes where it stopped.
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

# MIMIC-IV-ED benchmark (from MIMIC-Extract): (outlier_low, valid_low, valid_high, outlier_high).
# Outside the outlier bounds -> missing; between a valid and an outlier bound -> clipped.
VITAL_RANGES = {"temperature": (14.2, 26, 45, 47), "heartrate": (0, 0, 350, 390),
                "resprate": (0, 0, 300, 330), "o2sat": (0, 0, 100, 150),
                "sbp": (0, 0, 375, 375), "dbp": (0, 0, 375, 375), "acuity": (1, 1, 5, 5)}
RENAME = {"heartrate": "heart_rate", "sbp": "systolic_bp", "dbp": "diastolic_bp",
          "o2sat": "oxygen_saturation", "resprate": "respiratory_rate",
          "temperature": "temperature", "acuity": "triage_acuity"}
# The benchmark's comorbidity window is 356 x 5 days (a typo for 365 x 5 in its code);
# kept because the published numbers were produced with it.
COMORBIDITY_LOOKBACK_DAYS = 356 * 5
CHUNK = 2_000_000


def _table(root: Path, module: str, name: str) -> Path:
    for m in ((module, "hosp") if module == "core" else (module,)):
        for ext in (".csv.gz", ".csv"):
            p = Path(root) / m / f"{name}{ext}"
            if p.exists():
                return p
    raise FileNotFoundError(f"{name} not found under {root}/{module}")


def _cached(cache_dir, name, build):
    if cache_dir is None:
        return build()
    p = Path(cache_dir) / f"{name}.pkl"
    if p.exists():
        return pd.read_pickle(p)
    d = build()
    p.parent.mkdir(parents=True, exist_ok=True)
    d.to_pickle(p)
    return d


def _chunks(path, usecols, keep, **kw):
    out = []
    for b in pd.read_csv(path, usecols=usecols, chunksize=CHUNK, low_memory=False, **kw):
        b = keep(b)
        if len(b):
            out.append(b)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=usecols)


# ---------------------------------------------------------------------------
# Cohort and predictors (v1.0)
# ---------------------------------------------------------------------------
def load_stays(ed_v1: Path, mimic_v1: Path) -> pd.DataFrame:
    ed = pd.read_csv(_table(ed_v1, "ed", "edstays"), parse_dates=["intime", "outtime"])
    pt = pd.read_csv(_table(mimic_v1, "core", "patients"),
                     usecols=["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"])
    d = ed.drop(columns=[c for c in ("gender",) if c in ed]).merge(pt, on="subject_id", how="left")
    d["age"] = d.intime.dt.year - d.anchor_year + d.anchor_age
    d = d[d.age >= 18].copy()
    tr = pd.read_csv(_table(ed_v1, "ed", "triage"))
    d = d.merge(tr, on=["subject_id", "stay_id"], how="left").rename(columns={"chiefcomplaint": "chief_complaint"})
    return d.sort_values(["subject_id", "intime"]).reset_index(drop=True)


def add_predictors(d: pd.DataFrame, mimic_v1: Path, fever_pattern: str = r"fever|chill",
                   comorbidity_lookback_days: float | None = COMORBIDITY_LOOKBACK_DAYS,
                   exclude_observation: bool = False) -> pd.DataFrame:
    d = d.copy()
    # Triage vital signs: Fahrenheit -> Celsius, benchmark outlier rules, cohort median.
    d["temperature"] = (d.temperature - 32) * (5 / 9)   # benchmark order of operations; matters at 38 °C
    for c, (ol, vl, vh, oh) in VITAL_RANGES.items():
        x = pd.to_numeric(d[c], errors="coerce")
        x = x.where((x >= ol) & (x <= oh))
        x = x.clip(lower=vl, upper=vh)
        d[f"{RENAME[c]}_imputed"] = x.isna().astype("int8")
        d[RENAME[c]] = x.fillna(x.median())
    d = d.drop(columns=[c for c in VITAL_RANGES if c not in RENAME.values()])

    d["fever_or_chills"] = d.chief_complaint.astype(str).str.contains(
        fever_pattern, case=False, regex=True).astype("int8")

    # Hospital admissions with 0 < arrival - admittime <= 365 days.
    adm = pd.read_csv(_table(mimic_v1, "core", "admissions"),
                      usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
                      parse_dates=["admittime", "dischtime"])
    # The published analysis counted every admission, observation types included;
    # exclude_observation=True applies the UMN definition.
    counted = adm[~adm.admission_type.str.contains("OBSERVATION")] if exclude_observation else adm
    j = d[["stay_id", "subject_id", "intime"]].merge(counted, on="subject_id")
    gap = j.intime - j.admittime
    n = j[(gap > pd.Timedelta(0)) & (gap <= pd.Timedelta(days=365))].groupby("stay_id").size()
    d["hospitalisations_365d"] = d.stay_id.map(n).fillna(0).astype("int32")
    d["hospitalisations_365d_ge2"] = (d.hospitalisations_365d >= 2).astype("int8")

    # Comorbidity: hospital diagnoses of admissions discharged before arrival.
    dx = pd.read_csv(_table(mimic_v1, "hosp", "diagnoses_icd"),
                     usecols=["subject_id", "hadm_id", "icd_code", "icd_version"])
    f = cm.flags(dx.icd_code, dx.icd_version)
    dx = pd.concat([dx[["hadm_id"]], f], axis=1)[f.any(axis=1).to_numpy()]
    dx = dx.merge(adm[["subject_id", "hadm_id", "dischtime"]], on="hadm_id")
    j = d[["stay_id", "subject_id", "intime"]].merge(dx, on="subject_id")
    gap = j.intime - j.dischtime
    keep = gap > pd.Timedelta(0)
    if comorbidity_lookback_days is not None:
        keep &= gap <= pd.Timedelta(days=comorbidity_lookback_days)
    got = j[keep].groupby("stay_id")[list(cm.CATEGORIES)].max()
    for c in cm.CATEGORIES:
        d[c] = d.stay_id.map(got[c]).fillna(0).astype("int8")
    d["fluid_electrolyte_disorder"] = d.eci_FluidsLytes
    return d


# ---------------------------------------------------------------------------
# Outcomes (ED v2.2 + MIMIC-IV v3.1)
# ---------------------------------------------------------------------------
def _abx_names(names, memo={}):
    out = set()
    skip = re.compile(T.ANTIBIOTIC_EXCLUDE_NAME, re.I)
    for x in pd.Series(names).dropna().unique():
        if x not in memo:
            memo[x] = T.name_matches(x) and not skip.search(x)
        if memo[x]:
            out.add(x)
    return out


def antibiotic_times(ed_v22, mimic_v31, subjects, cache_dir=None) -> pd.DataFrame:
    """Systemic antibacterials actually given: emar, plus prescriptions for patients absent
    from emar, plus ED pyxis dispenses."""
    def emar():
        def keep(b):
            b = b[b.subject_id.isin(subjects) & b.event_txt.isin(T.EMAR_GIVEN)]
            return b[b.medication.isin(_abx_names(b.medication))][["subject_id", "charttime"]]
        return _chunks(_table(mimic_v31, "hosp", "emar"),
                       ["subject_id", "charttime", "medication", "event_txt"], keep)

    def rx():
        def keep(b):
            b = b[b.subject_id.isin(subjects) & ~b.route.isin(T.NON_SYSTEMIC_ROUTES)]
            return b[b.drug.isin(_abx_names(b.drug))][["subject_id", "starttime"]]
        return _chunks(_table(mimic_v31, "hosp", "prescriptions"),
                       ["subject_id", "starttime", "drug", "route"], keep)

    e = _cached(cache_dir, "abx_emar", emar)
    p = _cached(cache_dir, "abx_prescriptions", rx).rename(columns={"starttime": "charttime"})
    p = p[~p.subject_id.isin(set(e.subject_id))]
    px = pd.read_csv(_table(ed_v22, "ed", "pyxis"), usecols=["subject_id", "charttime", "name"])
    px = px[px.name.isin(_abx_names(px.name))][["subject_id", "charttime"]]
    a = pd.concat([e, p, px], ignore_index=True)
    a["charttime"] = pd.to_datetime(a.charttime)
    return a.dropna(subset=["charttime"]).sort_values(["subject_id", "charttime"])


def culture_times(mimic_v31, subjects, cache_dir=None) -> pd.DataFrame:
    """Any microbiology specimen (charttime, else chartdate)."""
    m = _cached(cache_dir, "cultures", lambda: _chunks(
        _table(mimic_v31, "hosp", "microbiologyevents"), ["subject_id", "charttime", "chartdate"],
        lambda b: b[b.subject_id.isin(subjects)]))
    t = pd.to_datetime(m.charttime).fillna(pd.to_datetime(m.chartdate))
    return (pd.DataFrame({"subject_id": m.subject_id, "charttime": t})
            .dropna(subset=["charttime"]).sort_values(["subject_id", "charttime"]))


def suspicion_pairs(abx: pd.DataFrame, cx: pd.DataFrame) -> pd.DataFrame:
    """Seymour 2016: culture then antibiotic within 72 h, or antibiotic then culture within
    24 h; the time of suspicion is the earlier of the pair (nearest preceding partner)."""
    a = abx.rename(columns={"charttime": "abx_time"}).sort_values("abx_time")
    c = cx.rename(columns={"charttime": "cx_time"}).sort_values("cx_time")
    x = pd.merge_asof(a, c, left_on="abx_time", right_on="cx_time", by="subject_id",
                      direction="backward", tolerance=pd.Timedelta(hours=T.ABX_AFTER_CULTURE_H))
    y = pd.merge_asof(c, a, left_on="cx_time", right_on="abx_time", by="subject_id",
                      direction="backward", tolerance=pd.Timedelta(hours=T.CULTURE_AFTER_ABX_H))
    s = pd.concat([x.dropna(subset=["cx_time"]), y.dropna(subset=["abx_time"])], ignore_index=True)
    s["t"] = s[["cx_time", "abx_time"]].min(axis=1)
    return s[["subject_id", "t"]]


def lab_values(mimic_v31, subjects, cache_dir=None, path=None) -> pd.DataFrame:
    def build():
        src = path or _table(mimic_v31, "hosp", "labevents")
        d = _chunks(src, ["subject_id", "itemid", "charttime", "valuenum"],
                    lambda b: b[b.itemid.isin(T.MIMIC_LAB_ITEMS) & b.valuenum.notna()
                                & b.subject_id.isin(subjects)])
        d["label"] = d.itemid.map(T.MIMIC_LAB_ITEMS)
        d["charttime"] = pd.to_datetime(d.charttime)
        return d[["subject_id", "label", "charttime", "valuenum"]]
    return _cached(cache_dir, "labs", build)


def first_wbc(d, mimic_v31, cache_dir=None, path=None) -> pd.Series:
    """First white-cell count from 1 h before to 6 h after arrival, for SIRS."""
    def build():
        src = path or _table(mimic_v31, "hosp", "labevents")
        subj = set(d.subject_id)
        w = _chunks(src, ["subject_id", "itemid", "charttime", "valuenum"],
                    lambda b: b[b.itemid.isin(T.MIMIC_WBC_ITEMS) & b.valuenum.notna() & b.subject_id.isin(subj)])
        w["charttime"] = pd.to_datetime(w.charttime)
        return w[["subject_id", "charttime", "valuenum"]]
    w = _cached(cache_dir, "wbc", build)
    lo, hi = T.WBC_WINDOW_H["mimic"]
    j = d[["stay_id", "subject_id", "intime"]].merge(w, on="subject_id")
    h = (j.charttime - j.intime).dt.total_seconds() / 3600
    j = j[(h >= lo) & (h <= hi)].assign(h=h).sort_values("h").groupby("stay_id").first()
    return d.stay_id.map(j.valuenum)


def vasopressor_times(mimic_v31, subjects, cache_dir=None) -> pd.DataFrame:
    def build():
        d = _chunks(_table(mimic_v31, "icu", "inputevents"), ["subject_id", "itemid", "starttime"],
                    lambda b: b[b.itemid.isin(T.MIMIC_VASOPRESSORS) & b.subject_id.isin(subjects)])
        return pd.DataFrame({"subject_id": d.subject_id, "charttime": pd.to_datetime(d.starttime)})
    return _cached(cache_dir, "vasopressors", build)


def _in_window(events, cohort, value, how):
    j = cohort[["stay_id", "subject_id", "lo", "hi"]].merge(events, on="subject_id")
    j = j[(j.charttime >= j.lo) & (j.charttime <= j.hi)]
    g = j.groupby("stay_id")[value]
    return g.max() if how == "max" else g.min()


def sepsis_icd(d, ed_v22, mimic_v31) -> pd.Series:
    """ICD-coded sepsis in any position of the ED or hospital diagnoses."""
    ed = pd.read_csv(_table(ed_v22, "ed", "diagnosis"), usecols=["stay_id", "icd_code", "icd_version"])
    hd = pd.read_csv(_table(mimic_v31, "hosp", "diagnoses_icd"), usecols=["hadm_id", "icd_code", "icd_version"])
    e = set(ed.stay_id[is_sepsis_icd(ed.icd_code, ed.icd_version)])
    h = set(hd.hadm_id[is_sepsis_icd(hd.icd_code, hd.icd_version)])
    return (d.stay_id.isin(e) | d.hadm_id.isin(h)).astype("int8")


def add_outcomes(d, ed_v22, mimic_v31, cache_dir=None, labevents=None,
                 match_published: bool = True, wbc_labevents=None) -> pd.DataFrame:
    """``match_published``: the published run searched ``emar``, ``prescriptions`` and
    ``microbiologyevents`` only for patients who also appear in MIMIC-IV-ED v2.2 (its
    extracts were cached from the v2.2 cohort). Two v1.0 stays of patients absent from
    v2.2 therefore lost their suspected infection, and one of them its Sepsis-3 label.
    Set False to search every patient (Sepsis-3 17,306 and composite 23,650 instead of
    17,305 and 23,649)."""
    d = d.copy()
    d["sepsis_icd"] = sepsis_icd(d, ed_v22, mimic_v31)
    subjects = set(d.subject_id)
    abx = antibiotic_times(ed_v22, mimic_v31, subjects, cache_dir)
    cx = culture_times(mimic_v31, subjects, cache_dir)
    if match_published:
        v22 = set(pd.read_csv(_table(ed_v22, "ed", "edstays"), usecols=["subject_id"]).subject_id)
        abx, cx = abx[abx.subject_id.isin(v22)], cx[cx.subject_id.isin(v22)]
    s = suspicion_pairs(abx, cx)
    # Earliest suspicion between 24 h before arrival and ED departure.
    j = d[["stay_id", "subject_id", "intime", "outtime"]].merge(s, on="subject_id")
    j = j[(j.t >= j.intime - pd.Timedelta(hours=T.SUSPICION_BEFORE_ARRIVAL_H)) & (j.t <= j.outtime)]
    d["time_suspicion"] = d.stay_id.map(j.groupby("stay_id").t.min())
    d["suspected_infection"] = d.time_suspicion.notna().astype("int8")
    c = d.loc[d.suspected_infection == 1, ["stay_id", "subject_id", "time_suspicion"]].copy()
    c["lo"] = c.time_suspicion - pd.Timedelta(hours=T.SOFA_WINDOW_BEFORE_H)
    c["hi"] = c.time_suspicion + pd.Timedelta(hours=T.SOFA_WINDOW_AFTER_H)

    labs = lab_values(mimic_v31, set(c.subject_id), cache_dir, labevents)
    for label, how, col in [("platelets", "min", "platelets_min"), ("bilirubin", "max", "bilirubin_max"),
                            ("creatinine", "max", "creatinine_max"), ("lactate", "max", "lactate_max")]:
        e = labs.loc[labs.label == label, ["subject_id", "charttime", "valuenum"]]
        d[col] = d.stay_id.map(_in_window(e, c, "valuenum", how))

    # Lowest MAP from ED vital signs ((SBP + 2 DBP)/3) inside the window, artefacts dropped.
    vs = pd.read_csv(_table(ed_v22, "ed", "vitalsign"), usecols=["stay_id", "charttime", "sbp", "dbp"],
                     parse_dates=["charttime"])
    vs = vs[vs.stay_id.isin(set(c.stay_id))]
    vs["map"] = pd.to_numeric(vs.sbp, errors="coerce") / 3 + pd.to_numeric(vs.dbp, errors="coerce") * 2 / 3
    v = vs.merge(c[["stay_id", "lo", "hi"]], on="stay_id")
    v = v[(v.charttime >= v.lo) & (v.charttime <= v.hi)]
    v = v[v["map"].between(*T.MAP_PLAUSIBLE) | v["map"].isna()]
    d["map_min"] = d.stay_id.map(v.groupby("stay_id")["map"].min())

    vp = vasopressor_times(mimic_v31, set(c.subject_id), cache_dir).assign(flag=1)
    d["on_vasopressor"] = d.stay_id.map(_in_window(vp, c, "flag", "max")).fillna(0).astype("int8")

    sofa = pd.DataFrame({"liver": sofa_liver(d.bilirubin_max), "renal": sofa_renal(d.creatinine_max),
                         "coagulation": sofa_coagulation(d.platelets_min),
                         "cardiovascular": sofa_cardiovascular(d.map_min.fillna(999), d.on_vasopressor)},
                        index=d.index)
    for k in sofa:
        d[f"sofa_{k}"] = sofa[k]
    d["sepsis3"] = sepsis3(d.suspected_infection, sofa, baseline_sofa(d)).astype("int8")
    d["septic_shock"] = septic_shock(d.sepsis3, d.on_vasopressor, d.lactate_max).astype("int8")
    d["composite"] = composite_outcome(d.sepsis_icd, d.sepsis3).astype("int8")
    d["wbc_6h"] = first_wbc(d, mimic_v31, cache_dir, wbc_labevents)
    # Arrival by ambulance or helicopter (qSepsis variable set; recorded in ED v2.2, not v1.0).
    ar = pd.read_csv(_table(ed_v22, "ed", "edstays"), usecols=["stay_id", "arrival_transport"])
    ems = set(ar.stay_id[ar.arrival_transport.isin(["AMBULANCE", "HELICOPTER"])])
    d["arrival_ems"] = d.stay_id.isin(ems).astype("int8")
    return d


def build(ed_v1, mimic_v1, ed_v22=None, mimic_v31=None, outcomes=True, cache_dir=None,
          labevents=None, match_published=True, wbc_labevents=None, fever_pattern=r"fever|chill",
          comorbidity_lookback_days=COMORBIDITY_LOOKBACK_DAYS, exclude_observation=False) -> pd.DataFrame:
    d = load_stays(Path(ed_v1), Path(mimic_v1))
    d = add_predictors(d, Path(mimic_v1), fever_pattern, comorbidity_lookback_days, exclude_observation)
    d["esrp"] = score_frame(d)
    if outcomes:
        d = add_outcomes(d, Path(ed_v22), Path(mimic_v31), cache_dir, labevents, match_published, wbc_labevents)
    return d


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ed-v1", required=True)
    p.add_argument("--mimic-v1", required=True)
    p.add_argument("--ed-v22", required=True)
    p.add_argument("--mimic-v31", required=True)
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--labevents", default=None,
                   help="optional pre-filtered copy of hosp/labevents (same columns) to save time")
    p.add_argument("--exclude-observation", action="store_true",
                   help="count only non-observation admissions (UMN definition)")
    p.add_argument("--all-patients", action="store_true",
                   help="search antibiotics and cultures for every patient (see add_outcomes)")
    p.add_argument("--out", default="results/bidmc_cohort.csv.gz")
    a = p.parse_args(argv)
    d = build(a.ed_v1, a.mimic_v1, a.ed_v22, a.mimic_v31, cache_dir=a.cache_dir,
              labevents=a.labevents, wbc_labevents=a.labevents,
              match_published=not a.all_patients,
              exclude_observation=a.exclude_observation)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.drop(columns=["time_suspicion", "intime", "outtime"]).to_csv(a.out, index=False)
    print(f"{len(d):,} stays; ICD {int(d.sepsis_icd.sum()):,}; Sepsis-3 {int(d.sepsis3.sum()):,}; "
          f"composite {int(d.composite.sum()):,}; written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
