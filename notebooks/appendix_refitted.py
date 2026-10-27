"""eTable 2, Panel A, lower block: models refitted within each external cohort by
five-fold cross-validation (an upper bound for a model tuned to the site).

    python notebooks/appendix_refitted.py results/bidmc_cohort.csv.gz BIDMC [--cache DIR]
    python notebooks/appendix_refitted.py results/stanford_cohort.csv.gz Stanford [--cache DIR]

The random forest saves each fold to --cache, so an interrupted run resumes.
Needs scikit-learn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esrp import evaluation as E                              # noqa: E402

ALTERED_MENTATION = r"alter|confus|unrespons|letharg|mental status|unconscious|obtund"
RF9 = ["age", "systolic_bp", "diastolic_bp", "heart_rate", "oxygen_saturation", "respiratory_rate",
       "fever_or_chills", "hospitalisations_365d", "fluid_electrolyte_disorder"]


def qsepsis_features(d: pd.DataFrame) -> pd.DataFrame:
    """The twelve Jiang variables available in the cohort, chief complaints split as in qSepsis."""
    cc = d.chief_complaint.astype(str)
    return pd.DataFrame({
        "age": d.age, "male": d.gender.astype(str).str.upper().str[0].eq("M").astype(int),
        "arrival_ems": d.arrival_ems, "temperature": d.temperature, "systolic_bp": d.systolic_bp,
        "diastolic_bp": d.diastolic_bp, "heart_rate": d.heart_rate, "spo2": d.oxygen_saturation,
        "resp_rate": d.respiratory_rate, "triage_acuity": d.triage_acuity, "fever": d.fever_or_chills,
        "altered_mentation": cc.str.contains(ALTERED_MENTATION, case=False).astype(int),
        "abdominal_pain": cc.str.contains("abdominal pain|abd pain", case=False).astype(int),
        "back_pain": cc.str.contains("back pain", case=False).astype(int)})


def cv_logistic(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("lr", LogisticRegression(max_iter=2000))])
    oof = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        pipe.fit(X.iloc[tr], y[tr]); oof[te] = pipe.predict_proba(X.iloc[te])[:, 1]
    return oof


def cv_forest(X: np.ndarray, y: np.ndarray, cache: Path | None, tag: str) -> np.ndarray:
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(X, y), 1):
        f = cache / f"rf9_{tag}_fold{k}.npy" if cache else None
        if f is not None and f.exists():
            oof[te] = np.load(f); continue
        rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=50, max_features="sqrt",
                                    n_jobs=-1, random_state=7).fit(X[tr], y[tr])
        oof[te] = rf.predict_proba(X[te])[:, 1]
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True); np.save(f, oof[te])
        print(f"  random forest fold {k}/5 done", flush=True)
    return oof


def main(argv=None):
    p = argparse.ArgumentParser(); p.add_argument("cohort"); p.add_argument("site")
    p.add_argument("--cache", default=None); p.add_argument("--only", choices=["logistic", "forest"])
    a = p.parse_args(argv)
    d = pd.read_csv(a.cohort, low_memory=False)
    y = d.composite.to_numpy(int); s = d.esrp.to_numpy(float)
    out = {}
    if a.only in (None, "logistic"):
        out["qSepsis variable set, five-fold cross-validation"] = cv_logistic(qsepsis_features(d), y)
    if a.only in (None, "forest"):
        X = d[RF9].astype(float); X = X.fillna(X.median()).to_numpy()
        out["Random forest on the nine ESRP variables"] = cv_forest(X, y, Path(a.cache) if a.cache else None, a.site)
    for lab, v in out.items():
        auc, lo, hi = E.auroc(y, v); pv = E.delong_paired(y, s, v)[5]
        print(f"{a.site}  {lab}: {auc:.4f} ({lo:.4f}-{hi:.4f})  p vs ESRP {pv:.2g}")


if __name__ == "__main__":
    main()
