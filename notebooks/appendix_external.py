"""Appendix tables, external validation (BIDMC and Stanford columns).

    python notebooks/appendix_external.py results/bidmc_cohort.csv.gz results/stanford_cohort.csv.gz

eTable 2, Panels A (comparators fixed or frozen) and B (published thresholds);
eTable 5, Panel A (outcome definitions); eTable 7, Panel A (thresholds);
eTable 8 (recalibration). Writes results/appendix_external.csv.
The two cohort-refitted models of eTable 2, Panel A are in appendix_refitted.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esrp import comparators as C, evaluation as E          # noqa: E402
from esrp.score import INTERCEPT, SLOPE                       # noqa: E402

D, EN = "·", "–"
VISITS_PER_ED_PER_DAY = {"BIDMC": 448_804 / 3_287,   # stays over the 3,287 days of 2011-2019
                         "Stanford": 162}      # eTable 7, Panel A note


def f(x, k=3):
    return f"{x:.{k}f}".replace(".", D).replace("-", "−")


def ci(est, lo, hi, k=3):
    return f"{f(est, k)} ({f(lo, k)}{EN}{f(hi, k)})"


def pval(p):
    return "p<0" + D + "0001" if p < 1e-4 else "p = " + f(p, 2 if p >= 0.01 else 3)


def scores(d: pd.DataFrame) -> dict:
    male = d.gender.astype(str).str.upper().str[0].eq("M").astype(int)
    abdominal = d.chief_complaint.astype(str).str.contains("abdominal pain|abd pain", case=False).astype(int)
    q = pd.DataFrame({"age": d.age, "resp_rate": d.respiratory_rate, "heart_rate": d.heart_rate,
                      "temp_c": d.temperature, "abdominal_pain": abdominal, "fever_or_chills": d.fever_or_chills,
                      "male": male, "triage_acuity": d.triage_acuity, "spo2": d.oxygen_saturation,
                      "systolic_bp": d.systolic_bp, "diastolic_bp": d.diastolic_bp,
                      "hospitalisations_365d_ge2": d.hospitalisations_365d_ge2,
                      "fluid_electrolyte_disorder": d.fluid_electrolyte_disorder})
    return {
        "Triage acuity (ESI)": C.esi_reversed(d.triage_acuity),
        "qSOFA": C.qsofa(d.respiratory_rate, d.systolic_bp),
        "NEWS2": C.news2(d.respiratory_rate, d.temperature, d.systolic_bp, d.heart_rate,
                         components=C.NEWS2_COMPONENTS_BENCHMARK),
        "MEWS": C.mews(d.systolic_bp, d.heart_rate, d.respiratory_rate, d.temperature),
        "REMS": C.rems(d.age, C.mean_arterial_pressure(d.systolic_bp, d.diastolic_bp), d.heart_rate,
                       d.respiratory_rate, d.oxygen_saturation),
        "SIRS": C.sirs(d.temperature, d.heart_rate, d.respiratory_rate, d.wbc_6h),
        "CART": C.cart(d.age, d.respiratory_rate, d.heart_rate, d.diastolic_bp),
        "qSepsis variable set, ten variables": C.transfer_model(q, "qsepsis_10"),
        "qSepsis variable set without triage acuity": C.transfer_model(q, "qsepsis_without_acuity"),
        "ESRP variables entered as continuous terms": C.transfer_model(q, "esrp_variables_continuous"),
    }


THRESHOLDS = [("Triage acuity (ESI) 1 or 2", "Triage acuity (ESI)", -2), ("qSOFA ≥2", "qSOFA", 2),
              ("NEWS2 ≥5", "NEWS2", 5), ("MEWS ≥5", "MEWS", 5), ("REMS ≥9", "REMS", 9),
              ("CART >20", "CART", 21), ("SIRS ≥2 criteria", "SIRS", 2)]


def oe_poisson(o, e):
    lo = chi2.ppf(0.025, 2 * o) / 2 if o > 0 else 0.0
    return o / e, lo / e, chi2.ppf(0.975, 2 * o + 2) / 2 / e


def column(name: str, d: pd.DataFrame) -> dict:
    y = d.composite.to_numpy(int); s = d.esrp.to_numpy(int); n = len(y); r = {}
    # eTable 2, Panel A
    r["2A ESRP integer score"] = ci(*E.auroc(y, s))
    sc = scores(d)
    for lab, v in sc.items():
        a = E.delong_paired(y, s, v)
        r[f"2A {lab}"] = f"{ci(*E.auroc(y, v))} {pval(a[5])}"
    # eTable 2, Panel B
    for lab, key, t in THRESHOLDS + [("ESRP score ≥31", None, 31)]:
        v = s if key is None else sc[key]
        op = E.operating_point(y, v, t)
        r[f"2B {lab}"] = (f"{f(100 * op['flagged'] / n, 2)}% | {ci(*op['sensitivity'])} | "
                          f"{ci(*op['specificity'])} | {ci(*op['ppv'])}")
    # eTable 5, Panel A
    for lab, col in [("Diagnosis codes only", "sepsis_icd"), ("Sepsis-3 criteria only", "sepsis3"),
                     ("Composite", "composite")]:
        yy = d[col].to_numpy(int); k = int(yy.sum()); lo, hi = E.wilson(k, n)
        pos, neg = E.score_counts(yy, s)
        r[f"5A {lab}"] = (f"{f(100 * k / n, 2)}% ({f(100 * lo, 2)}{EN}{f(100 * hi, 2)}) | {ci(*E.auroc(yy, s))} | "
                          f"{ci(E.auprc_counts(pos, neg), *E.bootstrap_integer(yy, s, E.auprc_counts))}")
    # eTable 7, Panel A
    for t in (10, 20, 21, 26, 30, 31, 35, 40, 45, 50, 55, 60):
        op = E.operating_point(y, s, t); share = op["flagged"] / n
        r[f"7A ≥{t}"] = (f"{op['flagged']:,} ({f(100 * share, 1)}%) | {ci(*op['sensitivity'])} | "
                              f"{ci(*op['specificity'])} | {ci(*op['ppv'])} | "
                              f"{f(E.alerts_per_ed_per_day(share, VISITS_PER_ED_PER_DAY[name]), 1)}")
    # eTable 8
    lp = INTERCEPT + SLOPE * s
    b_int, _ = E._irls(np.zeros((n, 0)), y, offset=lp)
    b_both, se_both = E._irls(lp, y)
    ks = np.arange(s.max() + 1); lp_k = INTERCEPT + SLOPE * ks
    variants = {"As applied": lp_k, "Intercept re-estimated": lp_k + b_int[0],
                "Intercept and slope re-estimated": b_both[0] + b_both[1] * lp_k}
    for lab, lin in variants.items():
        p_k = 1 / (1 + np.exp(-lin)); p = p_k[s]
        oe = oe_poisson(float(y.sum()), float(p.sum()))
        brier = lambda a, b, p_k=p_k: float((a * (1 - p_k) ** 2 + b * p_k ** 2).sum() / (a.sum() + b.sum()))
        pos, neg = E.score_counts(y, s)
        r[f"8 {lab}"] = f"{ci(*oe)} | {ci(brier(pos, neg), *E.bootstrap_integer(y, s, brier), k=5)}"
    cal = E.calibration(y, 1 / (1 + np.exp(-lp)))
    r["8 As applied, intercept"] = "+" + ci(*cal["intercept"]); r["8 slope"] = ci(*cal["slope"])
    r["8 Intercept re-estimated, offset"] = "+" + f(b_int[0], 4)
    r["8 Intercept and slope, coefficients"] = (f"a +{ci(b_both[0], b_both[0] - E.Z * se_both[0], b_both[0] + E.Z * se_both[0], 4)} "
                                                f"b {ci(b_both[1], b_both[1] - E.Z * se_both[1], b_both[1] + E.Z * se_both[1], 4)}")
    return r


def main(bidmc: str, stanford: str, out: str = "results/appendix_external.csv") -> pd.DataFrame:
    t = pd.DataFrame({"BIDMC": column("BIDMC", pd.read_csv(bidmc, low_memory=False)),
                      "Stanford": column("Stanford", pd.read_csv(stanford, low_memory=False))})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out)
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(t.to_string())
    return t


if __name__ == "__main__":
    main(*sys.argv[1:3])
