"""Table 3, external validation columns (BIDMC and Stanford).

    python -m esrp.cohorts.bidmc ... --out results/bidmc_cohort.csv.gz
    python -m esrp.cohorts.stanford ... --out results/stanford_cohort.csv.gz
    python notebooks/table3_external.py results/bidmc_cohort.csv.gz results/stanford_cohort.csv.gz

Prints each row of Table 3 as published and writes results/table3_external.csv.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esrp import comparators as C, evaluation as E          # noqa: E402
from esrp.score import INTERCEPT, SLOPE, THRESHOLDS          # noqa: E402

D, EN = "·", "–"


def f(x, k=3):
    return f"{x:.{k}f}".replace(".", D)


def ci(est, lo, hi, k=3):
    return f"{f(est, k)} ({f(lo, k)}{EN}{f(hi, k)})"


def comparators(d: pd.DataFrame) -> dict:
    male = d.gender.astype(str).str.upper().str[0].eq("M").astype(int)
    abdominal = d.chief_complaint.astype(str).str.contains("abdominal pain|abd pain", case=False).astype(int)
    q = pd.DataFrame({"age": d.age, "resp_rate": d.respiratory_rate, "heart_rate": d.heart_rate,
                      "temp_c": d.temperature, "abdominal_pain": abdominal, "fever_or_chills": d.fever_or_chills,
                      "male": male, "triage_acuity": d.triage_acuity, "spo2": d.oxygen_saturation,
                      "systolic_bp": d.systolic_bp})
    return {
        "Triage acuity (ESI)": C.esi_reversed(d.triage_acuity),
        "qSOFA": C.qsofa(d.respiratory_rate, d.systolic_bp),
        "NEWS2": C.news2(d.respiratory_rate, d.temperature, d.systolic_bp, d.heart_rate,
                         components=C.NEWS2_COMPONENTS_BENCHMARK),
        "SIRS": C.sirs(d.temperature, d.heart_rate, d.respiratory_rate, d.wbc_6h),
        "qSepsis variable set": C.transfer_model(q, "qsepsis_10"),
    }


def column(d: pd.DataFrame) -> dict:
    y = d.composite.to_numpy(int); s = d.esrp.to_numpy(int); n = len(y); k = int(y.sum())
    pos, neg = E.score_counts(y, s)
    prob = 1 / (1 + np.exp(-(INTERCEPT + SLOPE * s)))
    p_k = 1 / (1 + np.exp(-(INTERCEPT + SLOPE * np.arange(len(pos)))))
    brier_counts = lambda a, b: float((a * (1 - p_k) ** 2 + b * p_k ** 2).sum() / (a.sum() + b.sum()))
    r = {"Encounters, n": f"{n:,}", "Sepsis, n (%)": f"{k:,} ({f(100 * k / n, 2)}%)"}
    r["AUROC"] = ci(*E.auroc(y, s))
    r["AUPRC"] = ci(E.auprc_counts(pos, neg), *E.bootstrap_integer(y, s, E.auprc_counts))
    t = E.youden_threshold(y, s); op = E.operating_point(y, s, t)
    r["Threshold"] = f"≥{t}"
    for name, key in [("Sensitivity", "sensitivity"), ("Specificity", "specificity"),
                      ("Positive predictive value", "ppv"), ("Negative predictive value", "npv")]:
        r[name] = ci(*op[key])
    cal = E.calibration(y, prob)
    a, lo, hi = cal["intercept"]; r["Calibration intercept"] = f"+{f(a)} ({f(lo)}{EN}{f(hi)})"
    r["Calibration slope"] = ci(*cal["slope"])
    r["Observed:expected ratio"] = ci(*cal["oe"], k=2)
    r["Brier score"] = ci(cal["brier"], *E.bootstrap_integer(y, s, brier_counts), k=4)
    for name, v in comparators(d).items():
        r[name] = ci(*E.auroc(y, v))
    tiers = E.tier_table(y, s, prob, cutpoints=THRESHOLDS)
    for _, row in tiers.iterrows():
        r[f"Tier {row.tier}"] = (f"{row.n:,} ({f(100 * row.share, 1)}%) | "
                                 f"{f(100 * row.predicted, 2)} ({f(100 * row.predicted_ci[0], 2)}{EN}{f(100 * row.predicted_ci[1], 2)}) | "
                                 f"{row.observed:,} ({f(100 * row.observed_rate, 2)}; {f(100 * row.observed_ci[0], 2)}{EN}{f(100 * row.observed_ci[1], 2)}) | "
                                 f"{ci(row.oe, *row.oe_ci, k=2)} | {ci(row.lr_pos, *row.lr_pos_ci, k=2)}")
    return r


def main(bidmc: str, stanford: str, out: str = "results/table3_external.csv") -> pd.DataFrame:
    cols = {"BIDMC": column(pd.read_csv(bidmc, low_memory=False)),
            "Stanford": column(pd.read_csv(stanford, low_memory=False))}
    t = pd.DataFrame(cols)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(out)
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(t.to_string())
    return t


if __name__ == "__main__":
    main(*sys.argv[1:3])
