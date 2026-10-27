"""Data behind the external-validation figures (BIDMC and Stanford panels).

    python notebooks/figures_external.py results/bidmc_cohort.csv.gz results/stanford_cohort.csv.gz

Writes to results/:
  figure2_roc.csv        sensitivity at fixed 1 - specificity for ESRP, ESI, qSOFA, NEWS2, SIRS (Figure 2, B and C)
  efigure2_deciles.csv   mean predicted risk and observed sepsis by tenth of predicted risk (eFigure 2)
  efigure3_dca.csv       net benefit per 1,000 encounters, threshold 0.5% to 30% (eFigure 3)
and figures_external.png, a plain rendering of all three.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esrp import comparators as C, evaluation as E          # noqa: E402
from esrp.score import INTERCEPT, SLOPE                       # noqa: E402

FPR_GRID = [0, .01, .02, .03, .05, .075, .1, .15, .2, .25, .3, .4, .5, .6, .7, .8, .9, 1.0]
DCA_THRESHOLDS = [0.5] + list(range(1, 31))                  # per cent


def roc_points(y, s):
    """ROC vertices, one per distinct score value (ties form a single straight segment)."""
    y = np.asarray(y, int); s = np.asarray(s, float)
    ts = np.unique(s)[::-1]; P = y.sum(); N = len(y) - P
    tp = np.array([((s >= t) & (y == 1)).sum() for t in ts]); fp = np.array([((s >= t) & (y == 0)).sum() for t in ts])
    return np.r_[0, fp / N], np.r_[0, tp / P]


def roc_grid(y, s):
    fpr, tpr = roc_points(y, s)
    return np.interp(FPR_GRID, fpr, tpr)


def logistic_calibrated(score, y):
    """A comparator score mapped to a probability by a logistic fit in the same cohort (decision curves)."""
    x = np.asarray(score, float)
    b, _ = E._irls(x, y)
    return 1 / (1 + np.exp(-(b[0] + b[1] * x)))


def site(name, d):
    y = d.composite.to_numpy(int); s = d.esrp.to_numpy(float)
    news2 = C.news2(d.respiratory_rate, d.temperature, d.systolic_bp, d.heart_rate,
                    components=C.NEWS2_COMPONENTS_BENCHMARK)
    qsofa = C.qsofa(d.respiratory_rate, d.systolic_bp)
    sc = {"ESRP": s, "ESI": C.esi_reversed(d.triage_acuity), "qSOFA": qsofa, "NEWS2": news2,
          "SIRS": C.sirs(d.temperature, d.heart_rate, d.respiratory_rate, d.wbc_6h)}
    roc = pd.DataFrame({k: roc_grid(y, v) for k, v in sc.items()}, index=pd.Index(FPR_GRID, name="fpr"))
    roc.insert(0, "cohort", name)
    auc = {k: E.auroc(y, v)[0] for k, v in sc.items()}
    # eFigure 2: tenths of predicted risk; ties broken by cohort row order (subject, then arrival)
    p = 1 / (1 + np.exp(-(INTERCEPT + SLOPE * s)))
    q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
    dec = pd.DataFrame({"tenth": q, "predicted": p, "observed": y}).groupby("tenth").mean().reset_index()
    dec.insert(0, "cohort", name)
    # eFigure 3
    probs = {"ESRP": p, "qSOFA": logistic_calibrated(qsofa, y), "NEWS2": logistic_calibrated(news2, y),
             "MEWS": logistic_calibrated(C.mews(d.systolic_bp, d.heart_rate, d.respiratory_rate, d.temperature), y)}
    nb = None
    for k, v in probs.items():
        t = E.net_benefit(y, v, [x / 100 for x in DCA_THRESHOLDS])
        if nb is None:
            nb = pd.DataFrame({"cohort": name, "pt": DCA_THRESHOLDS, "treat_all": 1000 * t.treat_all})
        nb[k] = 1000 * t.model.to_numpy()
    return roc, auc, dec, nb


def plot(roc, auc, dec, dca, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 2, figsize=(9, 12))
    for j, nm in enumerate(["BIDMC", "Stanford"]):
        r = roc[roc.cohort == nm]
        for k in ["ESRP", "ESI", "qSOFA", "NEWS2", "SIRS"]:
            ax[0, j].plot(r.index, r[k], label=f"{k} {auc[nm][k]:.3f}")
        ax[0, j].plot([0, 1], [0, 1], ":", color="grey"); ax[0, j].set_title(f"{nm}: ROC"); ax[0, j].legend(fontsize=8)
        dd = dec[dec.cohort == nm]; hi = dd[["predicted", "observed"]].to_numpy().max() * 1.08
        ax[1, j].plot(dd.predicted, dd.observed, "-o"); ax[1, j].plot([0, hi], [0, hi], ":", color="grey")
        ax[1, j].set_title(f"{nm}: calibration by tenth"); ax[1, j].set_xlabel("predicted"); ax[1, j].set_ylabel("observed")
        c = dca[dca.cohort == nm]
        for k in ["treat_all", "ESRP", "qSOFA", "NEWS2", "MEWS"]:
            ax[2, j].plot(c.pt, c[k], label=k)
        ax[2, j].axhline(0, color="grey", lw=0.8); ax[2, j].set_ylim(-5, None)
        ax[2, j].set_title(f"{nm}: net benefit per 1,000"); ax[2, j].set_xlabel("threshold probability, %"); ax[2, j].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out, dpi=150)


def main(bidmc, stanford, outdir="results"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    res = [site(nm, pd.read_csv(f, low_memory=False)) for nm, f in [("BIDMC", bidmc), ("Stanford", stanford)]]
    roc = pd.concat([r[0] for r in res]); dec = pd.concat([r[2] for r in res]); dca = pd.concat([r[3] for r in res])
    auc = {"BIDMC": res[0][1], "Stanford": res[1][1]}
    roc.to_csv(out / "figure2_roc.csv"); dec.to_csv(out / "efigure2_deciles.csv", index=False)
    dca.to_csv(out / "efigure3_dca.csv", index=False)
    plot(roc, auc, dec, dca, out / "figures_external.png")
    print({k: {m: round(v, 4) for m, v in a.items()} for k, a in auc.items()})


if __name__ == "__main__":
    main(*sys.argv[1:3])
