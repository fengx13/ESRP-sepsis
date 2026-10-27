"""Performance measures used in the ESRP study (Table 3 and the appendix).

Intervals: DeLong for AUROC and paired AUROC differences; Wilson for proportions;
Wald for calibration intercept and slope; Poisson for the observed:expected ratio;
the log scale for likelihood ratios; percentile bootstrap for AUPRC, Brier score
and the mean predicted risk of a tier.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

Z = 1.959964


# ---------------------------------------------------------------- discrimination
def _midrank(x):
    """Average ranks (ties share the mean rank), 1-based."""
    return rankdata(x, method="average")


def _delong(y, *scores):
    y = np.asarray(y, dtype=int)
    for s in scores:
        if np.isnan(np.asarray(s, float)).any():
            raise ValueError("score has missing values; drop or impute them before computing the AUROC")
    pos = [np.asarray(s, float)[y == 1] for s in scores]
    neg = [np.asarray(s, float)[y == 0] for s in scores]
    m, n = len(pos[0]), len(neg[0])
    tx = np.array([_midrank(p) for p in pos]); ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.r_[p, q]) for p, q in zip(pos, neg)])
    auc = tz[:, :m].sum(1) / (m * n) - (m + 1) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    return auc, cov


def auroc(y, score):
    """AUROC with its DeLong 95% CI: (estimate, lower, upper)."""
    auc, cov = _delong(y, score); se = np.sqrt(cov[0, 0])
    return float(auc[0]), float(auc[0] - Z * se), float(auc[0] + Z * se)


def delong_paired(y, score_a, score_b):
    """Paired DeLong test on the same encounters: (auc_a, auc_b, difference, lower, upper, p)."""
    auc, cov = _delong(y, score_a, score_b)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]; se = np.sqrt(max(var, 1e-300)); d = auc[0] - auc[1]
    return float(auc[0]), float(auc[1]), float(d), float(d - Z * se), float(d + Z * se), float(2 * (1 - norm.cdf(abs(d / se))))


def auprc(y, score):
    """Area under the precision-recall curve as average precision, one step per distinct score.

    Tied scores are treated as a single threshold, so the value does not depend on the
    order of the rows (the integer ESRP score has many ties). This equals
    sklearn.metrics.average_precision_score.
    """
    s = np.asarray(score, float); order = np.argsort(-s, kind="mergesort"); ss = s[order]; yy = np.asarray(y, int)[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(yy)[last]; precision = tp / (last + 1); recall = tp / max(yy.sum(), 1)
    return float(np.sum(np.diff(np.r_[0, recall]) * precision))


def bootstrap_ci(y, score, fn, n_boot=1000, seed=7):
    """Percentile bootstrap 95% CI of fn(y, score)."""
    rng = np.random.default_rng(seed); y = np.asarray(y); s = np.asarray(score); n = len(y)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n); vals[b] = fn(y[i], s[i])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def score_counts(y, score):
    """Positives and negatives at each value of an integer score 0..max."""
    s = np.asarray(score).astype(int); y = np.asarray(y).astype(int); k = s.max() + 1
    return np.bincount(s[y == 1], minlength=k), np.bincount(s[y == 0], minlength=k)


def auprc_counts(pos, neg):
    """Average precision from per-score counts (one threshold per score value)."""
    tp = np.cumsum(pos[::-1]); fp = np.cumsum(neg[::-1]); keep = (tp + fp) > 0
    tp, fp = tp[keep], fp[keep]
    return float(np.sum(np.diff(np.r_[0, tp / max(tp[-1], 1)]) * tp / (tp + fp)))


def bootstrap_integer(y, score, stat, n_boot=1000, seed=7):
    """Percentile bootstrap 95% CI of ``stat(pos, neg)`` for an integer score.

    Resamples encounters exactly as a row bootstrap with the same seed would, but
    evaluates the statistic on per-score counts, so it takes seconds rather than minutes.
    """
    s = np.asarray(score).astype(int); y = np.asarray(y).astype(int); k = s.max() + 1
    code = s * 2 + y; rng = np.random.default_rng(seed); n = len(s); vals = np.empty(n_boot)
    for b in range(n_boot):
        c = np.bincount(code[rng.integers(0, n, n)], minlength=2 * k).reshape(k, 2)
        vals[b] = stat(c[:, 1], c[:, 0])
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------- operating points
def wilson(k, n):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n; d = 1 + Z * Z / n; c = (p + Z * Z / (2 * n)) / d
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def youden_threshold(y, score, candidates=range(0, 101)):
    y = np.asarray(y, int); s = np.asarray(score, float)
    return max(candidates, key=lambda t: (s >= t)[y == 1].mean() - (s >= t)[y == 0].mean())


def operating_point(y, score, threshold):
    """Sensitivity, specificity, PPV, NPV at score ≥ threshold, each as (estimate, lower, upper)."""
    y = np.asarray(y, int); flag = np.asarray(score, float) >= threshold
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    out = {}
    for name, k, n in [("sensitivity", tp, tp + fn), ("specificity", tn, tn + fp), ("ppv", tp, tp + fp), ("npv", tn, tn + fn)]:
        out[name] = (k / n if n else np.nan, *wilson(k, n))
    out["flagged"] = int(flag.sum())
    return out


def alerts_per_ed_per_day(flagged_fraction, visits_per_ed_per_day):
    """Alerts per ED per day = proportion flagged × the cohort's mean daily attendance per ED."""
    return flagged_fraction * visits_per_ed_per_day


def npv_at_prevalence(sensitivity, specificity, prevalence):
    """NPV by Bayes' theorem at another prevalence, holding sensitivity and specificity fixed."""
    return specificity * (1 - prevalence) / (specificity * (1 - prevalence) + (1 - sensitivity) * prevalence)


# ---------------------------------------------------------------- calibration
def _irls(X, y, offset=None, intercept=True, iters=60):
    y = np.asarray(y, float); X = np.asarray(X, float).reshape(len(y), -1)
    if intercept:
        X = np.c_[np.ones(len(y)), X]
    off = np.zeros(len(y)) if offset is None else np.asarray(offset, float)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ b + off))); w = np.clip(p * (1 - p), 1e-10, None)
        h = (X.T * w) @ X; step = np.linalg.solve(h, X.T @ (y - p)); b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b, np.sqrt(np.diag(np.linalg.inv(h)))


def calibration(y, prob):
    """Calibration intercept (slope fixed at 1), slope, O:E ratio and Brier score of predicted probabilities."""
    y = np.asarray(y, float); p = np.clip(np.asarray(prob, float), 1e-12, 1 - 1e-12); lp = np.log(p / (1 - p))
    bi, si = _irls(np.zeros((len(y), 0)), y, offset=lp)
    bs, ss = _irls(lp, y)
    o, e = y.sum(), p.sum(); oe = o / e
    return {"intercept": (float(bi[0]), float(bi[0] - Z * si[0]), float(bi[0] + Z * si[0])),
            "slope": (float(bs[1]), float(bs[1] - Z * ss[1]), float(bs[1] + Z * ss[1])),
            "oe": (float(oe), float(oe * np.exp(-Z / np.sqrt(o))), float(oe * np.exp(Z / np.sqrt(o)))),
            "brier": float(np.mean((p - y) ** 2))}


def recalibrate_intercept(y, score, intercept, slope):
    """Re-estimate the intercept of a logistic mapping with the slope fixed (eTable 8)."""
    lp = intercept + slope * np.asarray(score, float)
    b, _ = _irls(np.zeros((len(lp), 0)), y, offset=lp)
    return intercept + float(b[0])


# ---------------------------------------------------------------- risk tiers
def tier_table(y, score, prob, cutpoints=(21, 31, 45), labels=("low", "moderate", "high", "very high"), n_boot=300, seed=7):
    """One row per tier: n, share, mean predicted risk (bootstrap CI), observed rate (Wilson CI),
    O:E (Poisson CI) and the stratum-specific positive likelihood ratio (log-scale CI)."""
    y = np.asarray(y, int); s = np.asarray(score, float); p = np.asarray(prob, float)
    tier = np.digitize(s, cutpoints); n1, n0 = int(y.sum()), int(len(y) - y.sum()); rng = np.random.default_rng(seed)
    rows = []
    for i, lab in enumerate(labels):
        m = tier == i; n = int(m.sum()); k = int(y[m].sum()); pm = p[m]
        boot = np.array([pm[rng.integers(0, n, n)].mean() for _ in range(n_boot)]) if n else np.array([np.nan])
        o, e = float(k), float(pm.sum()); oe = o / e if e else np.nan
        a = k / n1; b = (n - k) / n0; lr = a / b
        se = np.sqrt((1 - a) / max(k, 1) + (1 - b) / max(n - k, 1))
        rows.append({"tier": lab, "n": n, "share": n / len(y), "predicted": pm.mean(),
                     "predicted_ci": tuple(np.percentile(boot, [2.5, 97.5])), "observed": k, "observed_rate": k / n if n else np.nan,
                     "observed_ci": wilson(k, n), "oe": oe, "oe_ci": (oe * np.exp(-Z / np.sqrt(o)), oe * np.exp(Z / np.sqrt(o))) if o else (np.nan, np.nan),
                     "lr_pos": lr, "lr_pos_ci": (lr * np.exp(-Z * se), lr * np.exp(Z * se))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- decision curve
def net_benefit(y, prob, thresholds):
    """Net benefit per encounter at each threshold probability, plus treat-all."""
    y = np.asarray(y, int); p = np.asarray(prob, float); n = len(y); prev = y.mean(); out = []
    for t in thresholds:
        w = t / (1 - t); flag = p >= t
        out.append({"threshold": t, "model": ((flag & (y == 1)).sum() - (flag & (y == 0)).sum() * w) / n,
                    "treat_all": prev - (1 - prev) * w, "treat_none": 0.0})
    return pd.DataFrame(out)
