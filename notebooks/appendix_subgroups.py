"""eTable 9, Panels C and D, external validation: subgroups and period of arrival.

    python notebooks/appendix_subgroups.py results/bidmc_cohort.csv.gz results/stanford_cohort.csv.gz

Febrile and afebrile use the first triage temperature as recorded (before
imputation); encounters without one are in neither group.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from esrp import evaluation as E                              # noqa: E402

D, EN = "·", "–"


def f(x, k=3):
    return f"{x:.{k}f}".replace(".", D)


def ci(t, k=3):
    return f"{f(t[0], k)} ({f(t[1], k)}{EN}{f(t[2], k)})"


def prop(k, n):
    return (k / n, *E.wilson(k, n)) if n else (np.nan,) * 3


def row(y, s):
    n = len(y); k = int(y.sum()); neg = s < 21
    p = prop(k, n)
    npv = prop(int((neg & (y == 0)).sum()), int(neg.sum()))
    return (f"{n:,} | {f(100 * p[0], 2)}% ({f(100 * p[1], 2)}{EN}{f(100 * p[2], 2)}) | {ci(E.auroc(y, s))} | "
            f"{ci(prop(int(((s >= 31) & (y == 1)).sum()), k))} | {ci(prop(int(((s < 31) & (y == 0)).sum()), n - k))} | "
            f"{ci(npv) if neg.any() else '—'} | {int((neg & (y == 1)).sum()):,}")


def subgroups(d: pd.DataFrame) -> dict:
    y = d.composite.to_numpy(int); s = d.esrp.to_numpy(int)
    t = d.temperature.where(d.temperature_imputed == 0)
    old = d.age.to_numpy() >= 70; afe = (t < 38.0).to_numpy(); feb = (t >= 38.0).to_numpy()
    fc = d.fever_or_chills.to_numpy() == 1; imm = d.immunocompromise.to_numpy() == 1
    groups = {"All encounters": np.ones(len(d), bool), "Age 70 years or older": old, "Age under 70 years": ~old,
              "Afebrile at triage, below 38·0 °C": afe, "Febrile at triage, 38·0 °C or above": feb,
              "No fever or chills complaint": ~fc, "Fever or chills complaint": fc,
              "Coded immunocompromising condition": imm, "No coded immunocompromising condition": ~imm,
              "Afebrile and no fever or chills complaint": afe & ~fc, "Age 70 or older and afebrile": old & afe}
    out = {f"9C {k}": row(y[m], s[m]) for k, m in groups.items()}
    out["9C note: no recorded temperature"] = f"{f(100 * t.isna().mean(), 2)}%"
    return out


def periods(d: pd.DataFrame) -> dict:
    y = d.composite.to_numpy(int); s = d.esrp.to_numpy(int); out = {}
    for g in sorted(d.anchor_year_group.dropna().unique()):
        m = (d.anchor_year_group == g).to_numpy(); k = int(y[m].sum()); p = prop(k, int(m.sum()))
        out[f"9D {g.replace(' - ', ' to ')}"] = (f"{int(m.sum()):,} | {f(100 * p[0], 2)} ({f(100 * p[1], 2)}{EN}{f(100 * p[2], 2)}) | "
                                                 f"{ci(E.auroc(y[m], s[m]))}")
    out["9D no anchor group"] = f"{int(d.anchor_year_group.isna().sum()):,}"
    return out


def main(bidmc, stanford, out="results/appendix_subgroups.csv"):
    b = pd.read_csv(bidmc, low_memory=False); st = pd.read_csv(stanford, low_memory=False)
    t = pd.DataFrame({"BIDMC": {**subgroups(b), **periods(b)}, "Stanford": subgroups(st)})
    Path(out).parent.mkdir(parents=True, exist_ok=True); t.to_csv(out)
    with pd.option_context("display.max_colwidth", 200, "display.width", 250):
        print(t.to_string())
    return t


if __name__ == "__main__":
    main(*sys.argv[1:3])
