"""End-to-end check against the published BIDMC numbers.

Needs MIMIC-IV-ED v1.0 and v2.2 and MIMIC-IV v1.0 and v3.1. Set ESRP_MIMIC_ROOT to the
folder holding mimic-iv-ed-1.0/, mimic-iv-1.0/, mimic-iv-ed-2.2/ and mimic-iv-3.1/
(optionally ESRP_CACHE for the filtered extracts) to run it.
"""
import os
from pathlib import Path

import pytest

ROOT = os.environ.get("ESRP_MIMIC_ROOT")
pytestmark = pytest.mark.skipif(not ROOT or not Path(ROOT).exists(), reason="ESRP_MIMIC_ROOT not set")


def test_bidmc_reproduces_published_counts_and_auroc():
    from esrp.cohorts import bidmc
    from esrp.evaluation import auroc, auprc
    r = Path(ROOT)
    d = bidmc.build(r / "mimic-iv-ed-1.0", r / "mimic-iv-1.0", r / "mimic-iv-ed-2.2", r / "mimic-iv-3.1",
                    cache_dir=os.environ.get("ESRP_CACHE"))
    assert len(d) == 448_804
    assert int(d.sepsis_icd.sum()) == 11_421
    assert int(d.sepsis3.sum()) == 17_305
    assert int(d.septic_shock.sum()) == 2_529
    assert int(d.composite.sum()) == 23_649
    a, lo, hi = auroc(d.composite, d.esrp)
    assert (round(a, 3), round(lo, 3), round(hi, 3)) == (0.784, 0.781, 0.786)
    assert round(auprc(d.composite, d.esrp), 3) == 0.210
