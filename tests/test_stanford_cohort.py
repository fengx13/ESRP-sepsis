"""End-to-end check against the published Stanford numbers.

Needs the MC-MED v1.0.0 files; set ESRP_MCMED=/path/to/mc-med-1.0.0 to run it.
"""
import os
from pathlib import Path

import pytest

MCMED = os.environ.get("ESRP_MCMED")
pytestmark = pytest.mark.skipif(not MCMED or not Path(MCMED).exists(), reason="ESRP_MCMED not set")


def test_stanford_reproduces_published_counts_and_auroc():
    from esrp.cohorts import stanford
    from esrp.evaluation import auroc, auprc
    d = stanford.build(MCMED)
    assert len(d) == 118_385
    assert int(d.sepsis_icd.sum()) == 843
    assert int(d.sepsis3.sum()) == 3_809
    assert int(d.septic_shock.sum()) == 490
    assert int(d.composite.sum()) == 4_228
    a, lo, hi = auroc(d.composite, d.esrp)
    assert (round(a, 3), round(lo, 3), round(hi, 3)) == (0.822, 0.816, 0.828)
    assert round(auprc(d.composite, d.esrp), 3) == 0.181
