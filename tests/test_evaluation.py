import numpy as np
from esrp.evaluation import auroc, delong_paired, auprc, wilson, operating_point, calibration, net_benefit, npv_at_prevalence

def test_auroc_perfect_and_ties():
    assert auroc([0, 0, 1, 1], [1, 2, 3, 4])[0] == 1.0
    assert auroc([0, 1, 0, 1], [1, 1, 1, 1])[0] == 0.5

def test_auprc_order_invariant_with_ties():
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]); s = np.array([3, 3, 3, 2, 2, 1, 1, 1])
    a = auprc(y, s); rng = np.random.default_rng(1)
    for _ in range(5):
        i = rng.permutation(len(y)); assert abs(auprc(y[i], s[i]) - a) < 1e-12
    # thresholds ≥3: P 1/3 R 1/3; ≥2: P 2/5 R 2/3; ≥1: P 3/8 R 1
    assert abs(a - (1/3*1/3 + 1/3*2/5 + 1/3*3/8)) < 1e-12

def test_wilson_and_operating_point():
    lo, hi = wilson(50, 100); assert 0.40 < lo < 0.41 and 0.59 < hi < 0.60
    op = operating_point([1, 1, 0, 0], [5, 1, 5, 1], 3)
    assert op["sensitivity"][0] == 0.5 and op["specificity"][0] == 0.5

def test_calibration_of_perfect_probabilities():
    rng = np.random.default_rng(0); p = rng.uniform(0.01, 0.5, 200_000); y = rng.uniform(size=p.size) < p
    c = calibration(y, p)
    assert abs(c["intercept"][0]) < 0.03 and abs(c["slope"][0] - 1) < 0.03 and abs(c["oe"][0] - 1) < 0.02

def test_paired_delong_identical_scores():
    a, b, d, lo, hi, p = delong_paired([0, 1, 0, 1, 1], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5.5])
    assert a == b and d == 0.0

def test_net_benefit_and_npv():
    nb = net_benefit([1, 0, 0, 0], [0.9, 0.1, 0.1, 0.1], [0.5])
    assert abs(nb.model[0] - 0.25) < 1e-12 and abs(nb.treat_all[0] - (0.25 - 0.75)) < 1e-12
    assert abs(npv_at_prevalence(0.95, 0.30, 0.10) - 0.30*0.9/(0.30*0.9 + 0.05*0.1)) < 1e-12
