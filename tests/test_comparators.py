import numpy as np
from esrp.comparators import qsofa, sirs, news2, mews, rems, cart, mean_arterial_pressure, transfer_model, NEWS2_COMPONENTS_BENCHMARK
import pandas as pd

def test_qsofa():
    assert qsofa([22, 21, 22], [100, 101, 101], gcs=[15, 15, 14]).tolist() == [2, 0, 2]

def test_sirs_missing_wbc_scores_zero():
    assert sirs([38.1, 36.0], [91, 90], [21, 20], wbc=[np.nan, 13]).tolist() == [3, 1]

def test_news2_full_vs_benchmark():
    full = news2([25], [39.5], [85], [135], spo2=[90], altered_consciousness=[1])
    assert full.tolist() == [3 + 2 + 3 + 3 + 3 + 3]
    four = news2([25], [39.5], [85], [135], spo2=[90], altered_consciousness=[1], components=NEWS2_COMPONENTS_BENCHMARK)
    assert four.tolist() == [3 + 2 + 3 + 3]

def test_mews_rems_cart():
    assert mews([65], [35], [5], [34]).tolist() == [3 + 2 + 2 + 2]
    assert rems([80], mean_arterial_pressure([60], [40]), [45], [8], [70]).tolist() == [6 + 4 + 3 + 2 + 4]
    assert rems([30], [69.7], [80], [16], [99]).tolist() == [2]          # continuous MAP bin, no gap
    assert cart([75], [30], [150], [30]).tolist() == [9 + 22 + 13 + 13]

def test_transfer_model_imputes_medians():
    df = pd.DataFrame({"age": [np.nan], "resp_rate": [18.0], "heart_rate": [80.0], "temp_c": [36.72], "abdominal_pain": [0],
                       "fever_or_chills": [0], "male": [0], "triage_acuity": [3], "spo2": [98.0], "systolic_bp": [129.0]})
    p = transfer_model(df, "qsepsis_10")
    lp = 6.7749614 + .0360216*47.4 + .0300349*18 + .0142306*80 - .1402717*36.72 - 1.0555912*3 - .0474856*98 - .0150395*129
    assert abs(p[0] - 1/(1+np.exp(-lp))) < 1e-9
