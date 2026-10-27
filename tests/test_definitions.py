import numpy as np, pandas as pd
from esrp.definitions import (is_sepsis_icd, mimic_flags, sofa_coagulation, sofa_liver, sofa_renal,
                              sofa_cardiovascular, baseline_sofa, sepsis3, septic_shock, composite_outcome)

def test_icd_prefixes_both_versions():
    codes = pd.Series(["A41.9", "038.9", "R65.20", "J18.9", "99591", "T81.44XA"])
    ver = pd.Series([10, 9, 10, 10, 9, 10])
    assert is_sepsis_icd(codes, ver).tolist() == [True, True, True, False, True, True]
    # version-aware: an ICD-9 "038" prefix must not match when the row says ICD-10
    assert not is_sepsis_icd(pd.Series(["0389"]), pd.Series([10])).iloc[0]
    assert is_sepsis_icd(pd.Series(["0389"])).iloc[0]            # no version column: match either list

def test_mimics():
    f = mimic_flags(pd.Series(["E11.10", "K85.9", "R57.0", "I26.99", "J45.901", "A41.9"]), pd.Series([10]*6))
    assert f.sum().to_dict() == {"dka": 1, "pancreatitis": 1, "cardiogenic_shock": 1, "pulmonary_embolism": 1, "copd_asthma": 1}
    assert not f.iloc[5].any()

def test_sofa_components():
    assert sofa_coagulation([10, 30, 80, 120, 200]).tolist() == [4, 3, 2, 1, 0]
    assert sofa_liver([12, 6, 2, 1.2, 1.0]).tolist() == [4, 3, 2, 1, 0]
    assert sofa_renal([5, 3.5, 2, 1.2, 1.1]).tolist() == [4, 3, 2, 1, 0]
    assert sofa_cardiovascular([60, 60, 80], [0, 1, 0]).tolist() == [1, 3, 0]

def test_sepsis3_uses_baseline():
    com = pd.DataFrame({"cci_Renal": [1, 0], "cci_Liver2": [0, 0]})
    base = baseline_sofa(com)
    assert base.renal.tolist() == [2, 0]
    sofa = pd.DataFrame({"liver": [0, 0], "renal": [2, 2], "coagulation": [0, 0], "cardiovascular": [0, 0]})
    # same absolute SOFA; the patient with chronic renal disease does not meet the criterion
    assert sepsis3([True, True], sofa, base).tolist() == [False, True]
    assert sepsis3([False, True], sofa, base).tolist() == [False, True]

def test_shock_and_composite():
    assert septic_shock([1, 1, 0], [1, 1, 1], [2.5, 2.0, 3.0]).tolist() == [True, False, False]
    assert composite_outcome([0, 1, 0], [0, 0, 1]).tolist() == [False, True, True]
