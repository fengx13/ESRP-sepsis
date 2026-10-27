import math
import pytest
from esrp.score import esrp_score, esrp_points, risk_tier, predicted_probability, MAX_SCORE, score_frame

BASE = dict(heart_rate=70, systolic_bp=120, age=25, diastolic_bp=95, oxygen_saturation=99,
            respiratory_rate=16, fever_or_chills=0, hospitalisations_365d_ge2=0, fluid_electrolyte_disorder=0)

def test_minimum_and_maximum():
    assert esrp_score({**BASE, "systolic_bp": 180}) == 0
    worst = dict(heart_rate=110, systolic_bp=89, age=70, diastolic_bp=59, oxygen_saturation=93,
                 respiratory_rate=22, fever_or_chills=1, hospitalisations_365d_ge2=1, fluid_electrolyte_disorder=1)
    assert esrp_score(worst) == MAX_SCORE == 100

def test_table2_boundaries():
    # boundaries are inclusive at the lower end of each interval (Table 2: [90, 110) etc.)
    assert esrp_points({**BASE, "heart_rate": 89})["heart_rate"] == 0
    assert esrp_points({**BASE, "heart_rate": 90})["heart_rate"] == 3
    assert esrp_points({**BASE, "heart_rate": 110})["heart_rate"] == 6
    assert esrp_points({**BASE, "systolic_bp": 89.9})["systolic_bp"] == 14
    assert esrp_points({**BASE, "systolic_bp": 90})["systolic_bp"] == 3
    assert esrp_points({**BASE, "systolic_bp": 180})["systolic_bp"] == 0
    assert esrp_points({**BASE, "age": 29})["age"] == 0
    assert esrp_points({**BASE, "age": 30})["age"] == 9
    assert esrp_points({**BASE, "age": 70})["age"] == 14
    assert esrp_points({**BASE, "diastolic_bp": 59})["diastolic_bp"] == 11
    assert esrp_points({**BASE, "diastolic_bp": 60})["diastolic_bp"] == 6
    assert esrp_points({**BASE, "diastolic_bp": 90})["diastolic_bp"] == 0
    assert esrp_points({**BASE, "oxygen_saturation": 93})["oxygen_saturation"] == 11
    assert esrp_points({**BASE, "oxygen_saturation": 94})["oxygen_saturation"] == 3
    assert esrp_points({**BASE, "oxygen_saturation": 99})["oxygen_saturation"] == 0
    assert esrp_points({**BASE, "respiratory_rate": 17})["respiratory_rate"] == 0
    assert esrp_points({**BASE, "respiratory_rate": 18})["respiratory_rate"] == 3
    assert esrp_points({**BASE, "respiratory_rate": 22})["respiratory_rate"] == 9

def test_tiers():
    assert risk_tier(0) == "low" and risk_tier(20) == "low"
    assert risk_tier(21) == "moderate" and risk_tier(30) == "moderate"
    assert risk_tier(31) == "high" and risk_tier(44) == "high"
    assert risk_tier(45) == "very high" and risk_tier(100) == "very high"

def test_probability_mapping():
    # Table 2 footnote: p = 1 / (1 + exp(-(-7.2541 + 0.111705 * score)))
    assert abs(predicted_probability(0) - 1 / (1 + math.exp(7.2541))) < 1e-12
    assert abs(predicted_probability(20) - 0.00656) < 5e-5
    assert abs(predicted_probability(45) - 0.0973) < 5e-4
    assert abs(predicted_probability(100) - 0.9805) < 5e-4

def test_missing_raises():
    with pytest.raises(ValueError):
        esrp_score({**BASE, "heart_rate": float("nan")})

def test_frame_matches_row():
    pd = pytest.importorskip("pandas")
    rows = [BASE, {**BASE, "heart_rate": 115, "fever_or_chills": 1}, {**BASE, "age": 80, "oxygen_saturation": 90}]
    df = pd.DataFrame(rows)
    assert list(score_frame(df)) == [esrp_score(r) for r in rows]
