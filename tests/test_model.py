"""Smoke tests for the exported model: it loads, prices a known car sensibly, and meets the quality bar."""
import json
from pathlib import Path

import pandas as pd

from vehicle_model import FEATURES, PriceModel, add_features

ROOT = Path(__file__).resolve().parent.parent
# Quality bar for a model to ship. The committed stack scores 13.0 % / 0.89.
MAX_TEST_MAPE, MIN_TEST_R2 = 14.0, 0.85


def held_out_car():
    """The single-sample inference car the notebook held out of training (results/metrics.json)."""
    m = json.loads((ROOT / "results" / "metrics.json").read_text())["inference"]
    return pd.DataFrame([m["car"]]), m["actual_lakhs"]


def test_add_features_shape_and_age():
    car, _ = held_out_car()
    X = add_features(car)
    assert list(X.columns) == FEATURES
    assert X.age.iloc[0] == 2025 - 2012
    assert X.brand.iloc[0] == "TOYOTA"


def test_model_prices_held_out_car_within_interval():
    model = PriceModel.load(ROOT / "model" / "price_model.joblib")
    car, actual = held_out_car()
    X = add_features(car)
    price = float(model.predict(X)[0])
    lo, hi = (float(v[0]) for v in model.interval(X))
    assert lo < price < hi
    assert lo <= actual <= hi


def test_model_meets_quality_bar():
    mt = PriceModel.load(ROOT / "model" / "price_model.joblib").metrics
    assert mt["test_mape"] <= MAX_TEST_MAPE, mt
    assert mt["test_r2"] >= MIN_TEST_R2, mt


def test_app_renders_an_estimate():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()
    assert not at.exception, at.exception
    assert "lakhs" in at.metric[0].value
