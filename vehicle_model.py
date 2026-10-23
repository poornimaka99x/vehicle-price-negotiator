"""Features, model definitions and the exported price model, shared by the notebook and the app.

Everything the saved model needs lives in this module, so `model/price_model.joblib`
loads from app.py without the notebook.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Listings were scraped Dec 2024 - Feb 2025, so a car's age is measured from 2025.
REFERENCE_YEAR = 2025

# Raw CSV header -> tidy name used everywhere else.
RENAME = {
    "Brand": "brand", "Model": "model", "YOM": "yom", "Engine (cc)": "engine_cc",
    "Gear": "gear", "Fuel Type": "fuel", "Millage(KM)": "mileage_km", "Town": "town",
    "Date": "date", "Leasing": "leasing", "Condition": "condition",
    "AIR CONDITION": "air_condition", "POWER STEERING": "power_steering",
    "POWER MIRROR": "power_mirror", "POWER WINDOW": "power_window", "Price": "price",
}

# What a user (or the app) supplies about one car. Mileage is deliberately absent:
# in this dataset it is exactly 11,000 km x age for every row (see notebook section 1.1),
# so it carries no information beyond the year of manufacture.
INPUT_COLUMNS = [
    "brand", "model", "yom", "engine_cc", "gear", "fuel", "town",
    "leasing", "condition", "air_condition", "power_steering", "power_mirror", "power_window",
]

CATEGORICAL = [
    "brand", "model", "gear", "fuel", "town", "leasing", "condition",
    "air_condition", "power_steering", "power_mirror", "power_window",
]
NUMERIC = ["age", "engine_cc"]
FEATURES = CATEGORICAL + NUMERIC


def load_raw(path) -> pd.DataFrame:
    """Read the Kaggle CSV unchanged apart from tidy column names."""
    return pd.read_csv(path, index_col=0).rename(columns=RENAME)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """INPUT_COLUMNS -> model FEATURES."""
    out = pd.DataFrame(index=df.index)
    for c in CATEGORICAL:
        out[c] = df[c].astype(str).str.strip().str.upper()
    # Depreciation follows age, not the calendar year.
    out["age"] = (REFERENCE_YEAR - pd.to_numeric(df["yom"], errors="coerce")).clip(lower=0)
    out["engine_cc"] = pd.to_numeric(df["engine_cc"], errors="coerce")
    return out[FEATURES]


# ---------------------------------------------------------------- model definitions
# Every model below is fitted on log(price) and predicts log(price).

class MarketComparable:
    """What a buyer does by hand: the median price of the same make + model + year,
    backing off to make + model, then make, then the overall median."""
    keys = [["brand", "model", "age"], ["brand", "model"], ["brand"]]

    def fit(self, X, y):
        X = X.assign(y=y)
        self.tables = [X.groupby(k).y.median() for k in self.keys]
        self.global_ = float(np.median(y))
        return self

    def predict(self, X):
        out = pd.Series(np.nan, index=X.index)
        for k, t in zip(self.keys, self.tables):
            idx = pd.MultiIndex.from_frame(X[k]) if len(k) > 1 else X[k[0]]
            out = out.fillna(pd.Series(t.reindex(idx).values, index=X.index))
        return out.fillna(self.global_).values


class LGBModel:
    """LightGBM with native categorical splits. Categories are frozen at fit time;
    a category never seen in training becomes missing and takes LightGBM's 'missing' branch."""

    def __init__(self, **params):
        self.params = params

    def _frame(self, X):
        X = X.copy()
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c], categories=self.categories[c])
        return X

    def fit(self, X, y):
        import lightgbm as lgb
        self.categories = {c: sorted(X[c].unique().tolist()) for c in CATEGORICAL}
        self.m = lgb.LGBMRegressor(**self.params).fit(self._frame(X), y)
        return self

    def predict(self, X):
        return self.m.predict(self._frame(X))


class StackedModel:
    """Positive linear meta-learner over base models' log-price predictions.
    The weights are learned on out-of-fold predictions, never on the base models' own training fit."""

    def __init__(self, base_models: dict, weights: dict, intercept: float):
        self.base_models, self.weights, self.intercept = base_models, weights, intercept

    def predict(self, X):
        return self.intercept + sum(w * self.base_models[n].predict(X) for n, w in self.weights.items())


# ---------------------------------------------------------------- SHAP
# Exact TreeSHAP, in log-price units, summed back to the 13 input FEATURES.
# Stacking and voting are weighted sums of base models, and SHAP is additive,
# so an ensemble's SHAP is the same weighted sum of its base models' SHAP.

def _onehot_to_features(pipe, X, contrib):
    """Sum a pipeline's per-output-column SHAP back to the original feature it came from."""
    names = pipe[0].get_feature_names_out()          # e.g. cat__brand_TOYOTA, num__age
    owner = [max((f for f in FEATURES if n.startswith(f"cat__{f}_") or n == f"num__{f}"), key=len) for n in names]
    return pd.DataFrame(contrib, index=X.index, columns=names).T.groupby(owner).sum().T[FEATURES]


def explain(model, X):
    """(base value, SHAP DataFrame [rows x FEATURES]) with base + row sum == model.predict(X), both log price.
    Returns None for models with no tree structure to explain (market-comparable lookup, Ridge)."""
    X = X[FEATURES]
    if isinstance(model, StackedModel):
        parts = {n: explain(model.base_models[n], X) for n in model.weights}
        base = model.intercept + sum(w * parts[n][0] for n, w in model.weights.items())
        return base, sum(w * parts[n][1] for n, w in model.weights.items())
    if isinstance(model, LGBModel):
        c = model.m.predict(model._frame(X), pred_contrib=True)
        return float(c[0, -1]), pd.DataFrame(c[:, :-1], index=X.index, columns=FEATURES)
    kind = type(model).__name__
    if kind == "CatBoostRegressor":
        from catboost import Pool
        c = model.get_feature_importance(Pool(X, cat_features=model.get_cat_feature_indices()), type="ShapValues")
        return float(c[0, -1]), pd.DataFrame(c[:, :-1], index=X.index, columns=FEATURES)
    if kind == "Pipeline":
        est, Xt = model[-1], model[0].transform(X)
        if type(est).__name__ == "XGBRegressor":
            import xgboost as xgb
            c = est.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)
            return float(c[0, -1]), _onehot_to_features(model, X, c[:, :-1])
        if type(est).__name__ in ("RandomForestRegressor", "BaggingRegressor"):
            import shap
            trees = est.estimators_ if type(est).__name__ == "BaggingRegressor" else [est]
            Xd = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
            # A bag is the mean of its trees, so its SHAP is the mean of the trees' SHAP.
            exps = [shap.TreeExplainer(t) for t in trees]
            c = np.mean([e.shap_values(Xd, check_additivity=False) for e in exps], axis=0)
            base = float(np.mean([np.ravel(e.expected_value)[0] for e in exps]))
            return base, _onehot_to_features(model, X, c)
    return None


class PriceModel:
    """The exported model: any fitted log-price model plus an empirical 80 % interval."""

    def __init__(self, model, name, log_resid_quantiles, metrics=None):
        self.model, self.name = model, name
        self.q_lo, self.q_hi = log_resid_quantiles   # 10th / 90th pct of out-of-fold log residuals
        self.metrics = metrics or {}

    def predict_log(self, X):
        return np.asarray(self.model.predict(X[FEATURES]), dtype=float)

    def predict(self, X):
        """Point prediction in lakhs (1 lakh = Rs 100,000)."""
        return np.exp(self.predict_log(X))

    def interval(self, X):
        """80 % range in lakhs, from residuals the model never trained on."""
        p = self.predict_log(X)
        return np.exp(p + self.q_lo), np.exp(p + self.q_hi)

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path, compress=3)

    @staticmethod
    def load(path) -> "PriceModel":
        # joblib uses pickle, and loading a pickle can run code: load only a model you trained yourself.
        return joblib.load(path)
