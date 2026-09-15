from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_ridge():
    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                Ridge(alpha=10.0),
            ),
        ]
    )


def build_random_forest():
    return RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )


def build_hist_gradient_boosting():
    return HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )


def build_model(model_name: str):
    if model_name == "ridge":
        return build_ridge()

    if model_name == "random_forest":
        return build_random_forest()

    if model_name == "hist_gradient_boosting":
        return build_hist_gradient_boosting()

    raise ValueError(
        f"Unknown model: {model_name}"
    )