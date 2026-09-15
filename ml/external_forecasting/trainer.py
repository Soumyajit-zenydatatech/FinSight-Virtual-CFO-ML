from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.external_forecasting.config import (
    EXTERNAL_FACTORS,
    MODEL_A_FILENAME,
    MODEL_A_METADATA_FILENAME,
    MIN_TRAIN_ROWS,
    MODEL_CANDIDATES,
)
from ml.external_forecasting.data import (
    complete_monthly_index,
    load_external_history,
)
from ml.external_forecasting.features import (
    create_factor_features,
)
from ml.external_forecasting.metrics import (
    calculate_metrics,
)
from ml.external_forecasting.models import (
    build_model,
)


def _prepare_factor_data(
    df: pd.DataFrame,
    factor: str,
):
    """
    Prepare supervised learning data.

    The model predicts the monthly change:

        delta_t = value_t - value_(t-1)

    Features are constructed only from values available
    before month t.
    """
    data = df[
        ["date", factor]
    ].copy()

    data[factor] = pd.to_numeric(
        data[factor],
        errors="coerce",
    )

    data = data.dropna(
        subset=[factor]
    )

    data = create_factor_features(
        data,
        factor,
    )

    data["target_delta"] = (
        data[factor]
        - data[factor].shift(1)
    )

    data = (
        data
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )

    feature_columns = [
        column
        for column in data.columns
        if column not in {
            "date",
            factor,
            "target_delta",
        }
    ]

    return data, feature_columns


def _walk_forward_splits(
    n_rows: int,
    min_train_size: int,
):
    """
    Expanding-window time-series validation.
    """
    for test_index in range(
        min_train_size,
        n_rows,
    ):
        train_indices = np.arange(
            0,
            test_index,
        )

        test_indices = np.array(
            [test_index]
        )

        yield train_indices, test_indices


def _evaluate_model(
    data: pd.DataFrame,
    feature_columns,
    factor: str,
    model_name: str,
):
    """
    Evaluate one-step-ahead level forecasts.

    The model predicts a delta, then the delta is added
    to the previous actual level.
    """
    if len(data) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Not enough rows for {factor}: "
            f"{len(data)}"
        )

    X = data[feature_columns]
    y_delta = data["target_delta"]

    actual_levels = []
    predicted_levels = []

    min_train_size = max(
        18,
        min(36, len(data) // 2),
    )

    for train_idx, test_idx in _walk_forward_splits(
        len(data),
        min_train_size,
    ):
        model = build_model(model_name)

        X_train = X.iloc[train_idx]
        y_train = y_delta.iloc[train_idx]

        X_test = X.iloc[test_idx]

        model.fit(
            X_train,
            y_train,
        )

        predicted_delta = float(
            model.predict(X_test)[0]
        )

        test_position = test_idx[0]

        previous_level = float(
            data.iloc[
                test_position - 1
            ][factor]
        )

        predicted_level = (
            previous_level
            + predicted_delta
        )

        actual_level = float(
            data.iloc[
                test_position
            ][factor]
        )

        predicted_levels.append(
            predicted_level
        )

        actual_levels.append(
            actual_level
        )

    return calculate_metrics(
        np.array(actual_levels),
        np.array(predicted_levels),
    )


def _calculate_delta_bounds(
    data: pd.DataFrame,
    factor: str,
):
    """
    Calculate robust historical monthly-change limits.

    These are used only as a safety guard during recursive
    forecasting.
    """
    deltas = (
        data[factor]
        .diff()
        .dropna()
    )

    if len(deltas) == 0:
        return {
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "min": float(
            deltas.quantile(0.01)
        ),
        "max": float(
            deltas.quantile(0.99)
        ),
    }


def train_external_model(
    model_store_dir: str,
):
    model_store = Path(
        model_store_dir
    )

    model_store.mkdir(
        parents=True,
        exist_ok=True,
    )

    history = load_external_history(
        model_store_dir
    )

    history = complete_monthly_index(
        history
    )

    factors = [
        factor
        for factor in EXTERNAL_FACTORS
        if factor in history.columns
    ]

    if not factors:
        raise ValueError(
            "No external factors available."
        )

    models = {}
    best_models = {}
    metrics = {}
    feature_columns_by_factor = {}
    delta_bounds = {}

    for factor in factors:
        data, feature_columns = (
            _prepare_factor_data(
                history,
                factor,
            )
        )

        if len(data) < MIN_TRAIN_ROWS:
            raise ValueError(
                f"{factor} has only "
                f"{len(data)} usable rows."
            )

        feature_columns_by_factor[
            factor
        ] = feature_columns

        delta_bounds[
            factor
        ] = _calculate_delta_bounds(
            data,
            factor,
        )

        model_scores = {}

        # -------------------------------------------------
        # Naive baseline:
        # prediction = previous month's actual level
        # -------------------------------------------------
        naive_actual = []
        naive_predicted = []

        min_train_size = max(
            18,
            min(36, len(data) // 2),
        )

        for train_idx, test_idx in (
            _walk_forward_splits(
                len(data),
                min_train_size,
            )
        ):
            test_position = test_idx[0]

            previous_level = float(
                data.iloc[
                    test_position - 1
                ][factor]
            )

            actual_level = float(
                data.iloc[
                    test_position
                ][factor]
            )

            naive_predicted.append(
                previous_level
            )

            naive_actual.append(
                actual_level
            )

        naive_metrics = calculate_metrics(
            np.array(naive_actual),
            np.array(naive_predicted),
        )

        model_scores["naive"] = naive_metrics

        # -------------------------------------------------
        # ML candidates
        # -------------------------------------------------
        for model_name in MODEL_CANDIDATES:
            if model_name == "naive":
                continue

            model_metrics = _evaluate_model(
                data,
                feature_columns,
                factor,
                model_name,
            )

            model_scores[
                model_name
            ] = model_metrics

        # -------------------------------------------------
        # Select lowest WAPE
        # -------------------------------------------------
        best_model_name = min(
            model_scores,
            key=lambda name:
                model_scores[name]["WAPE"],
        )

        best_models[
            factor
        ] = best_model_name

        metrics[
            factor
        ] = model_scores

        # If naive wins, we don't need an ML model.
        if best_model_name == "naive":
            models[factor] = None

        else:
            final_model = build_model(
                best_model_name
            )

            final_data = data.copy()

            final_model.fit(
                final_data[
                    feature_columns
                ],
                final_data[
                    "target_delta"
                ],
            )

            models[
                factor
            ] = final_model

    package = {
        "model_version": "A.2-delta",
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "factors": factors,
        "models": models,
        "best_models": best_models,
        "metrics": metrics,
        "feature_columns": (
            feature_columns_by_factor
        ),
        "delta_bounds": delta_bounds,
        "training_start": str(
            history["date"].min().date()
        ),
        "training_end": str(
            history["date"].max().date()
        ),
        "training_rows": len(history),
    }

    output_dir = model_store / "external_forecast"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = (
        output_dir
        / MODEL_A_FILENAME
    )

    metadata_path = (
        output_dir
        / MODEL_A_METADATA_FILENAME
    )

    joblib.dump(
        package,
        model_path,
    )

    metadata = {
        "model_version": package[
            "model_version"
        ],
        "trained_at": package[
            "trained_at"
        ],
        "factors": factors,
        "best_models": best_models,
        "metrics": metrics,
        "delta_bounds": delta_bounds,
        "training_start": package[
            "training_start"
        ],
        "training_end": package[
            "training_end"
        ],
        "training_rows": package[
            "training_rows"
        ],
    }

    import json

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    return {
        "model_path": str(
            model_path
        ),
        "metadata_path": str(
            metadata_path
        ),
        "training_rows": len(
            history
        ),
        "training_start": str(
            history["date"].min().date()
        ),
        "training_end": str(
            history["date"].max().date()
        ),
        "best_models": best_models,
        "metrics": metrics,
    }