import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


def smape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = (
        np.abs(y_true) +
        np.abs(y_pred)
    )

    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return float(
        100
        * np.mean(
            2
            * np.abs(y_pred[mask] - y_true[mask])
            / denominator[mask]
        )
    )


def wape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return 0.0

    return float(
        100
        * np.sum(np.abs(y_true - y_pred))
        / denominator
    )


def calculate_metrics(y_true, y_pred):
    return {
        "MAE": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "sMAPE": smape(
            y_true,
            y_pred,
        ),
        "WAPE": wape(
            y_true,
            y_pred,
        ),
    }