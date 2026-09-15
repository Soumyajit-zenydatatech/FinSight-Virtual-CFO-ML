from pathlib import Path
import pandas as pd

from ml.external_forecasting.config import (
    MODEL_A_FILENAME,
)
from ml.external_forecasting.data import (
    load_external_history,
)
from ml.external_forecasting.predictor import (
    forecast_external_factors,
)


def get_model_a_path(
    model_store_dir: str,
):
    return (
        Path(model_store_dir)
        / "external_forecast"
        / MODEL_A_FILENAME
    )


def get_future_external_factors(
    model_store_dir: str,
    start_date: str,
    end_date: str,
):
    model_path = get_model_a_path(
        model_store_dir
    )

    if not model_path.exists():
        raise FileNotFoundError(
            "Model A has not been trained yet."
        )

    historical_df = (
        load_external_history(
            model_store_dir
        )
    )

    return forecast_external_factors(
        model_file_path=str(model_path),
        start_date=start_date,
        end_date=end_date,
        historical_df=historical_df,
    )
def get_future_external_factors_from_csv(
    model_store_dir: str,
    start_date: str,
    end_date: str,
):
    forecast_path = (
        Path(model_store_dir)
        / "external_forecast"
        / "forecast_2026.csv"
    )

    if not forecast_path.exists():
        raise FileNotFoundError(
            f"External forecast CSV not found: {forecast_path}"
        )

    forecast = pd.read_csv(forecast_path)

    forecast["date"] = (
        pd.to_datetime(forecast["date"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    start = (
        pd.to_datetime(start_date)
        .to_period("M")
        .to_timestamp()
    )

    end = (
        pd.to_datetime(end_date)
        .to_period("M")
        .to_timestamp()
    )

    forecast = forecast[
        (forecast["date"] >= start)
        & (forecast["date"] <= end)
    ].copy()

    return forecast.reset_index(drop=True)