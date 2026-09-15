from config import settings

from ml.external_forecasting.service import (
    get_future_external_factors,
)


forecast = get_future_external_factors(
    model_store_dir=settings.MODEL_STORE_DIR,
    start_date="2026-01-01",
    end_date="2026-12-01",
)

print("\nMODEL A 2026 FORECAST\n")

print(
    forecast.to_string(
        index=False
    )
)
output_path = "model_store/external_forecast/forecast_2026.csv"

forecast.to_csv(
    output_path,
    index=False
)

print(f"\nForecast saved to: {output_path}")