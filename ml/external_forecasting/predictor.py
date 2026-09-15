from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.external_forecasting.features import (
    create_factor_features,
)


def forecast_external_factors(
    model_file_path: str,
    start_date: str,
    end_date: str,
    historical_df: pd.DataFrame,
):
    """
    Forecast external factors recursively.

    Model A predicts monthly changes rather than absolute
    factor levels.

        predicted_level =
            previous_level + predicted_delta
    """
    model_path = Path(
        model_file_path
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model A not found: {model_path}"
        )

    package = joblib.load(
        model_path
    )

    factors = package[
        "factors"
    ]

    models = package[
        "models"
    ]

    feature_columns = package[
        "feature_columns"
    ]

    delta_bounds = package.get(
        "delta_bounds",
        {},
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

    future_dates = pd.date_range(
        start=start,
        end=end,
        freq="MS",
    )

    history = historical_df.copy()

    history["date"] = (
        pd.to_datetime(
            history["date"]
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    for factor in factors:
        history[factor] = pd.to_numeric(
            history[factor],
            errors="coerce",
        )

    history = (
        history[
            ["date"] + factors
        ]
        .sort_values("date")
        .groupby(
            "date",
            as_index=False,
        )
        .median(numeric_only=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Complete the monthly index.
    full_index = pd.date_range(
        start=history["date"].min(),
        end=history["date"].max(),
        freq="MS",
    )

    history = (
        history
        .set_index("date")
        .reindex(full_index)
    )

    history.index.name = "date"

    history[factors] = (
        history[factors]
        .ffill()
        .bfill()
    )

    history = history.reset_index()

    result_rows = []

    working = history.copy()

    for future_date in future_dates:

        row = {
            "date": future_date
        }

        for factor in factors:

            # -------------------------------------------------
            # Naive model
            # -------------------------------------------------
            if models[factor] is None:

                prediction = float(
                    working[factor]
                    .dropna()
                    .iloc[-1]
                )

            else:

                temp = pd.concat(
                    [
                        working[
                            ["date", factor]
                        ],
                        pd.DataFrame(
                            [
                                {
                                    "date": future_date,
                                    factor: np.nan,
                                }
                            ]
                        ),
                    ],
                    ignore_index=True,
                )

                temp = create_factor_features(
                    temp,
                    factor,
                )

                latest = temp.iloc[-1]

                feature_values = pd.DataFrame(
                    [
                        {
                            column: latest[column]
                            for column in feature_columns[factor]
                        }
                    ]
                )

                feature_values = (
                    feature_values
                    .replace(
                        [
                            np.inf,
                            -np.inf,
                        ],
                        np.nan,
                    )
                    .fillna(0)
                )

                predicted_delta = float(
                    models[factor].predict(
                        feature_values
                    )[0]
                )

                # -------------------------------------------------
                # Robust delta safety guard
                #
                # This limits the CHANGE, not the final level.
                # -------------------------------------------------
                bounds = delta_bounds.get(
                    factor
                )

                if bounds:

                    predicted_delta = float(
                        np.clip(
                            predicted_delta,
                            bounds["min"],
                            bounds["max"],
                        )
                    )

                previous_level = float(
                    working[factor]
                    .dropna()
                    .iloc[-1]
                )

                prediction = (
                    previous_level
                    + predicted_delta
                )

            row[factor] = float(
                prediction
            )

    # -----------------------------------------------------
    # IMPORTANT:
    # Add ONE complete future row only after ALL factors
    # have been predicted for this month.
    # -----------------------------------------------------
        working.loc[
            len(working)
        ] = row

        result_rows.append(
            row
        )


    return pd.DataFrame(
        result_rows
    )