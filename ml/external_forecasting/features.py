import numpy as np
import pandas as pd


DEFAULT_LAGS = (1, 2, 3, 6, 12)


def create_factor_features(
    df: pd.DataFrame,
    target: str,
    lags=DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Create features using only information available before
    the current month.

    The target column itself remains the monthly level.

    The trainer will predict:

        current_level - previous_level

    rather than predicting the absolute level directly.
    """
    result = df.copy()

    result["date"] = (
        pd.to_datetime(result["date"])
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    result = (
        result
        .sort_values("date")
        .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # Previous levels
    # ---------------------------------------------------------
    for lag in lags:
        result[f"{target}_lag_{lag}"] = (
            result[target].shift(lag)
        )

    # ---------------------------------------------------------
    # Previous-month change
    # ---------------------------------------------------------
    result[f"{target}_change_1"] = (
        result[target].shift(1)
        - result[target].shift(2)
    )

    # ---------------------------------------------------------
    # Previous-month percentage change
    # ---------------------------------------------------------
    previous = result[target].shift(1)

    result[f"{target}_pct_change_1"] = (
        (
            result[target].shift(1)
            - result[target].shift(2)
        )
        / previous.replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Year-over-year change
    #
    # Uses only the value from 12 months ago.
    # ---------------------------------------------------------
    result[f"{target}_yoy_change"] = (
        result[target].shift(1)
        - result[target].shift(13)
    )

    # ---------------------------------------------------------
    # Year-over-year percentage change
    # ---------------------------------------------------------
    yoy_previous = result[target].shift(13)

    result[f"{target}_yoy_pct_change"] = (
        (
            result[target].shift(1)
            - result[target].shift(13)
        )
        / yoy_previous.replace(0, np.nan)
    )

    # ---------------------------------------------------------
    # Seasonal difference
    #
    # Difference between the previous month and the same
    # month one year earlier.
    # ---------------------------------------------------------
    result[f"{target}_seasonal_change"] = (
        result[target].shift(1)
        - result[target].shift(13)
    )

    # ---------------------------------------------------------
    # Rolling statistics calculated strictly from past values.
    # ---------------------------------------------------------
    past = result[target].shift(1)

    result[f"{target}_rolling_mean_3"] = (
        past.rolling(3).mean()
    )

    result[f"{target}_rolling_mean_6"] = (
        past.rolling(6).mean()
    )

    result[f"{target}_rolling_mean_12"] = (
        past.rolling(12).mean()
    )

    # ---------------------------------------------------------
    # Rolling volatility of previous values.
    # ---------------------------------------------------------
    result[f"{target}_rolling_std_3"] = (
        past.rolling(3).std()
    )

    result[f"{target}_rolling_std_6"] = (
        past.rolling(6).std()
    )

    result[f"{target}_rolling_std_12"] = (
        past.rolling(12).std()
    )

    # ---------------------------------------------------------
    # Short-term trend
    #
    # Difference between recent averages.
    # ---------------------------------------------------------
    result[f"{target}_trend_3_6"] = (
        result[f"{target}_rolling_mean_3"]
        - result[f"{target}_rolling_mean_6"]
    )

    result[f"{target}_trend_6_12"] = (
        result[f"{target}_rolling_mean_6"]
        - result[f"{target}_rolling_mean_12"]
    )

    # ---------------------------------------------------------
    # Calendar features
    # ---------------------------------------------------------
    result["month"] = (
        result["date"].dt.month
    )

    result["quarter"] = (
        result["date"].dt.quarter
    )

    # ---------------------------------------------------------
    # Cyclical month features
    #
    # These help models understand that December and January
    # are adjacent rather than far apart.
    # ---------------------------------------------------------
    result["month_sin"] = (
        np.sin(
            2 * np.pi * result["month"] / 12
        )
    )

    result["month_cos"] = (
        np.cos(
            2 * np.pi * result["month"] / 12
        )
    )

    return result