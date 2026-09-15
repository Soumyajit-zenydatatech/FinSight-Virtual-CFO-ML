from pathlib import Path

import pandas as pd

from ml.external_forecasting.config import EXTERNAL_FACTORS


def load_external_history(model_store_dir: str) -> pd.DataFrame:
    """
    Load external-factor history and convert it to one monthly
    observation per factor.

    Missing months are created later by complete_monthly_index().
    """
    path = Path(model_store_dir) / "external_factors.csv"

    if not path.exists():
        raise FileNotFoundError(
            f"External factor history not found: {path}"
        )

    df = pd.read_csv(path)

    date_candidates = [
        "date",
        "Date",
        "Order Date",
        "order_date",
    ]

    date_column = None

    for column in date_candidates:
        if column in df.columns:
            date_column = column
            break

    if date_column is None:
        raise ValueError(
            "No date column found. Expected one of: "
            + ", ".join(date_candidates)
        )

    df["date"] = (
        pd.to_datetime(
            df[date_column],
            errors="coerce",
        )
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    df = df.dropna(subset=["date"]).copy()

    available_factors = [
        factor
        for factor in EXTERNAL_FACTORS
        if factor in df.columns
    ]

    if not available_factors:
        raise ValueError(
            "No external factor columns found."
        )

    for factor in available_factors:
        df[factor] = pd.to_numeric(
            df[factor],
            errors="coerce",
        )

    # If there are multiple records in one month,
    # aggregate them to one monthly value.
    df = (
        df[
            ["date"] + available_factors
        ]
        .groupby("date", as_index=False)
        .median(numeric_only=True)
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


def complete_monthly_index(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a complete monthly time index.

    Missing factor observations are forward-filled because
    external economic factors are point-in-time monthly values.
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
        .drop_duplicates("date")
        .set_index("date")
    )

    full_index = pd.date_range(
        start=result.index.min(),
        end=result.index.max(),
        freq="MS",
    )

    result = result.reindex(full_index)

    result.index.name = "date"

    factor_columns = [
        column
        for column in result.columns
    ]

    # Fill gaps using the most recent known observation.
    result[factor_columns] = (
        result[factor_columns]
        .ffill()
    )

    # If a factor starts with missing values,
    # backfill those initial values.
    result[factor_columns] = (
        result[factor_columns]
        .bfill()
    )

    return result.reset_index()