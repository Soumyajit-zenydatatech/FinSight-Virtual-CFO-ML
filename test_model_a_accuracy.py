from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. CONFIGURATION
# ============================================================

CSV_PATH = Path("model_store/external_factors.csv")

FACTORS = [
    "CCI",
    "CPI",
    "Oil",
    "GDP",
    "Unemployment",
    "ROI",
]

TEST_MONTHS = 12


# ============================================================
# 2. METRIC FUNCTIONS
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    errors = actual - predicted

    mae = np.mean(np.abs(errors))

    rmse = np.sqrt(np.mean(errors ** 2))

    actual_total = np.sum(np.abs(actual))

    if actual_total == 0:
        wape = np.nan
    else:
        wape = (
            np.sum(np.abs(errors))
            / actual_total
            * 100
        )

    smape_denominator = (
        np.abs(actual) + np.abs(predicted)
    ) / 2

    smape_values = np.where(
        smape_denominator == 0,
        0,
        np.abs(errors) / smape_denominator
    )

    smape = np.mean(smape_values) * 100

    if len(actual) < 2:
        directional_accuracy = np.nan
    else:
        actual_direction = np.sign(np.diff(actual))
        predicted_direction = np.sign(np.diff(predicted))

        directional_accuracy = (
            np.mean(
                actual_direction == predicted_direction
            )
            * 100
        )

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE": wape,
        "sMAPE": smape,
        "Directional Accuracy": directional_accuracy,
    }


def calculate_improvement(old_error, new_error):
    if old_error == 0:
        return np.nan

    return (
        (old_error - new_error)
        / old_error
        * 100
    )


# ============================================================
# 3. LOAD AND CLEAN DATA
# ============================================================

def load_data():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found: {CSV_PATH.resolve()}"
        )

    df = pd.read_csv(CSV_PATH)

    print("\nCSV columns found:")
    print(df.columns.tolist())

    DATE_COLUMN = "Order Date"

    if DATE_COLUMN not in df.columns:
        raise ValueError(
            f"The CSV must contain a '{DATE_COLUMN}' column."
        )

    missing_factors = [
        factor
        for factor in FACTORS
        if factor not in df.columns
    ]

    if missing_factors:
        raise ValueError(
            f"Missing factor columns: {missing_factors}"
        )

    df["date"] = pd.to_datetime(
        df[DATE_COLUMN],
        errors="coerce"
    )

    df = df.dropna(subset=["date"])

    for factor in FACTORS:
        df[factor] = pd.to_numeric(
            df[factor],
            errors="coerce"
        )

    df = df.sort_values("date")

    # Convert dates to monthly dates
    df["date"] = (
        df["date"]
        .dt.to_period("M")
        .dt.to_timestamp()
    )

    # If duplicate months exist, aggregate them
    df = (
        df.groupby("date")[FACTORS]
        .median()
        .reset_index()
    )

    # Sort again after grouping
    df = df.sort_values("date")

    print("\nCleaned data:")
    print(df.head())

    print("\nDate range:")
    print(df["date"].min(), "to", df["date"].max())

    print("\nNumber of monthly rows:")
    print(len(df))
    print("\nMissing months:")

    all_months = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="MS"
    )

    missing_months = all_months.difference(df["date"])

    print(missing_months)
    print("Number of missing months:", len(missing_months))

    print("\nMissing factor values:")
    print(df[FACTORS].isna().sum())

    return df


# ============================================================
# 4. NAIVE BASELINE EVALUATION
# ============================================================

def evaluate_naive_baseline(df):
    if len(df) <= TEST_MONTHS:
        raise ValueError(
            "Not enough rows for the selected test period."
        )

    train_df = df.iloc[:-TEST_MONTHS].copy()
    test_df = df.iloc[-TEST_MONTHS:].copy()

    results = []

    print("\n" + "=" * 70)
    print("NAIVE BASELINE ACCURACY")
    print("=" * 70)

    print(
        f"Training period: "
        f"{train_df['date'].min().date()} "
        f"to "
        f"{train_df['date'].max().date()}"
    )

    print(
        f"Testing period: "
        f"{test_df['date'].min().date()} "
        f"to "
        f"{test_df['date'].max().date()}"
    )

    for factor in FACTORS:
        # Naive forecast = last training value repeated
        last_training_value = train_df[factor].iloc[-1]

        actual = test_df[factor].to_numpy()

        predicted = np.repeat(
            last_training_value,
            len(test_df)
        )

        valid_mask = (
            ~np.isnan(actual)
            & ~np.isnan(predicted)
        )

        actual = actual[valid_mask]
        predicted = predicted[valid_mask]

        metrics = calculate_metrics(
            actual,
            predicted
        )

        results.append({
            "Factor": factor,
            "Model": "Naive",
            **metrics,
        })

        print(f"\n{factor}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")

    return pd.DataFrame(results)


# ============================================================
# 5. MAIN FUNCTION
# ============================================================

def main():
    print("MODEL A ACCURACY TEST")

    df = load_data()

    results_df = evaluate_naive_baseline(df)

    output_path = Path(
        "model_a_naive_accuracy_results.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        results_df.to_string(index=False)
    )

    print(
        f"\nResults saved to: "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()