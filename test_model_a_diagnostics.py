import joblib
import pandas as pd

MODEL_PATH = "model_store/external_forecast/model_A.pkl"
DATA_PATH = "model_store/external_factors.csv"

package = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

df["Order Date"] = pd.to_datetime(df["Order Date"])
df["date"] = (
    df["Order Date"]
    .dt.to_period("M")
    .dt.to_timestamp()
)

df = (
    df[
        ["date", "Oil", "GDP", "Unemployment"]
    ]
    .sort_values("date")
    .drop_duplicates("date")
    .reset_index(drop=True)
)

print("\nSELECTED MODELS:")
for factor, model_name in package["best_models"].items():
    print(f"{factor}: {model_name}")

print("\nLATEST HISTORICAL VALUES:")
print(df.tail(12).to_string(index=False))

print("\nHISTORICAL RANGES:")
print(
    df[
        ["Oil", "GDP", "Unemployment"]
    ].agg(["min", "max"]).to_string()
)

print("\n2026 MODEL A FORECAST:")
print(
    pd.read_csv(
        "model_store/external_forecast/forecast_2026.csv"
    ).to_string(index=False)
)