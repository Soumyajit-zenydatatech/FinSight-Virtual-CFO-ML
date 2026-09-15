import pandas as pd

path = "model_store/external_factors.csv"

df = pd.read_csv(path)

print("\nCOLUMNS:")
print(df.columns.tolist())

print("\nLAST 20 ROWS:")
print(
    df[
        ["Order Date", "Oil", "GDP", "Unemployment"]
    ].tail(20).to_string(index=False)
)

print("\nRANGES:")
print(
    df[
        ["Oil", "GDP", "Unemployment"]
    ].agg(["min", "max"]).to_string()
)

print("\nDATA TYPES:")
print(
    df[
        ["Order Date", "Oil", "GDP", "Unemployment"]
    ].dtypes
)