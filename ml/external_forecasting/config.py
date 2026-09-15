from pathlib import Path

EXTERNAL_FACTORS = [
    "CCI",
    "CPI",
    "Oil",
    "GDP",
    "Unemployment",
    "ROI",
]

DATE_COLUMN = "date"

MODEL_A_FILENAME = "model_A.pkl"
MODEL_A_METADATA_FILENAME = "metadata.json"

DEFAULT_HORIZON = 12

MIN_TRAIN_ROWS = 24

MODEL_CANDIDATES = [
    "naive",
    "ridge",
    "random_forest",
    "hist_gradient_boosting",
]