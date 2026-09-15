from config import settings

from ml.external_forecasting.trainer import (
    train_external_model,
)


result = train_external_model(
    settings.MODEL_STORE_DIR
)

print("\nMODEL A TRAINING COMPLETE\n")

print(
    "Model path:",
    result["model_path"],
)

print(
    "Training rows:",
    result["training_rows"],
)

print(
    "Date range:",
    result["training_start"],
    "to",
    result["training_end"],
)

print("\nBest models:")

for factor, model in result[
    "best_models"
].items():
    print(
        f"  {factor}: {model}"
    )

print("\nMetrics:")

for factor, model_metrics in result["metrics"].items():

    print(f"\n{factor}")

    for model_name, metric_values in model_metrics.items():

        print(f"  {model_name}:")

        for metric, value in metric_values.items():

            print(
                f"    {metric}: {value:.4f}"
            )