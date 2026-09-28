import os
import pandas as pd
from datetime import datetime


# ============================================================
# MODEL MONITORING
# ============================================================

PREDICTION_FILE = "monitoring/predictions.csv"


def log_prediction(
    prediction,
    probability
):
    """
    Store model prediction for monitoring.
    """

    os.makedirs("monitoring", exist_ok=True)

    new_record = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction": int(prediction),
        "default_probability": float(probability)
    }])

    if os.path.exists(PREDICTION_FILE):

        new_record.to_csv(
            PREDICTION_FILE,
            mode="a",
            header=False,
            index=False
        )

    else:

        new_record.to_csv(
            PREDICTION_FILE,
            index=False
        )


def monitor_predictions():

    if not os.path.exists(PREDICTION_FILE):

        print("No prediction data available yet.")
        return

    df = pd.read_csv(PREDICTION_FILE)

    total_predictions = len(df)

    default_predictions = (
        df["prediction"] == 1
    ).sum()

    non_default_predictions = (
        df["prediction"] == 0
    ).sum()

    default_rate = (
        default_predictions / total_predictions
    )

    average_probability = (
        df["default_probability"].mean()
    )

    print("=" * 60)
    print("LOAN DEFAULT MODEL MONITORING")
    print("=" * 60)

    print(f"Total Predictions     : {total_predictions}")
    print(f"Default Predictions   : {default_predictions}")
    print(f"No Default Predictions: {non_default_predictions}")

    print(
        f"Default Prediction Rate: "
        f"{default_rate:.4f}"
    )

    print(
        f"Average Default Probability: "
        f"{average_probability:.4f}"
    )

    print("=" * 60)

    # --------------------------------------------------------
    # BASIC MONITORING ALERT
    # --------------------------------------------------------

    if default_rate > 0.30:

        print("WARNING: High default prediction rate!")

    elif default_rate < 0.05:

        print("INFO: Very low default prediction rate.")

    else:

        print("STATUS: Prediction rate within monitoring range.")


if __name__ == "__main__":

    monitor_predictions()