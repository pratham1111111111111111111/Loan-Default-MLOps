import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. MLflow
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Default_Prediction")


# ============================================================
# 2. Load dataset
# ============================================================

DATA_PATH = "data/Loan_default.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)


# ============================================================
# 3. Feature Engineering
# ============================================================

def categorize_employment_stability(months):

    if months < 6:
        return "New Hire (Unstable)"

    elif months < 24:
        return "Moderately Stable"

    else:
        return "Highly Stable"


def categorize_credit_score(score):

    if score < 580:
        return "Poor"

    elif score < 670:
        return "Fair"

    elif score < 740:
        return "Good"

    elif score < 800:
        return "Very Good"

    else:
        return "Excellent"


df["EmploymentStability"] = df[
    "MonthsEmployed"
].apply(categorize_employment_stability)


df["CreditScoreCategory"] = df[
    "CreditScore"
].apply(categorize_credit_score)


df["LoanToIncomeRatio"] = (
    df["LoanAmount"] / df["Income"]
)


# ============================================================
# 4. Features and target
# ============================================================

X = df.drop(
    columns=["Default", "LoanID"]
)

y = df["Default"]


# ============================================================
# 5. Encoding
# ============================================================

X_encoded = pd.get_dummies(
    X,
    drop_first=True
)


# ============================================================
# 6. Train/Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("Training Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)


# ============================================================
# 7. Train Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# ============================================================
# 8. Prediction
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 9. Evaluation
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print("\n" + "=" * 60)
print("RETRAINED RANDOM FOREST")
print("=" * 60)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)


# ============================================================
# 10. Save model artifacts
# ============================================================

os.makedirs(
    "loan_models",
    exist_ok=True
)


joblib.dump(
    model,
    "loan_models/random_forest_model.joblib"
)


joblib.dump(
    X_encoded.columns.tolist(),
    "loan_models/feature_columns.joblib"
)


# ============================================================
# 11. MLflow tracking
# ============================================================

with mlflow.start_run(
    run_name="RandomForest_Auto_Retraining"
):

    mlflow.log_param(
        "model",
        "Random Forest"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "features",
        len(X_encoded.columns)
    )

    mlflow.log_metric(
        "accuracy",
        accuracy
    )

    mlflow.log_metric(
        "precision",
        precision
    )

    mlflow.log_metric(
        "recall",
        recall
    )

    mlflow.log_metric(
        "f1_score",
        f1
    )

    mlflow.sklearn.log_model(
        model,
        "random_forest_model"
    )


print("\n" + "=" * 60)
print("AUTO RETRAINING COMPLETED")
print("=" * 60)

print("New model saved:")
print(
    "loan_models/random_forest_model.joblib"
)