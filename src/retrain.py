import os
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. PATHS
# ============================================================

DATA_PATH = "data/Loan_default.csv"
MODEL_DIR = "loan_models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# 2. MLflow CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("Loan_Default_Prediction")

# ============================================================
# 3. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

# ============================================================
# 4. FEATURE ENGINEERING
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


def calculate_loan_to_income(loan_amount, income):
    # Prevent division by zero
    income = income.replace(0, np.nan)

    ratio = loan_amount / income

    return ratio.fillna(0)


# Employment stability
df["EmploymentStability"] = df["MonthsEmployed"].apply(
    categorize_employment_stability
)

# Credit score category
df["CreditScoreCategory"] = df["CreditScore"].apply(
    categorize_credit_score
)

# Loan-to-income ratio
df["LoanToIncomeRatio"] = calculate_loan_to_income(
    df["LoanAmount"],
    df["Income"]
)

# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

if "Default" not in df.columns:
    raise ValueError("Target column 'Default' not found.")

X = df.drop(
    columns=["Default", "LoanID"],
    errors="ignore"
)

y = df["Default"]

# ============================================================
# 6. ENCODING
# ============================================================

X_encoded = pd.get_dummies(
    X,
    drop_first=True
)

# Convert boolean columns to integers
X_encoded = X_encoded.astype(float)

print("Encoded Feature Count:", len(X_encoded.columns))

# ============================================================
# 7. SAVE FEATURE COLUMNS
# ============================================================

feature_columns = list(X_encoded.columns)

joblib.dump(
    feature_columns,
    os.path.join(
        MODEL_DIR,
        "feature_columns.joblib"
    )
)

# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Training Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ============================================================
# 9. SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.joblib"
    )
)

# ============================================================
# 10. RANDOM FOREST MODEL
# ============================================================

print()
print("=" * 60)
print("RETRAINED RANDOM FOREST")
print("=" * 60)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

# Random Forest does not require scaling
model.fit(
    X_train,
    y_train
)

# ============================================================
# 11. PREDICTION
# ============================================================

predictions = model.predict(X_test)

# ============================================================
# 12. EVALUATION
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions,
    zero_division=0
)

recall = recall_score(
    y_test,
    predictions,
    zero_division=0
)

f1 = f1_score(
    y_test,
    predictions,
    zero_division=0
)

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# ============================================================
# 13. SAVE RETRAINED MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "random_forest_model.joblib"
)

joblib.dump(
    model,
    model_path
)

print()
print("New model saved:")
print(model_path)

# ============================================================
# 14. MLflow TRACKING
# ============================================================

with mlflow.start_run(
    run_name="Retrained_Random_Forest"
):

    # Parameters
    mlflow.log_param(
        "model",
        "Random Forest"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "random_state",
        42
    )

    mlflow.log_param(
        "class_weight",
        "balanced"
    )

    mlflow.log_param(
        "features",
        len(feature_columns)
    )

    mlflow.log_param(
        "dataset_rows",
        len(df)
    )

    mlflow.log_param(
        "training_rows",
        len(X_train)
    )

    mlflow.log_param(
        "testing_rows",
        len(X_test)
    )

    # Metrics
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

    # --------------------------------------------------------
    # IMPORTANT:
    # MLflow/skops requires explicit trust for the internal
    # sklearn Tree type used by Random Forest.
    # --------------------------------------------------------

    mlflow.sklearn.log_model(
        model,
        name="random_forest_model",
        skops_trusted_types=[
            "sklearn.tree._tree.Tree"
        ]
    )

print()
print("=" * 60)
print("AUTO RETRAINING COMPLETED")
print("=" * 60)
print("Model :", model_path)
print("MLflow experiment: Loan_Default_Prediction")
print("=" * 60)