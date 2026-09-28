import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# --------------------------------------------------
# 1. MLflow experiment
# --------------------------------------------------

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Loan_Default_Prediction")

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv("data/Loan_default.csv")

print("Dataset Shape:", df.shape)

# --------------------------------------------------
# 3. SAME FEATURE ENGINEERING AS NOTEBOOK
# --------------------------------------------------

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
    return loan_amount / income


df["EmploymentStability"] = df["MonthsEmployed"].apply(
    categorize_employment_stability
)

df["CreditScoreCategory"] = df["CreditScore"].apply(
    categorize_credit_score
)

df["LoanToIncomeRatio"] = calculate_loan_to_income(
    df["LoanAmount"],
    df["Income"]
)

# --------------------------------------------------
# 4. Separate X and y
# --------------------------------------------------

X = df.drop(columns=["Default", "LoanID"])
y = df["Default"]

# --------------------------------------------------
# 5. SAME encoding as training
# --------------------------------------------------

X_encoded = pd.get_dummies(X, drop_first=True)

# --------------------------------------------------
# 6. Load saved feature columns
# --------------------------------------------------

feature_columns = joblib.load(
    "models/feature_columns.joblib"
)

print("Expected Features:", len(feature_columns))
print("Current Features:", len(X_encoded.columns))

# Make sure exactly the same columns exist
X_encoded = X_encoded.reindex(
    columns=feature_columns,
    fill_value=0
)

print("Final Feature Shape:", X_encoded.shape)

# --------------------------------------------------
# 7. Load models
# --------------------------------------------------

models = {
    "Logistic Regression":
        joblib.load("models/logistic_regression_model.joblib"),

    "Random Forest":
        joblib.load("models/random_forest_model.joblib"),

    "Decision Tree":
        joblib.load("models/decision_tree_model.joblib")
}

# --------------------------------------------------
# 8. Load scaler
# --------------------------------------------------

scaler = joblib.load("models/scaler.joblib")

# --------------------------------------------------
# 9. MLflow tracking
# --------------------------------------------------

for model_name, model in models.items():

    with mlflow.start_run(run_name=model_name):

        # Logistic Regression requires scaling
        if model_name == "Logistic Regression":
            X_input = scaler.transform(X_encoded)
        else:
            X_input = X_encoded

        predictions = model.predict(X_input)

        accuracy = accuracy_score(y, predictions)
        precision = precision_score(y, predictions, zero_division=0)
        recall = recall_score(y, predictions, zero_division=0)
        f1 = f1_score(y, predictions, zero_division=0)

        # Parameters
        mlflow.log_param("model", model_name)
        mlflow.log_param("features", len(feature_columns))
        mlflow.log_param("dataset_rows", len(df))

        # Metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Model
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        print("\n" + "=" * 60)
        print(model_name)
        print("=" * 60)
        print("Accuracy :", accuracy)
        print("Precision:", precision)
        print("Recall   :", recall)
        print("F1 Score :", f1)