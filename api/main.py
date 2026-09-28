from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import mlflow
import mlflow.sklearn
import os
from datetime import datetime


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Default Prediction API",
    description="ML API for predicting loan default using MLflow",
    version="1.0.0"
)


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

MODEL_NAME = "Loan_Default_RandomForest"
MODEL_VERSION = "1"

MODEL_URI = f"models:/{MODEL_NAME}/{MODEL_VERSION}"


# ============================================================
# LOAD MODEL FROM MLFLOW MODEL REGISTRY
# ============================================================

print("=" * 60)
print("LOADING MODEL FROM MLFLOW")
print("=" * 60)

print("Model Name:", MODEL_NAME)
print("Model Version:", MODEL_VERSION)
print("Model URI:", MODEL_URI)


model = mlflow.sklearn.load_model(MODEL_URI)

print("MLflow model loaded successfully")


# ============================================================
# LOAD FEATURE INFORMATION
# ============================================================

FEATURE_PATH = "models/feature_columns.joblib"

import joblib

feature_columns = joblib.load(FEATURE_PATH)

print("Expected features:", len(feature_columns))


# ============================================================
# MONITORING CONFIGURATION
# ============================================================

MONITORING_DIR = "monitoring"
PREDICTION_FILE = os.path.join(
    MONITORING_DIR,
    "predictions.csv"
)

os.makedirs(MONITORING_DIR, exist_ok=True)


# ============================================================
# INPUT DATA MODEL
# ============================================================

class LoanApplication(BaseModel):

    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float

    Education: str
    EmploymentType: str
    MaritalStatus: str

    HasMortgage: str
    HasDependents: str
    LoanPurpose: str
    HasCoSigner: str


# ============================================================
# FEATURE ENGINEERING
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

    if income == 0:

        return 0

    return loan_amount / income


# ============================================================
# SAVE PREDICTION FOR MONITORING
# ============================================================

def log_prediction(
    prediction,
    probability
):

    record = pd.DataFrame([{

        "timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "prediction":
            int(prediction),

        "default_probability":
            float(probability)

    }])


    # --------------------------------------------------------
    # Create file if it does not exist
    # --------------------------------------------------------

    if os.path.exists(PREDICTION_FILE):

        record.to_csv(

            PREDICTION_FILE,

            mode="a",

            header=False,

            index=False

        )

    else:

        record.to_csv(

            PREDICTION_FILE,

            index=False

        )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Loan Default Prediction API is running",

        "model":
            MODEL_NAME,

        "model_version":
            MODEL_VERSION,

        "features":
            len(feature_columns)

    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model":
            MODEL_NAME,

        "version":
            MODEL_VERSION,

        "model_uri":
            MODEL_URI,

        "feature_count":
            len(feature_columns),

        "features":
            feature_columns

    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    # ========================================================
    # CONVERT INPUT TO DICTIONARY
    # ========================================================

    data = application.model_dump()


    # ========================================================
    # CONVERT TO DATAFRAME
    # ========================================================

    df = pd.DataFrame([data])


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    df["EmploymentStability"] = (

        df["MonthsEmployed"]

        .apply(
            categorize_employment_stability
        )

    )


    df["CreditScoreCategory"] = (

        df["CreditScore"]

        .apply(
            categorize_credit_score
        )

    )


    df["LoanToIncomeRatio"] = (

        df.apply(

            lambda row:
            calculate_loan_to_income(
                row["LoanAmount"],
                row["Income"]
            ),

            axis=1

        )

    )


    # ========================================================
    # ONE-HOT ENCODING
    # SAME METHOD USED DURING TRAINING
    # ========================================================

    df_encoded = pd.get_dummies(

        df,

        drop_first=True

    )


    # ========================================================
    # MATCH TRAINING FEATURES
    # ========================================================

    df_encoded = df_encoded.reindex(

        columns=feature_columns,

        fill_value=0

    )


    # ========================================================
    # ENSURE NUMERIC DATA
    # ========================================================

    df_encoded = df_encoded.astype(float)


    # ========================================================
    # MODEL PREDICTION
    # ========================================================

    prediction = model.predict(

        df_encoded

    )[0]


    # ========================================================
    # DEFAULT PROBABILITY
    # ========================================================

    probability = model.predict_proba(

        df_encoded

    )[0][1]


    # ========================================================
    # LOG PREDICTION FOR MONITORING
    # ========================================================

    log_prediction(

        prediction,

        probability

    )


    # ========================================================
    # RESULT
    # ========================================================

    if prediction == 1:

        result = "Loan Default"

    else:

        result = "No Loan Default"


    # ========================================================
    # API RESPONSE
    # ========================================================

    return {

        "prediction":
            int(prediction),

        "result":
            result,

        "default_probability":
            round(
                float(probability),
                4
            )

    }