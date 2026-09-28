from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from datetime import datetime


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Loan Default Prediction API",
    description="Dockerized ML API for Loan Default Prediction",
    version="2.0.0"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = "models/random_forest_model.joblib"
FEATURE_PATH = "models/feature_columns.joblib"

model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)

print("=" * 60)
print("LOAN DEFAULT MODEL LOADED")
print("=" * 60)
print("Model: Random Forest")
print("Model path:", MODEL_PATH)
print("Expected features:", len(feature_columns))


# ============================================================
# MONITORING
# ============================================================

MONITORING_DIR = "monitoring"

PREDICTION_FILE = os.path.join(
    MONITORING_DIR,
    "predictions.csv"
)

os.makedirs(
    MONITORING_DIR,
    exist_ok=True
)


def log_prediction(prediction, probability):

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


def calculate_loan_to_income(
    loan_amount,
    income
):

    if income == 0:
        return 0

    return loan_amount / income


# ============================================================
# HOME / HEALTH CHECK
# ============================================================

@app.get("/")
def home():

    return {

        "message":
            "Loan Default Prediction API is running",

        "model":
            "Random Forest",

        "deployment":
            "Docker",

        "feature_count":
            len(feature_columns)

    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():

    return {

        "model":
            "Random Forest",

        "model_file":
            MODEL_PATH,

        "feature_count":
            len(feature_columns),

        "features":
            feature_columns

    }


# ============================================================
# PREDICTION
# ============================================================

@app.post("/predict")
def predict(application: LoanApplication):

    # --------------------------------------------------------
    # Convert input to dictionary
    # --------------------------------------------------------

    data = application.model_dump()

    df = pd.DataFrame([data])


    # --------------------------------------------------------
    # Feature Engineering
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # One-Hot Encoding
    # --------------------------------------------------------

    df_encoded = pd.get_dummies(
        df,
        drop_first=True
    )


    # --------------------------------------------------------
    # Match Training Features
    # --------------------------------------------------------

    df_encoded = df_encoded.reindex(

        columns=feature_columns,

        fill_value=0

    )


    # --------------------------------------------------------
    # Convert to Numeric
    # --------------------------------------------------------

    df_encoded = df_encoded.astype(float)


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = model.predict(
        df_encoded
    )[0]


    probability = model.predict_proba(
        df_encoded
    )[0][1]


    # --------------------------------------------------------
    # Save Prediction
    # --------------------------------------------------------

    log_prediction(
        prediction,
        probability
    )


    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if prediction == 1:

        result = "Loan Default"

    else:

        result = "No Loan Default"


    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return {

        "prediction":
            int(prediction),

        "result":
            result,

        "default_probability":
            round(
                float(probability),
                4
            ),

        "model":
            "Random Forest",

        "deployment":
            "Docker"

    }