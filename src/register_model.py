import mlflow
from mlflow.tracking import MlflowClient

# ==========================================
# MLflow configuration
# ==========================================

mlflow.set_tracking_uri("sqlite:///mlflow.db")

client = MlflowClient()

experiment_name = "Loan_Default_Prediction"

# ==========================================
# Get experiment
# ==========================================

experiment = client.get_experiment_by_name(
    experiment_name
)

if experiment is None:
    raise Exception("Experiment not found!")

print("Experiment ID:", experiment.experiment_id)

# ==========================================
# Get runs ordered by F1 score
# ==========================================

runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.f1_score DESC"]
)

if not runs:
    raise Exception("No MLflow runs found!")

best_run = runs[0]

print("\nBest Run:")
print("Run ID:", best_run.info.run_id)
print("Model:", best_run.data.params.get("model"))
print("F1 Score:", best_run.data.metrics.get("f1_score"))
print("Accuracy:", best_run.data.metrics.get("accuracy"))

# ==========================================
# Get LoggedModel ID
# ==========================================

run = mlflow.get_run(best_run.info.run_id)

if not run.outputs.model_outputs:
    raise Exception("No LoggedModel found for this run!")

logged_model_id = run.outputs.model_outputs[0].model_id

print("Logged Model ID:", logged_model_id)

# ==========================================
# Register best model
# ==========================================

model_name = "Loan_Default_RandomForest"

registered_model = mlflow.register_model(
    model_uri=f"models:/{logged_model_id}",
    name=model_name
)

print("\n==========================================")
print("MODEL REGISTERED SUCCESSFULLY")
print("==========================================")

print("Model Name:", registered_model.name)
print("Version:", registered_model.version)
print("Run ID:", best_run.info.run_id)
print("Logged Model ID:", logged_model_id)

print("\nMLflow Model Registry completed.")