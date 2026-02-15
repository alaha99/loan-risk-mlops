import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("loan_risk")
runs = client.search_runs(experiment.experiment_id)

best_run = max(runs, key=lambda r: r.data.metrics["accuracy"])

model_uri = f"runs:/{best_run.info.run_id}/model"

mlflow.register_model(model_uri, "LoanRiskModel")

print("Best model registered.")
