import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os
import json

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.tracking import MlflowClient


mlflow.set_experiment("loan_risk")

df = pd.read_csv("feature_store/offline_features.csv")

X = df.drop(["applicant_id", "high_risk"], axis=1)
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

C_values = [0.5, 2.0]

best_accuracy = 0
best_run_id = None
best_metrics = None

for C_value in C_values:

    with mlflow.start_run() as run:

        model = LogisticRegression(C=C_value, max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_C_{C_value}.joblib"
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact("feature_store/offline_features.csv")

        mlflow.sklearn.log_model(model, "model")

        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = run.info.run_id
            best_metrics = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec
            }


# Save metrics for DVC
os.makedirs("metrics", exist_ok=True)
with open("metrics/train_metrics.json", "w") as f:
    json.dump(best_metrics, f, indent=4)


# Register Best Model
model_uri = f"runs:/{best_run_id}/model"

registered_model = mlflow.register_model(
    model_uri,
    "loan_risk_model"
)

client = MlflowClient()
client.transition_model_version_stage(
    name="loan_risk_model",
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True
)

print("Training complete.")
print(f"Best model accuracy: {best_accuracy}")
print("Best model moved to Production.")
