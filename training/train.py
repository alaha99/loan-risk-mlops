import mlflow
import mlflow.sklearn
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from mlflow.tracking import MlflowClient

# -----------------------------
# SET TRACKING URI (IMPORTANT)
# -----------------------------
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# -----------------------------
# D1: Set Experiment Name
# -----------------------------
mlflow.set_experiment("loan_risk")

# -----------------------------
# Load Features
# -----------------------------
df = pd.read_csv("feature_store/offline_features.csv")

X = df.drop(["applicant_id", "high_risk"], axis=1)
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

best_accuracy = 0
best_run_id = None

# -----------------------------
# D2: Train 2 Model Versions
# -----------------------------
for C in [0.5, 2.0]:

    with mlflow.start_run(run_name=f"LogReg_C_{C}") as run:

        model = LogisticRegression(C=C, max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log Parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)

        # Log Metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Log Artifacts
        model_path = f"model_C_{C}.joblib"
        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)
        mlflow.log_artifact("feature_store/offline_features.csv")

        mlflow.sklearn.log_model(model, "model")

        # Track Best Model
        if acc > best_accuracy:
            best_accuracy = acc
            best_run_id = run.info.run_id

# -----------------------------
# D4: Register Best Model
# -----------------------------
client = MlflowClient()
model_name = "loan_risk_model"

result = mlflow.register_model(
    f"runs:/{best_run_id}/model",
    model_name
)

client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Production"
)

print("Best model registered and moved to Production.")
print("Best Run ID:", best_run_id)
print("Best Accuracy:", best_accuracy)
