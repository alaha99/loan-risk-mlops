import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

mlflow.set_experiment("loan_risk")

df = pd.read_csv("feature_store/offline_features.csv")

X = df.drop(["applicant_id", "high_risk"], axis=1)
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = [0.5, 2.0]

for C_value in models:

    with mlflow.start_run():

        model = LogisticRegression(C=C_value, max_iter=1000)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        # Log parameters
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C_value)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        # Save model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/model_C_{C_value}.joblib"
        joblib.dump(model, model_path)

        # Log artifacts
        mlflow.log_artifact(model_path)
        mlflow.log_artifact("feature_store/offline_features.csv")
