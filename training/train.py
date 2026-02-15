import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib

mlflow.set_experiment("loan_risk")

df = pd.read_csv("feature_store/offline_features.csv")

X = df.drop(["applicant_id", "high_risk"], axis=1)
y = df["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

for C in [0.5, 2.0]:
    with mlflow.start_run():
        model = LogisticRegression(C=C)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)

        joblib.dump(model, "model.joblib")
        mlflow.log_artifact("model.joblib")
        mlflow.log_artifact("feature_store/offline_features.csv")

        mlflow.sklearn.log_model(model, "model")
