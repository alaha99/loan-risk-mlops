import sqlite3
import mlflow.pyfunc
import pandas as pd

model = mlflow.pyfunc.load_model("models:/loan_risk_model/Production")

def predict(applicant_id):
    conn = sqlite3.connect("feature_store/online_store.db")
    df = pd.read_sql("SELECT * FROM features WHERE applicant_id=?", conn, params=(applicant_id,))
    conn.close()

    X = df.drop(["applicant_id", "high_risk"], axis=1)

    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    return {
        "prediction": "high_risk" if pred == 1 else "low_risk",
        "probability": float(prob)
    }
