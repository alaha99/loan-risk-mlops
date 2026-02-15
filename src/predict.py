import sqlite3
import mlflow.pyfunc
import pandas as pd

def predict(applicant_id):

    conn = sqlite3.connect("feature_store/online_store.db")
    df = pd.read_sql("SELECT * FROM features WHERE applicant_id=?", conn, params=(applicant_id,))
    conn.close()

    model = mlflow.pyfunc.load_model("models:/LoanRiskModel/Production")

    X = df.drop(["applicant_id", "high_risk"], axis=1)
    prediction = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    label = "high risk" if prediction == 1 else "low risk"

    return {"prediction": label, "probability": float(prob)}
