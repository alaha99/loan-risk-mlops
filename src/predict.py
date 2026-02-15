import sqlite3
import mlflow.pyfunc
import pandas as pd
import argparse


def predict(applicant_id):

    conn = sqlite3.connect("feature_store/online_store.db")

    df = pd.read_sql(
        "SELECT * FROM features WHERE applicant_id=?",
        conn,
        params=(applicant_id,)
    )
    conn.close()

    if df.empty:
        print("Applicant not found.")
        return

    model = mlflow.pyfunc.load_model(
        "models:/loan_risk_model/Production"
    )

    X = df.drop(["applicant_id", "high_risk"], axis=1)

    prediction = model.predict(X)[0]

    # Load underlying sklearn model for probability
    sklearn_model = mlflow.sklearn.load_model(
        "models:/loan_risk_model/Production"
    )

    prob = sklearn_model.predict_proba(X)[0][1]

    label = "HIGH RISK" if prediction == 1 else "LOW RISK"

    print(f"Applicant ID: {applicant_id}")
    print(f"Prediction: {label}")
    print(f"Probability: {round(prob, 4)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant_id", type=int, required=True)
    args = parser.parse_args()

    predict(args.applicant_id)
