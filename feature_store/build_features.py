import pandas as pd
from datetime import timedelta
import os

os.makedirs("feature_store", exist_ok=True)

apps = pd.read_csv("data/raw/applications.csv", parse_dates=["application_date"])
labels = pd.read_csv("data/raw/labels.csv")

rows = []

for applicant in apps["applicant_id"].unique():

    user = apps[apps["applicant_id"] == applicant].sort_values("application_date")
    last_date = user["application_date"].max()

    # 90D rejection ratio
    last_90 = user[user["application_date"] >= last_date - timedelta(days=90)]
    total_90 = len(last_90)
    rejected_90 = len(last_90[last_90["approved"] == 0])
    rejected_ratio = rejected_90 / total_90 if total_90 > 0 else 0

    # Loan velocity
    avg_30 = user[user["application_date"] >= last_date - timedelta(days=30)]["loan_amount"].mean()
    avg_150 = user[user["application_date"] >= last_date - timedelta(days=150)]["loan_amount"].mean()
    loan_velocity = (avg_30 - avg_150) if pd.notna(avg_30) else 0

    # Recency
    days_since_last = 0

    rows.append({
        "applicant_id": applicant,
        "rejected_ratio_90d": rejected_ratio,
        "loan_amount_velocity": loan_velocity,
        "days_since_last_application": days_since_last
    })

features = pd.DataFrame(rows)
features = features.merge(labels, on="applicant_id")

features.to_csv("feature_store/offline_features.csv", index=False)
print("Offline feature store created.")
