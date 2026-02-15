import pandas as pd
import sqlite3
from datetime import timedelta

applications = pd.read_csv("data/applications.csv", parse_dates=["application_date"])
labels = pd.read_csv("data/labels.csv")

applications = applications.sort_values("application_date")

def compute_features(df):
    feature_rows = []

    for applicant in df["applicant_id"].unique():
        user_df = df[df["applicant_id"] == applicant]

        last_date = user_df["application_date"].max()

        total_120 = user_df[
            user_df["application_date"] >= last_date - timedelta(days=120)
        ].shape[0]

        avg_150 = user_df[
            user_df["application_date"] >= last_date - timedelta(days=150)
        ]["loan_amount"].mean()

        rejected_90 = user_df[
            (user_df["application_date"] >= last_date - timedelta(days=90))
            & (user_df["approved"] == 0)
        ].shape[0]

        feature_rows.append({
            "applicant_id": applicant,
            "total_apps_120d": total_120,
            "avg_amount_150d": avg_150,
            "rejected_90d": rejected_90
        })

    return pd.DataFrame(feature_rows)

features = compute_features(applications)
features = features.merge(labels, on="applicant_id")

# Offline store
features.to_csv("feature_store/offline_features.csv", index=False)

# Online store (SQLite)
conn = sqlite3.connect("feature_store/online_store.db")
features.to_sql("features", conn, if_exists="replace", index=False)
conn.close()
