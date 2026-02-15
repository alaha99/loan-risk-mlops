import pandas as pd
from datetime import timedelta
from scipy.stats import ks_2samp

# -----------------------------
# Feature Computation Function
# -----------------------------
def compute_features(df):
    df = df.sort_values("application_date")
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


# -----------------------------
# Load Historical Data
# -----------------------------
historical = pd.read_csv(
    "data/applications.csv",
    parse_dates=["application_date"]
)

offline_features = compute_features(historical)


# -----------------------------
# Load Live Data
# -----------------------------
live = pd.read_csv(
    "data/live_applications.csv",
    parse_dates=["application_date"]
)

live_features = compute_features(live)


# -----------------------------
# Drift Detection
# -----------------------------
drift_detected = False

feature_columns = [
    "total_apps_120d",
    "avg_amount_150d",
    "rejected_90d"
]

for col in feature_columns:
    stat, p_value = ks_2samp(
        offline_features[col],
        live_features[col]
    )

    print(f"{col} p-value: {p_value}")

    if p_value < 0.05:
        drift_detected = True

# -----------------------------
# Final Output
# -----------------------------
if drift_detected:
    print("DRIFT DETECTED")
else:
    print("No Drift")
