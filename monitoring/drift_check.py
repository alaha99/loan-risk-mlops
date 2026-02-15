import pandas as pd
from scipy.stats import ks_2samp

offline = pd.read_csv("feature_store/offline_features.csv")
live = pd.read_csv("data/live_applications.csv")

drift_detected = False

for col in ["loan_amount"]:
    stat, p = ks_2samp(offline["avg_amount_150d"], live["loan_amount"])
    if p < 0.05:
        drift_detected = True

if drift_detected:
    print("DRIFT DETECTED")
else:
    print("No drift")
