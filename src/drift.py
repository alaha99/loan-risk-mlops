import pandas as pd
import numpy as np


def psi(expected, actual, bins=10):

    breakpoints = np.linspace(
        min(expected.min(), actual.min()),
        max(expected.max(), actual.max()),
        bins + 1
    )

    e_counts = np.histogram(expected, breakpoints)[0]
    a_counts = np.histogram(actual, breakpoints)[0]

    e_perc = e_counts / len(expected)
    a_perc = a_counts / len(actual)

    psi_value = np.sum(
        (e_perc - a_perc) *
        np.log((e_perc + 1e-6) / (a_perc + 1e-6))
    )

    return psi_value


offline = pd.read_csv("feature_store/offline_features.csv")
live = pd.read_csv("feature_store/live_features.csv")

features = [
    col for col in offline.columns
    if col not in ["applicant_id", "high_risk"]
]

drift_detected = False

for feature in features:

    score = psi(offline[feature], live[feature])

    print(f"{feature} PSI: {round(score,4)}")

    if score > 0.2:
        drift_detected = True

if drift_detected:
    print("DRIFT DETECTED")
else:
    print("NO DRIFT")
