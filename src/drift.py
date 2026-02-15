import pandas as pd
import numpy as np

def psi(expected, actual, bins=10):
    breakpoints = np.linspace(min(expected.min(), actual.min()),
                              max(expected.max(), actual.max()),
                              bins + 1)

    e_counts = np.histogram(expected, breakpoints)[0]
    a_counts = np.histogram(actual, breakpoints)[0]

    e_perc = e_counts / len(expected)
    a_perc = a_counts / len(actual)

    return np.sum((e_perc - a_perc) *
                  np.log((e_perc + 1e-6) / (a_perc + 1e-6)))

offline = pd.read_csv("feature_store/offline_features.csv")
live = pd.read_csv("feature_store/live_features.csv")

score = psi(offline["loan_amount_velocity"], live["loan_amount_velocity"])

if score > 0.2:
    print("DRIFT DETECTED")
else:
    print("No drift")
