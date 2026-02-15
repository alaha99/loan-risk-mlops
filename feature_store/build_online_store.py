import pandas as pd
import sqlite3

df = pd.read_csv("feature_store/offline_features.csv")

conn = sqlite3.connect("feature_store/online_store.db")
df.to_sql("features", conn, if_exists="replace", index=False)
conn.close()

print("Online store updated.")
