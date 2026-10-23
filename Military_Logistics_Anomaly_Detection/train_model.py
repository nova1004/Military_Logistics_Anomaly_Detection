"""
train_model.py
--------------
Isolation Forest anomaly detection on SIMULATED shipment data.

Run:  python generate_dataset.py   (once, creates shipment_data.csv)
      python train_model.py

Outputs:
  shipment_anomaly_results.csv      all shipments + prediction + score
  logistics_risk_report.csv         only flagged shipments
  visualizations/*.png              three graphs
NOTE: Flagged records are statistical outliers in simulated data. They are NOT
confirmed sabotage, threats or security breaches.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # lets the script save images without opening a window
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42
CONTAMINATION = 0.05   # ESTIMATE of the share of anomalies, not a real-world rate

os.makedirs("visualizations", exist_ok=True)

# ---- Step 1: Load data -----------------------------------------------------
df = pd.read_csv("shipment_data.csv")
print("Loaded shipments:", df.shape[0], "rows,", df.shape[1], "columns")

# ---- Step 2: Check data quality -------------------------------------------
print("\nMissing values per column:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Duplicate shipment IDs:", df["shipment_id"].duplicated().sum())

df = df.drop_duplicates().reset_index(drop=True)

# ---- Step 3: Select features ----------------------------------------------
# The ID and the text category are NOT fed to the model.
feature_columns = [
    "shipment_quantity",
    "route_distance_km",
    "expected_delivery_days",
    "actual_delivery_days",
    "delivery_delay_days",
    "expected_inventory",
    "recorded_inventory",
    "inventory_discrepancy",
]

# Handle missing / invalid numbers: make non-numeric values NaN, then fill
# with the column median. Negative quantities or distances are invalid.
df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
for col in ["shipment_quantity", "route_distance_km"]:
    df.loc[df[col] <= 0, col] = np.nan
for col in feature_columns:
    df[col] = df[col].fillna(df[col].median())

X = df[feature_columns]

# ---- Step 4: Scale features -----------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- Step 5: Train Isolation Forest ---------------------------------------
model = IsolationForest(
    n_estimators=200,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED,
)
model.fit(X_scaled)

# ---- Step 6: Predict and score --------------------------------------------
# predict(): +1 = normal, -1 = anomaly
# decision_function(): lower = more abnormal, negative = flagged as anomaly
raw_prediction = model.predict(X_scaled)
df["prediction"] = np.where(raw_prediction == -1, "Anomaly", "Normal")
df["anomaly_score"] = model.decision_function(X_scaled).round(4)

# ---- Step 7: Save results --------------------------------------------------
df.to_csv("shipment_anomaly_results.csv", index=False)

flagged = df[df["prediction"] == "Anomaly"].sort_values("anomaly_score")
risk_columns = ["shipment_id", "category"] + feature_columns + ["prediction", "anomaly_score"]
flagged[risk_columns].to_csv("logistics_risk_report.csv", index=False)

# ---- Step 8: Visualizations ------------------------------------------------
normal = df[df["prediction"] == "Normal"]
anom = df[df["prediction"] == "Anomaly"]

# Graph 1: score distribution
plt.figure(figsize=(8, 5))
plt.hist(df["anomaly_score"], bins=40, color="steelblue", edgecolor="white")
plt.axvline(0, color="red", linestyle="--", label="Decision boundary (score = 0)")
plt.title("Distribution of Anomaly Scores (Isolation Forest)")
plt.xlabel("Anomaly score (lower = more unusual)")
plt.ylabel("Number of shipments")
plt.legend()
plt.tight_layout()
plt.savefig("visualizations/01_anomaly_score_distribution.png", dpi=150)
plt.close()

# Graph 2: delay vs distance
plt.figure(figsize=(8, 5))
plt.scatter(normal["route_distance_km"], normal["delivery_delay_days"],
            c="steelblue", s=20, alpha=0.6, label="Normal")
plt.scatter(anom["route_distance_km"], anom["delivery_delay_days"],
            c="red", s=40, marker="x", label="Flagged anomaly")
plt.title("Delivery Delay vs Route Distance")
plt.xlabel("Route distance (km)")
plt.ylabel("Delivery delay (days)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("visualizations/02_delay_vs_distance.png", dpi=150)
plt.close()

# Graph 3: inventory discrepancy vs shipment quantity
plt.figure(figsize=(8, 5))
plt.scatter(normal["shipment_quantity"], normal["inventory_discrepancy"],
            c="steelblue", s=20, alpha=0.6, label="Normal")
plt.scatter(anom["shipment_quantity"], anom["inventory_discrepancy"],
            c="red", s=40, marker="x", label="Flagged anomaly")
plt.title("Inventory Discrepancy vs Shipment Quantity")
plt.xlabel("Shipment quantity (units)")
plt.ylabel("Inventory discrepancy (recorded - expected)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("visualizations/03_inventory_discrepancy_vs_quantity.png", dpi=150)
plt.close()

# ---- Step 9: Summary -------------------------------------------------------
total = len(df)
n_flagged = len(anom)
print("\n===== SUMMARY =====")
print("Total shipments analyzed:", total)
print(f"Flagged anomalies: {n_flagged} ({n_flagged / total * 100:.1f}%)")
print("\nMost unusual flagged records:")
print(flagged[["shipment_id", "category", "shipment_quantity", "route_distance_km",
               "delivery_delay_days", "inventory_discrepancy", "anomaly_score"]].head(5).to_string(index=False))
print("\nFiles created:")
print(" - shipment_anomaly_results.csv")
print(" - logistics_risk_report.csv")
print(" - visualizations/01_anomaly_score_distribution.png")
print(" - visualizations/02_delay_vs_distance.png")
print(" - visualizations/03_inventory_discrepancy_vs_quantity.png")

# ---- Step 10 (optional): compare with SIMULATED tags ----------------------
# These tags come from the data generator, were never given to the model, and
# only describe records WE made unusual. This is a demonstration, not a
# real-world accuracy measure.
if os.path.exists("simulated_labels.csv"):
    labels = pd.read_csv("simulated_labels.csv")
    check = df.merge(labels, on="shipment_id")
    flagged_mask = check["prediction"] == "Anomaly"
    injected_mask = check["injected_unusual"] == 1
    both = int((flagged_mask & injected_mask).sum())
    print("\n===== OPTIONAL: COMPARISON WITH SIMULATED TAGS =====")
    print("Injected unusual records in data:", int(injected_mask.sum()))
    print("Flagged by model:", int(flagged_mask.sum()))
    print("Flagged AND injected:", both)
    print("Flagged but not injected:", int((flagged_mask & ~injected_mask).sum()))
    print("Injected but not flagged:", int((~flagged_mask & injected_mask).sum()))
    print("\nFlagged count by injected type:")
    print(check[injected_mask].groupby("injected_type")["prediction"]
          .apply(lambda s: f"{(s == 'Anomaly').sum()} of {len(s)}").to_string())
