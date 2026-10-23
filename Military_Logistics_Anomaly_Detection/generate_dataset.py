"""
generate_dataset.py
-------------------
Creates a SIMULATED shipment dataset for the project
"Military Logistics Anomaly Detection Using Machine Learning".

IMPORTANT: Every record produced here is synthetic (random numbers made by this
script). None of it is genuine military logistics data.

How the data is made
1. Most shipments (about 95%) are "normal": their delay, route distance,
   quantity and inventory mismatch stay inside typical ranges.
2. A small share (about 5%) are "injected unusual" shipments with one kind of
   odd behaviour (very late delivery, large inventory mismatch, unusual
   quantity or abnormal route distance).
3. The hidden "injected" tag is saved in a SEPARATE file so it can never leak
   into the model's training features.

Run:  python generate_dataset.py
Outputs: shipment_data.csv, simulated_labels.csv
"""

import numpy as np
import pandas as pd

# ---- Settings --------------------------------------------------------------
RANDOM_SEED = 42        # fixed seed -> same data every run
N_SHIPMENTS = 800       # total records (task asks for roughly 500-1000)
ANOMALY_RATE = 0.05     # share of injected unusual records

rng = np.random.default_rng(RANDOM_SEED)

# ---- Shipment categories and their typical quantity ranges (units) ---------
categories = ["Rations", "Fuel", "Medical Supplies", "Spare Parts", "Ammunition"]
quantity_range = {
    "Rations": (200, 800),
    "Fuel": (300, 1000),
    "Medical Supplies": (50, 300),
    "Spare Parts": (40, 250),
    "Ammunition": (100, 500),
}

# ---- 1. Normal records -----------------------------------------------------
category = rng.choice(categories, size=N_SHIPMENTS)

shipment_quantity = np.array(
    [rng.integers(quantity_range[c][0], quantity_range[c][1] + 1) for c in category]
).astype(float)

# Route distance in km: mostly 100-900 km
route_distance_km = rng.uniform(100, 900, size=N_SHIPMENTS)

# Expected delivery time (days) = travel time at ~350 km/day + 1 to 2 days handling
expected_delivery_days = route_distance_km / 350 + rng.uniform(1, 2, size=N_SHIPMENTS)

# Actual delivery time = expected time + small random variation
# (average is slightly late, which is realistic)
delivery_noise = rng.normal(loc=0.2, scale=0.5, size=N_SHIPMENTS)
actual_delivery_days = np.maximum(expected_delivery_days + delivery_noise, 0.5)

# Inventory: what the system expects vs what was recorded at the destination
expected_inventory = shipment_quantity + rng.integers(20, 200, size=N_SHIPMENTS)
inventory_noise = rng.normal(loc=0, scale=2.0, size=N_SHIPMENTS)  # tiny counting error
recorded_inventory = expected_inventory + inventory_noise

# ---- 2. Inject unusual records --------------------------------------------
n_anomalies = int(N_SHIPMENTS * ANOMALY_RATE)
anomaly_idx = rng.choice(N_SHIPMENTS, size=n_anomalies, replace=False)
anomaly_type = np.array(["normal"] * N_SHIPMENTS, dtype=object)

# Split the injected records into four simple groups
groups = np.array_split(anomaly_idx, 4)

# Group 0: very late deliveries
for i in groups[0]:
    actual_delivery_days[i] = expected_delivery_days[i] + rng.uniform(4, 9)
    anomaly_type[i] = "extreme_delay"

# Group 1: large inventory mismatch (recorded far from expected)
for i in groups[1]:
    sign = rng.choice([-1, 1])
    recorded_inventory[i] = expected_inventory[i] + sign * rng.uniform(60, 150)
    anomaly_type[i] = "inventory_mismatch"

# Group 2: unusual shipment quantity (much larger than category norm)
for i in groups[2]:
    high = quantity_range[category[i]][1]
    shipment_quantity[i] = high * rng.uniform(2.0, 3.0)
    expected_inventory[i] = shipment_quantity[i] + rng.integers(20, 200)
    recorded_inventory[i] = expected_inventory[i] + rng.normal(0, 2.0)
    anomaly_type[i] = "unusual_quantity"

# Group 3: abnormal route distance (far longer than normal), with matching times
for i in groups[3]:
    route_distance_km[i] = rng.uniform(1800, 2800)
    expected_delivery_days[i] = route_distance_km[i] / 350 + rng.uniform(1, 2)
    actual_delivery_days[i] = expected_delivery_days[i] + rng.normal(0.2, 0.5)
    anomaly_type[i] = "abnormal_distance"

# ---- 3. Build the DataFrame ------------------------------------------------
df = pd.DataFrame(
    {
        "shipment_id": [f"SHP{str(n).zfill(4)}" for n in range(1, N_SHIPMENTS + 1)],
        "category": category,
        "shipment_quantity": shipment_quantity.round(0),
        "route_distance_km": route_distance_km.round(1),
        "expected_delivery_days": expected_delivery_days.round(2),
        "actual_delivery_days": actual_delivery_days.round(2),
        "expected_inventory": expected_inventory.round(0),
        "recorded_inventory": recorded_inventory.round(0),
    }
)

# Derived columns
df["delivery_delay_days"] = (df["actual_delivery_days"] - df["expected_delivery_days"]).round(2)
df["inventory_discrepancy"] = (df["recorded_inventory"] - df["expected_inventory"]).round(0)

# Put columns in a readable order
df = df[
    [
        "shipment_id", "category", "shipment_quantity", "route_distance_km",
        "expected_delivery_days", "actual_delivery_days", "delivery_delay_days",
        "expected_inventory", "recorded_inventory", "inventory_discrepancy",
    ]
]

# ---- 4. Save ---------------------------------------------------------------
df.to_csv("shipment_data.csv", index=False)

# Hidden simulation tags kept in a separate file (never used for training)
labels = pd.DataFrame(
    {
        "shipment_id": df["shipment_id"],
        "injected_unusual": (anomaly_type != "normal").astype(int),
        "injected_type": anomaly_type,
    }
)
labels.to_csv("simulated_labels.csv", index=False)

print("Simulated dataset created (NOT real military data).")
print("Records:", len(df))
print("Injected unusual records:", int(labels["injected_unusual"].sum()))
print("Saved: shipment_data.csv, simulated_labels.csv")
print(df.head())
