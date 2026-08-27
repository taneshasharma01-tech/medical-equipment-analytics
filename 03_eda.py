import pandas as pd

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# Separate normal and failed machines
normal = df[df["Machine failure"] == 0]
failed = df[df["Machine failure"] == 1]

print("===== FAILURE COMPARISON =====")

print("\nNumber of normal machines:", len(normal))
print("Number of failed machines:", len(failed))

# Variables we will use for prediction
features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

print("\n===== NORMAL vs FAILED: MEAN VALUES =====")

comparison = df.groupby("Machine failure")[features].mean().T
comparison.columns = ["Normal", "Failed"]

print(comparison)

print("\n===== NORMAL vs FAILED: MEDIAN VALUES =====")

median_comparison = df.groupby("Machine failure")[features].median().T
median_comparison.columns = ["Normal", "Failed"]

print(median_comparison)

print("\n===== MACHINE TYPE DISTRIBUTION =====")

print(pd.crosstab(
    df["Type"],
    df["Machine failure"],
    normalize="index"
) * 100)