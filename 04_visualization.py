import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

# Create labels
normal_torque = df[df["Machine failure"] == 0]["Torque [Nm]"]
failed_torque = df[df["Machine failure"] == 1]["Torque [Nm]"]

normal_tool = df[df["Machine failure"] == 0]["Tool wear [min]"]
failed_tool = df[df["Machine failure"] == 1]["Tool wear [min]"]

normal_speed = df[df["Machine failure"] == 0]["Rotational speed [rpm]"]
failed_speed = df[df["Machine failure"] == 1]["Rotational speed [rpm]"]


# =========================
# 1. Torque Distribution
# =========================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [failed_torque, normal_torque],
    tick_labels=["Failed", "Normal"]
)

plt.title("Torque Distribution: Normal vs Failed Machines")
plt.xlabel("Machine Status")
plt.ylabel("Torque [Nm]")
plt.tight_layout()

plt.savefig("torque_failure.png", dpi=300)
plt.show()
plt.close()


# =========================
# 2. Tool Wear Distribution
# =========================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [failed_tool, normal_tool],
    tick_labels=["Failed", "Normal"]
)

plt.title("Tool Wear Distribution: Normal vs Failed Machines")
plt.xlabel("Machine Status")
plt.ylabel("Tool Wear [min]")
plt.tight_layout()

plt.savefig("toolwear_failure.png", dpi=300)
plt.show()
plt.close()


# =========================
# 3. Rotational Speed
# =========================

plt.figure(figsize=(8, 5))

plt.boxplot(
    [failed_speed, normal_speed],
    tick_labels=["Failed", "Normal"]
)

plt.title("Rotational Speed: Normal vs Failed Machines")
plt.xlabel("Machine Status")
plt.ylabel("Rotational Speed [rpm]")
plt.tight_layout()

plt.savefig("speed_failure.png", dpi=300)
plt.show()
plt.close()

print("\nAll visualizations generated successfully.")