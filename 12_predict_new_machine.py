import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")


# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

X = df.drop(columns=[
    "UDI",
    "Product ID",
    "Machine failure",
    "TWF",
    "HDF",
    "PWF",
    "OSF",
    "RNF"
])

y = df["Machine failure"]


# ============================================================
# 3. FEATURE TYPES
# ============================================================

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]


# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            "passthrough",
            numerical_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


X_processed = preprocessor.fit_transform(X)


# ============================================================
# 5. TRAIN FINAL RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_processed, y)


# ============================================================
# 6. ENTER NEW MACHINE DATA
# ============================================================

print("\n============================================")
print("PREDICTIVE MAINTENANCE SYSTEM")
print("============================================")

print("\nEnter machine sensor values:\n")

machine_type = input(
    "Machine Type (L / M / H): "
).upper()

air_temperature = float(
    input("Air temperature [K]: ")
)

process_temperature = float(
    input("Process temperature [K]: ")
)

rotational_speed = float(
    input("Rotational speed [rpm]: ")
)

torque = float(
    input("Torque [Nm]: ")
)

tool_wear = float(
    input("Tool wear [min]: ")
)


# ============================================================
# 7. CREATE NEW MACHINE DATAFRAME
# ============================================================

new_machine = pd.DataFrame({
    "Type": [machine_type],
    "Air temperature [K]": [air_temperature],
    "Process temperature [K]": [process_temperature],
    "Rotational speed [rpm]": [rotational_speed],
    "Torque [Nm]": [torque],
    "Tool wear [min]": [tool_wear]
})


# ============================================================
# 8. PREPROCESS NEW DATA
# ============================================================

new_machine_processed = preprocessor.transform(
    new_machine
)


# ============================================================
# 9. PREDICT FAILURE PROBABILITY
# ============================================================

failure_probability = model.predict_proba(
    new_machine_processed
)[0][1]


# ============================================================
# 10. APPLY OPTIMIZED THRESHOLD
# ============================================================

threshold = 0.30

if failure_probability >= threshold:
    prediction = "FAILURE"
else:
    prediction = "NORMAL"


# ============================================================
# 11. RISK LEVEL
# ============================================================

if failure_probability < 0.15:
    risk = "LOW"
elif failure_probability < 0.30:
    risk = "MEDIUM"
else:
    risk = "HIGH"


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n============================================")
print("PREDICTION RESULT")
print("============================================")

print(
    f"Failure probability: "
    f"{failure_probability * 100:.2f}%"
)

print(
    f"Classification threshold: "
    f"{threshold:.2f}"
)

print(
    f"Prediction: {prediction}"
)

print(
    f"Risk level: {risk}"
)

print("============================================")


# ============================================================
# 13. RECOMMENDATION
# ============================================================

if prediction == "FAILURE":

    print(
        "\nRecommendation: "
        "Schedule inspection / preventive maintenance."
    )

else:

    print(
        "\nRecommendation: "
        "Continue normal operation and monitoring."
    )