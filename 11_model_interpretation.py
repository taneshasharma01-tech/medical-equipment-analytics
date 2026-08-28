import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("FINAL MODEL INTERPRETATION")
print("=" * 60)

df = pd.read_csv("data/ai4i2020.csv")

print("\nDataset shape:", df.shape)

# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

features = [
    "Type",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

target = "Machine failure"

X = df[features]
y = df[target]

# ============================================================
# 3. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 4. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["Type"]
        )
    ],
    remainder="passthrough"
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ============================================================
# 5. TRAIN RANDOM FOREST
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(
    X_train_processed,
    y_train
)

print("Training completed.")

# ============================================================
# 6. PREDICTION
# ============================================================

decision_threshold = 0.30

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]

y_pred = (
    y_probability >= decision_threshold
).astype(int)

# ============================================================
# 7. CONFUSION MATRIX
# ============================================================

tn, fp, fn, tp = confusion_matrix(
    y_test,
    y_pred
).ravel()

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(f"True Negatives  (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives  (TP): {tp}")

# ============================================================
# 8. PERFORMANCE METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

false_positive_rate = (
    fp / (fp + tn)
    if (fp + tn) > 0
    else 0
)

false_negative_rate = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)

negative_predictive_value = (
    tn / (tn + fn)
    if (tn + fn) > 0
    else 0
)

print("\n" + "=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

print(f"Accuracy:             {accuracy:.4f}")
print(f"Failure Precision:    {precision:.4f}")
print(f"Failure Recall:       {recall:.4f}")
print(f"Failure F1-score:     {f1:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"False Positive Rate:  {false_positive_rate:.4f}")
print(f"False Negative Rate:  {false_negative_rate:.4f}")
print(f"Negative Predictive Value: {negative_predictive_value:.4f}")

# ============================================================
# 9. PERCENTAGE INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("PERCENTAGE INTERPRETATION")
print("=" * 60)

print(
    f"Failure detection rate: {recall * 100:.2f}%"
)

print(
    f"Missed failure rate:     {false_negative_rate * 100:.2f}%"
)

print(
    f"Normal machine correctly identified: "
    f"{specificity * 100:.2f}%"
)

print(
    f"False alarm rate:        "
    f"{false_positive_rate * 100:.2f}%"
)

# ============================================================
# 10. OPERATIONAL INTERPRETATION
# ============================================================

print("\n" + "=" * 60)
print("OPERATIONAL INTERPRETATION")
print("=" * 60)

print(
    f"\nOut of {tp + fn} actual machine failures, "
    f"the model detected {tp}."
)

print(
    f"The model missed {fn} actual failures."
)

print(
    f"Out of {tn + fp} normal machines, "
    f"{tn} were correctly identified as normal."
)

print(
    f"The model generated {fp} false alarms."
)

print(
    f"\nA threshold of {decision_threshold:.2f} "
    "prioritizes failure detection over precision."
)

print(
    "This reduces missed failures compared with a "
    "higher decision threshold, but increases false alarms."
)

print("\n" + "=" * 60)
print("INTERPRETATION COMPLETE")
print("=" * 60)