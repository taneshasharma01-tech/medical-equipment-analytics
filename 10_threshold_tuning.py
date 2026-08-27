import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")

print("Dataset shape:", df.shape)

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
# 3. IDENTIFY COLUMNS
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
        ("numerical", "passthrough", numerical_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

X_processed = preprocessor.fit_transform(X)

# ============================================================
# 5. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================
# 6. TRAIN RANDOM FOREST
# ============================================================

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed.")

# ============================================================
# 7. PREDICT PROBABILITIES
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]

print("\nROC-AUC:")
print(roc_auc_score(y_test, y_probability))

# ============================================================
# 8. TEST DIFFERENT THRESHOLDS
# ============================================================

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
              0.40, 0.45, 0.50, 0.55, 0.60]

print("\n===== THRESHOLD ANALYSIS =====")

for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.3f} | "
        f"Recall: {recall:.3f} | "
        f"TP: {tp} | FN: {fn}"
    )

# ============================================================
# 9. SELECT THRESHOLD
# ============================================================

selected_threshold = 0.30

y_pred_final = (
    y_probability >= selected_threshold
).astype(int)

# ============================================================
# 10. FINAL RESULTS
# ============================================================

print("\n============================================")
print("FINAL RESULTS AT THRESHOLD =", selected_threshold)
print("============================================")

print("\n===== CLASSIFICATION REPORT =====")

print(
    classification_report(
        y_test,
        y_pred_final,
        target_names=["Normal", "Failure"]
    )
)

print("\n===== CONFUSION MATRIX =====")

print(confusion_matrix(y_test, y_pred_final))

# ============================================================
# 11. SAVE THRESHOLD COMPARISON PLOT
# ============================================================

recalls = []
precisions = []

for threshold in thresholds:

    y_pred = (y_probability >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    recalls.append(recall)
    precisions.append(precision)

plt.figure(figsize=(8, 5))

plt.plot(
    thresholds,
    recalls,
    marker="o",
    label="Failure Recall"
)

plt.plot(
    thresholds,
    precisions,
    marker="o",
    label="Failure Precision"
)

plt.axvline(
    selected_threshold,
    linestyle="--",
    label=f"Selected Threshold = {selected_threshold}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")

plt.title("Precision-Recall Trade-off")

plt.legend()

plt.tight_layout()

plt.savefig(
    "threshold_optimization.png",
    dpi=300
)

plt.show()
plt.close()

print("\nThreshold optimization completed.")