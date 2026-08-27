import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    classification_report
)

# ============================================================
# 1. LOAD DATA
# ============================================================

print("=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)

df = pd.read_csv("data/ai4i2020.csv")

print("\nDataset shape:", df.shape)

# ============================================================
# 2. FEATURES AND TARGET
# ============================================================

# Only use machine parameters available for prediction
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

print("\nFeatures used:")
for feature in features:
    print("-", feature)

print("\nTarget:", target)

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
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            ["Type"]
        )
    ],
    remainder="passthrough"
)

# ============================================================
# 5. RANDOM FOREST MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

# ============================================================
# 6. TRAIN MODEL
# ============================================================

print("\nTraining final Random Forest...")

pipeline.fit(
    X_train,
    y_train
)

print("Training completed.")

# ============================================================
# 7. PREDICTION
# ============================================================

# Failure probability
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Use tuned decision threshold
decision_threshold = 0.30

y_pred = (
    y_prob >= decision_threshold
).astype(int)

print("\nDecision threshold:", decision_threshold)

# ============================================================
# 8. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "Normal",
            "Failure"
        ]
    )
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Normal",
        "Failure"
    ]
)

disp.plot()

plt.title(
    "Random Forest Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 9. ROC CURVE
# ============================================================

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_prob
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

print("\n" + "=" * 60)
print("ROC-AUC")
print("=" * 60)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

plt.figure(figsize=(8, 6))

plt.plot(
    fpr,
    tpr,
    label=f"Random Forest (AUC = {roc_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate"
)

plt.title(
    "ROC Curve - Random Forest"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "roc_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 10. PRECISION-RECALL CURVE
# ============================================================

precision, recall, thresholds_pr = (
    precision_recall_curve(
        y_test,
        y_prob
    )
)

average_precision = (
    average_precision_score(
        y_test,
        y_prob
    )
)

print("\n" + "=" * 60)
print("AVERAGE PRECISION")
print("=" * 60)

print(
    f"Average Precision: {average_precision:.4f}"
)

plt.figure(figsize=(8, 6))

plt.plot(
    recall,
    precision,
    label=f"Random Forest (AP = {average_precision:.4f})"
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Precision-Recall Curve - Random Forest"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "precision_recall_curve.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

# ============================================================
# 11. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("FINAL EVALUATION SUMMARY")
print("=" * 60)

print(
    f"Model: Random Forest"
)

print(
    f"Number of trees: 200"
)

print(
    f"Decision threshold: {decision_threshold:.2f}"
)

print(
    f"ROC-AUC: {roc_auc:.4f}"
)

print(
    f"Average Precision: {average_precision:.4f}"
)

print("\nAll evaluation plots generated successfully.")

print("\nSaved:")
print("- confusion_matrix.png")
print("- roc_curve.png")
print("- precision_recall_curve.png")