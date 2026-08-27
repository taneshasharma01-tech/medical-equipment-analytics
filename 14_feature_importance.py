import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

print("=" * 60)
print("FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")

print("\nDataset shape:", df.shape)

# ============================================================
# 2. PREPARE DATA
# ============================================================

# Remove identifier, target-related, and non-feature columns
columns_to_drop = [
    "UDI",
    "Product ID",
    "Target",
    "Machine failure",
    "Failure Type"
]

X = df.drop(
    columns=[c for c in columns_to_drop if c in df.columns]
)

# Target variable
y = df["Machine failure"]

# Convert machine Type into numerical dummy variables
if "Type" in X.columns:
    X = pd.get_dummies(
        X,
        columns=["Type"],
        drop_first=False
    )

print("\nFeatures used for prediction:")
for feature in X.columns:
    print("-", feature)

# ============================================================
# 3. TRAIN TEST SPLIT
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
# 4. TRAIN RANDOM FOREST
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
# 5. ROC-AUC
# ============================================================

y_probability = model.predict_proba(X_test)[:, 1]

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

print("\nROC-AUC:", round(roc_auc, 4))

# ============================================================
# 6. FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

# Sort from highest to lowest importance
importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)

for _, row in importance.iterrows():
    print(
        f"{row['Feature']:<30} "
        f"{row['Importance']:.4f}"
    )

# ============================================================
# 7. SAVE RESULTS
# ============================================================

importance.to_csv(
    "feature_importance_results.csv",
    index=False
)

# ============================================================
# 8. FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

# Highest importance at top
plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 9. FINAL MESSAGE
# ============================================================

print("\nFeature importance analysis completed.")

print("Saved:")
print("- feature_importance_results.csv")
print("- feature_importance.png")

print("\nTop 5 most important features:")

for i, (_, row) in enumerate(
    importance.head(5).iterrows(),
    start=1
):
    print(
        f"{i}. {row['Feature']} "
        f"({row['Importance']:.4f})"
    )