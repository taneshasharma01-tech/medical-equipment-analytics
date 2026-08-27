import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# =========================
# 1. Load dataset
# =========================

df = pd.read_csv("data/ai4i2020.csv")


# =========================
# 2. Define features/target
# =========================

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


# =========================
# 3. Train-test split
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =========================
# 4. Preprocessing
# =========================

categorical_features = ["Type"]

numerical_features = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            SimpleImputer(strategy="median"),
            numerical_features
        )
    ]
)


# =========================
# 5. Random Forest
# =========================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)


# =========================
# 6. Pipeline
# =========================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# =========================
# 7. Train model
# =========================

pipeline.fit(X_train, y_train)


# =========================
# 8. Get feature names
# =========================

feature_names = pipeline.named_steps[
    "preprocessor"
].get_feature_names_out()

importances = pipeline.named_steps[
    "model"
].feature_importances_


# =========================
# 9. Create importance table
# =========================

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)


print("\n===== FEATURE IMPORTANCE =====")
print(importance_df.to_string(index=False))


# =========================
# 10. Plot feature importance
# =========================

plt.figure(figsize=(9, 6))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.xlabel("Feature Importance")
plt.ylabel("Feature")
plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig(
    "feature_importance.png",
    dpi=300
)

plt.show()
plt.close()