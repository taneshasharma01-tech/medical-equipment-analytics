import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Predictive Maintenance System")

st.write(
    "Machine failure prediction using Random Forest "
    "and sensor-based operating parameters."
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("data/ai4i2020.csv")


# ============================================================
# FEATURES AND TARGET
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
# PREPROCESSING
# ============================================================

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
# TRAIN MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

model.fit(X_processed, y)


# ============================================================
# SELECTED DECISION THRESHOLD
# ============================================================

threshold = 0.30


# ============================================================
# SIDEBAR INPUT
# ============================================================

st.sidebar.header("Machine Parameters")

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)

air_temperature = st.sidebar.number_input(
    "Air temperature [K]",
    min_value=295.0,
    max_value=305.0,
    value=300.0,
    step=0.1
)

process_temperature = st.sidebar.number_input(
    "Process temperature [K]",
    min_value=305.0,
    max_value=315.0,
    value=310.0,
    step=0.1
)

rotational_speed = st.sidebar.number_input(
    "Rotational speed [rpm]",
    min_value=1000,
    max_value=3000,
    value=1500,
    step=10
)

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=0.0,
    max_value=100.0,
    value=40.0,
    step=0.5
)

tool_wear = st.sidebar.number_input(
    "Tool wear [min]",
    min_value=0,
    max_value=300,
    value=100,
    step=5
)


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "Predict Machine Failure",
    type="primary"
)


# ============================================================
# DASHBOARD OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Dataset Size",
        f"{len(df):,} machines"
    )

with col2:
    st.metric(
        "Failure Rate",
        f"{y.mean() * 100:.2f}%"
    )

with col3:
    st.metric(
        "Model",
        "Random Forest"
    )


st.divider()


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    new_machine = pd.DataFrame({
        "Type": [machine_type],
        "Air temperature [K]": [air_temperature],
        "Process temperature [K]": [process_temperature],
        "Rotational speed [rpm]": [rotational_speed],
        "Torque [Nm]": [torque],
        "Tool wear [min]": [tool_wear]
    })

    new_machine_processed = preprocessor.transform(
        new_machine
    )

    failure_probability = model.predict_proba(
        new_machine_processed
    )[0][1]

    # ========================================================
    # CLASSIFICATION
    # ========================================================

    if failure_probability >= threshold:

        prediction = "FAILURE"
        risk = "HIGH"

    else:

        prediction = "NORMAL"

        if failure_probability < 0.15:
            risk = "LOW"
        else:
            risk = "MEDIUM"


    # ========================================================
    # RESULTS
    # ========================================================

    st.header("Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)

    with result_col1:

        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.2f}%"
        )

    with result_col2:

        if prediction == "FAILURE":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

    with result_col3:

        if risk == "HIGH":
            st.error(f"Risk Level: {risk}")

        elif risk == "MEDIUM":
            st.warning(f"Risk Level: {risk}")

        else:
            st.success(f"Risk Level: {risk}")


    # ========================================================
    # RECOMMENDATION
    # ========================================================

    st.subheader("Operational Recommendation")

    if prediction == "FAILURE":

        st.warning(
            "Potential failure condition detected. "
            "Consider inspection and preventive maintenance."
        )

    else:

        st.success(
            "No high-risk failure condition detected. "
            "Continue normal operation and monitoring."
        )


else:

    st.info(
        "Enter machine parameters in the sidebar "
        "and click 'Predict Machine Failure'."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.subheader("Model Information")

info_col1, info_col2 = st.columns(2)

with info_col1:

    st.write("**Algorithm:** Random Forest")
    st.write("**Number of trees:** 200")
    st.write("**Decision threshold:** 0.30")
    st.write("**Failure Precision:** 51.35%")
    st.write("**Failure F1-score:** 63.69%")

with info_col2:

    st.write("**Failure Recall:** 83.82%")
    st.write("**Specificity:** 97.20%")
    st.write("**False Positive Rate:** 2.80%")
    st.write("**False Negative Rate:** 16.18%")
    st.write("**ROC-AUC:** 0.9721")


# ============================================================
# INTERPRETATION
# ============================================================

st.divider()

st.subheader("Model Interpretation")

st.write(
    "The decision threshold of 0.30 is selected to prioritize "
    "failure detection over precision."
)

st.write(
    "On the evaluation test set, the model detected 57 out of "
    "68 actual failures, corresponding to an 83.82% failure recall."
)

st.write(
    "The model correctly identified 1,878 out of 1,932 normal "
    "machines, giving a specificity of 97.20%."
)