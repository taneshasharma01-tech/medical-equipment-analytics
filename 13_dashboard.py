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

@st.cache_data
def load_data():
    return pd.read_csv("data/ai4i2020.csv")


df = load_data()


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
# FEATURE TYPES
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
# PREPROCESSING
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
# TRAIN RANDOM FOREST MODEL
# ============================================================

@st.cache_resource
def train_model(X_processed, y):

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(X_processed, y)

    return model


model = train_model(X_processed, y)


# ============================================================
# DECISION THRESHOLD
# ============================================================

threshold = 0.30


# ============================================================
# DATASET PARAMETER RANGES
# ============================================================

air_min = float(df["Air temperature [K]"].min())
air_max = float(df["Air temperature [K]"].max())

process_min = float(df["Process temperature [K]"].min())
process_max = float(df["Process temperature [K]"].max())

speed_min = int(df["Rotational speed [rpm]"].min())
speed_max = int(df["Rotational speed [rpm]"].max())

torque_min = float(df["Torque [Nm]"].min())
torque_max = float(df["Torque [Nm]"].max())

toolwear_min = int(df["Tool wear [min]"].min())
toolwear_max = int(df["Tool wear [min]"].max())


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Machine Parameters")

st.sidebar.write(
    "Enter operating parameters and run the prediction."
)


# ============================================================
# MACHINE TYPE
# ============================================================

machine_type = st.sidebar.selectbox(
    "Machine Type",
    ["L", "M", "H"]
)


# ============================================================
# AIR TEMPERATURE
# ============================================================

air_temperature = st.sidebar.number_input(
    "Air temperature [K]",
    min_value=air_min,
    max_value=air_max,
    value=300.0,
    step=0.1,
    format="%.1f"
)


# ============================================================
# PROCESS TEMPERATURE
# ============================================================

process_temperature = st.sidebar.number_input(
    "Process temperature [K]",
    min_value=process_min,
    max_value=process_max,
    value=310.0,
    step=0.1,
    format="%.1f"
)


# ============================================================
# ROTATIONAL SPEED
# ============================================================

rotational_speed = st.sidebar.number_input(
    "Rotational speed [rpm]",
    min_value=speed_min,
    max_value=speed_max,
    value=1500,
    step=1
)


# ============================================================
# TORQUE
# ============================================================

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=torque_min,
    max_value=torque_max,
    value=40.0,
    step=0.1,
    format="%.1f"
)


# ============================================================
# TOOL WEAR
# ============================================================

tool_wear = st.sidebar.number_input(
    "Tool wear [min]",
    min_value=toolwear_min,
    max_value=toolwear_max,
    value=100,
    step=1
)


# ============================================================
# SHOW VALID RANGES
# ============================================================

with st.sidebar.expander("Valid parameter ranges"):

    st.write(
        f"**Air temperature:** "
        f"{air_min:.1f} – {air_max:.1f} K"
    )

    st.write(
        f"**Process temperature:** "
        f"{process_min:.1f} – {process_max:.1f} K"
    )

    st.write(
        f"**Rotational speed:** "
        f"{speed_min} – {speed_max} rpm"
    )

    st.write(
        f"**Torque:** "
        f"{torque_min:.1f} – {torque_max:.1f} Nm"
    )

    st.write(
        f"**Tool wear:** "
        f"{toolwear_min} – {toolwear_max} min"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.sidebar.button(
    "Predict Machine Failure",
    type="primary",
    use_container_width=True
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

    # --------------------------------------------------------
    # CREATE NEW MACHINE DATA
    # --------------------------------------------------------

    new_machine = pd.DataFrame({

        "Type": [machine_type],

        "Air temperature [K]": [
            air_temperature
        ],

        "Process temperature [K]": [
            process_temperature
        ],

        "Rotational speed [rpm]": [
            rotational_speed
        ],

        "Torque [Nm]": [
            torque
        ],

        "Tool wear [min]": [
            tool_wear
        ]
    })


    # --------------------------------------------------------
    # PREPROCESS NEW MACHINE
    # --------------------------------------------------------

    new_machine_processed = preprocessor.transform(
        new_machine
    )


    # --------------------------------------------------------
    # FAILURE PROBABILITY
    # --------------------------------------------------------

    failure_probability = model.predict_proba(
        new_machine_processed
    )[0][1]


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if failure_probability >= threshold:

        prediction = "FAILURE"

    else:

        prediction = "NORMAL"


    # --------------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------------

    if failure_probability < 0.15:

        risk = "LOW"

    elif failure_probability < threshold:

        risk = "MEDIUM"

    else:

        risk = "HIGH"


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.header("Prediction Result")


    result_col1, result_col2, result_col3 = st.columns(3)


    # --------------------------------------------------------
    # FAILURE PROBABILITY
    # --------------------------------------------------------

    with result_col1:

        st.metric(
            "Failure Probability",
            f"{failure_probability * 100:.2f}%"
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with result_col2:

        if prediction == "FAILURE":

            st.error(
                f"Prediction: {prediction}"
            )

        else:

            st.success(
                f"Prediction: {prediction}"
            )


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    with result_col3:

        if risk == "HIGH":

            st.error(
                f"Risk Level: {risk}"
            )

        elif risk == "MEDIUM":

            st.warning(
                f"Risk Level: {risk}"
            )

        else:

            st.success(
                f"Risk Level: {risk}"
            )


    # ========================================================
    # OPERATIONAL RECOMMENDATION
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

    st.write("**Class weighting:** Balanced")

    st.write("**Failure Precision:** 51.35%")

    st.write("**Failure F1-score:** 63.69%")


with info_col2:

    st.write("**Failure Recall:** 83.82%")

    st.write("**Specificity:** 97.20%")

    st.write("**False Positive Rate:** 2.80%")

    st.write("**False Negative Rate:** 16.18%")

    st.write("**ROC-AUC:** 0.9721")


# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.divider()

st.subheader("Model Interpretation")


st.write(
    "The decision threshold of 0.30 is selected to "
    "prioritize failure detection over precision."
)


st.write(
    "On the evaluation test set, the model detected "
    "57 out of 68 actual failures, corresponding to "
    "an 83.82% failure recall."
)


st.write(
    "The model correctly identified 1,878 out of "
    "1,932 normal machines, giving a specificity "
    "of 97.20%."
)


# ============================================================
# INPUT SUMMARY
# ============================================================

if predict_button:

    st.divider()

    st.subheader("Machine Input Summary")

    summary_df = pd.DataFrame({
        "Parameter": [
            "Machine Type",
            "Air Temperature [K]",
            "Process Temperature [K]",
            "Rotational Speed [rpm]",
            "Torque [Nm]",
            "Tool Wear [min]"
        ],

        "Value": [
            machine_type,
            f"{air_temperature:.1f}",
            f"{process_temperature:.1f}",
            f"{rotational_speed:.0f}",
            f"{torque:.1f}",
            f"{tool_wear:.0f}"
        ]
    })

    st.table(summary_df)