import streamlit as st
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        margin-bottom: 25px;
    }

    .risk-box {
        padding: 18px;
        border-radius: 10px;
        text-align: center;
        font-size: 22px;
        font-weight: 600;
    }

    .section-title {
        font-size: 28px;
        font-weight: 650;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">⚙️ Predictive Maintenance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine failure prediction using Random Forest and '
    'sensor-based operating parameters.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    data = pd.read_csv("data/ai4i2020.csv")

    return data


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

categorical_features = [
    "Type"
]

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
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )
    ]
)


# ============================================================
# PROCESS DATA
# ============================================================

X_processed = preprocessor.fit_transform(X)


# ============================================================
# TRAIN RANDOM FOREST
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
# DATASET RANGES
# ============================================================

air_min = float(df["Air temperature [K]"].min())
air_max = float(df["Air temperature [K]"].max())

process_min = float(df["Process temperature [K]"].min())
process_max = float(df["Process temperature [K]"].max())

speed_min = int(df["Rotational speed [rpm]"].min())
speed_max = int(df["Rotational speed [rpm]"].max())

torque_min = float(df["Torque [Nm]"].min())
torque_max = float(df["Torque [Nm]"].max())

wear_min = int(df["Tool wear [min]"].min())
wear_max = int(df["Tool wear [min]"].max())


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("Machine Parameters")

st.sidebar.caption(
    "Enter operating parameters and run the prediction."
)


# Machine type

machine_type = st.sidebar.selectbox(
    "Machine Type",
    sorted(df["Type"].unique())
)


# Air temperature

air_temperature = st.sidebar.number_input(
    "Air temperature [K]",
    min_value=air_min,
    max_value=air_max,
    value=float(df["Air temperature [K]"].median()),
    step=0.1,
    format="%.1f"
)


# Process temperature

process_temperature = st.sidebar.number_input(
    "Process temperature [K]",
    min_value=process_min,
    max_value=process_max,
    value=float(df["Process temperature [K]"].median()),
    step=0.1,
    format="%.1f"
)


# Rotational speed

rotational_speed = st.sidebar.number_input(
    "Rotational speed [rpm]",
    min_value=speed_min,
    max_value=speed_max,
    value=int(df["Rotational speed [rpm]"].median()),
    step=10
)


# Torque

torque = st.sidebar.number_input(
    "Torque [Nm]",
    min_value=torque_min,
    max_value=torque_max,
    value=float(df["Torque [Nm]"].median()),
    step=0.1,
    format="%.1f"
)


# Tool wear

tool_wear = st.sidebar.number_input(
    "Tool wear [min]",
    min_value=wear_min,
    max_value=wear_max,
    value=int(df["Tool wear [min]"].median()),
    step=1
)


st.sidebar.divider()


predict_button = st.sidebar.button(
    "🔍 Predict Machine Failure",
    type="primary",
    use_container_width=True
)


# ============================================================
# DATASET OVERVIEW
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Dataset Size",
        f"{len(df):,} machines"
    )


with col2:

    failure_rate = y.mean() * 100

    st.metric(
        "Failure Rate",
        f"{failure_rate:.2f}%"
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
    # NEW MACHINE DATA
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
    # PREPROCESS
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


    probability_percent = (
        failure_probability * 100
    )


    # --------------------------------------------------------
    # CLASSIFICATION
    # --------------------------------------------------------

    if failure_probability >= threshold:

        prediction = "FAILURE"
        risk = "HIGH"

    elif failure_probability >= 0.15:

        prediction = "NORMAL"
        risk = "MEDIUM"

    else:

        prediction = "NORMAL"
        risk = "LOW"


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    st.markdown(
        '<div class="section-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    st.write("")


    result_col1, result_col2, result_col3 = st.columns(3)


    # --------------------------------------------------------
    # PROBABILITY
    # --------------------------------------------------------

    with result_col1:

        st.metric(
            "Failure Probability",
            f"{probability_percent:.2f}%"
        )

        st.progress(
            min(failure_probability, 1.0)
        )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    with result_col2:

        if prediction == "FAILURE":

            st.error(
                f"🚨 Prediction: {prediction}"
            )

        else:

            st.success(
                f"✅ Prediction: {prediction}"
            )


    # --------------------------------------------------------
    # RISK
    # --------------------------------------------------------

    with result_col3:

        if risk == "HIGH":

            st.error(
                f"🔴 Risk Level: {risk}"
            )

        elif risk == "MEDIUM":

            st.warning(
                f"🟠 Risk Level: {risk}"
            )

        else:

            st.success(
                f"🟢 Risk Level: {risk}"
            )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    st.subheader("Machine Operating Parameters")

    input_col1, input_col2, input_col3, input_col4, input_col5 = (
        st.columns(5)
    )


    with input_col1:

        st.metric(
            "Machine Type",
            machine_type
        )


    with input_col2:

        st.metric(
            "Air Temp.",
            f"{air_temperature:.1f} K"
        )


    with input_col3:

        st.metric(
            "Process Temp.",
            f"{process_temperature:.1f} K"
        )


    with input_col4:

        st.metric(
            "Speed",
            f"{rotational_speed} rpm"
        )


    with input_col5:

        st.metric(
            "Tool Wear",
            f"{tool_wear} min"
        )


    # ========================================================
    # OPERATIONAL RECOMMENDATION
    # ========================================================

    st.subheader("Operational Recommendation")


    if prediction == "FAILURE":

        st.warning(
            "⚠️ Potential failure condition detected. "
            "Consider inspection and preventive maintenance "
            "before continued operation."
        )

    elif risk == "MEDIUM":

        st.info(
            "⚠️ Moderate failure risk detected. "
            "Continue monitoring machine parameters and "
            "consider preventive inspection."
        )

    else:

        st.success(
            "✅ No high-risk failure condition detected. "
            "Continue normal operation and monitoring."
        )


else:

    st.info(
        "Enter machine parameters in the sidebar and click "
        "'Predict Machine Failure'."
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Model Information</div>',
    unsafe_allow_html=True
)


info_col1, info_col2 = st.columns(2)


with info_col1:

    st.write("**Algorithm:** Random Forest")

    st.write("**Number of trees:** 200")

    st.write("**Class weighting:** Balanced")

    st.write("**Decision threshold:** 0.30")

    st.write("**Failure Precision:** 51.35%")


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

st.markdown(
    '<div class="section-title">Model Interpretation</div>',
    unsafe_allow_html=True
)


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


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-title">Feature Importance</div>',
    unsafe_allow_html=True
)


try:

    categorical_encoder = (
        preprocessor
        .named_transformers_["categorical"]
    )

    categorical_names = (
        categorical_encoder
        .get_feature_names_out(
            categorical_features
        )
    )

    feature_names = (
        numerical_features
        + list(categorical_names)
    )

    importances = model.feature_importances_

    importance_df = pd.DataFrame({

        "Feature": feature_names,

        "Importance": importances

    }).sort_values(
        "Importance",
        ascending=False
    )

    importance_df["Importance"] = (
        importance_df["Importance"] * 100
    )

    st.bar_chart(
        importance_df.set_index("Feature")["Importance"]
    )

    st.caption(
        "Feature importance values represent the relative "
        "contribution of input variables to the Random Forest model."
    )

except Exception:

    st.info(
        "Feature importance visualization is unavailable."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Medical Equipment Analytics | "
    "Machine Learning-Based Predictive Maintenance System"
)