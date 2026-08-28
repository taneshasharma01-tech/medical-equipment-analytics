# Medical Equipment Analytics

## Machine Learning-Based Predictive Maintenance System

A machine learning-based predictive maintenance system for identifying potential machine failures using sensor-based operating parameters.

The system uses a **Random Forest classifier** to estimate machine failure probability and provides an interactive **Streamlit dashboard** for real-time prediction and risk assessment.

---

## 🚀 Live Demo

👉 **Live Dashboard:**  
https://medical-equipment-analytics-7fg3xc8qeme69mj4jenwbhw.streamlit.app/

---

## Problem Statement

Unexpected equipment failure can lead to machine downtime, increased maintenance costs, and reduced operational efficiency.

The objective of this project is to develop a predictive maintenance system that uses machine operating parameters to identify machines that may be at risk of failure before an actual failure occurs.

---

## Objective

- Analyze machine operating parameters associated with machine failure.
- Explore relationships between sensor variables and machine failure.
- Develop a machine-learning classification model.
- Optimize the decision threshold for improved failure detection.
- Interpret important predictive features.
- Deploy an interactive machine failure prediction dashboard.

---

## Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

The dataset contains machine operating parameters including:

- Machine Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

### Target Variable

**Machine Failure**

- `0` → No failure
- `1` → Machine failure

The dataset contains **10,000 machine observations** with an overall failure rate of approximately **3.39%**.

---

## Machine Learning Methodology

The project follows the following workflow:

```text
Data Loading
      ↓
Data Quality Assessment
      ↓
Exploratory Data Analysis
      ↓
Data Visualization
      ↓
Feature Engineering / Preprocessing
      ↓
Model Development
      ↓
Feature Importance Analysis
      ↓
Model Improvement
      ↓
Model Comparison
      ↓
Final Evaluation
      ↓
Decision Threshold Optimization
      ↓
Model Interpretation
      ↓
Interactive Dashboard
      ↓
Streamlit Deployment