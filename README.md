# Medical Equipment Analytics

## Machine Learning-Based Predictive Maintenance System

A machine learning-based predictive maintenance system for identifying potential machine failures using sensor-based operating parameters.

The system uses a **Random Forest classifier** to estimate machine failure probability and provides an interactive **Streamlit dashboard** for real-time prediction and risk assessment.

---

## Live Demo

**Live Dashboard:**  
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

- `0` -> No failure
- `1` -> Machine failure

The dataset contains **10,000 machine observations** with an overall failure rate of approximately **3.39%**.

---

## Machine Learning Methodology

The project follows the following workflow:

```text
Data Loading
      |
      v
Data Quality Assessment
      |
      v
Exploratory Data Analysis
      |
      v
Data Visualization
      |
      v
Feature Engineering / Preprocessing
      |
      v
Model Development
      |
      v
Feature Importance Analysis
      |
      v
Model Improvement
      |
      v
Model Comparison
      |
      v
Final Evaluation
      |
      v
Decision Threshold Optimization
      |
      v
Model Interpretation
      |
      v
Interactive Dashboard
      |
      v
Streamlit Deployment
```

---

## Model

A **Random Forest Classifier** was developed for machine failure prediction.

The model uses:

- Machine Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

Categorical machine type is handled using **One-Hot Encoding**, while numerical sensor variables are passed directly to the model.

To address class imbalance, the Random Forest model uses **class-weight balancing**.

---

## Model Performance

The final evaluation used a decision threshold of **0.30** to prioritize detection of potential machine failures.

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9721 |
| Failure Recall | 83.82% |
| Failure Precision | 51.35% |
| Failure F1-score | 63.69% |
| Specificity | 97.20% |
| False Positive Rate | 2.80% |
| False Negative Rate | 16.18% |

At the selected threshold, the model detected **57 out of 68 actual failures**, corresponding to a failure recall of **83.82%**.

The model correctly identified **1,878 out of 1,932 normal machines**, resulting in a specificity of **97.20%**.

---

## Decision Threshold Optimization

Because machine failure is the minority class, accuracy alone is not sufficient for evaluating the predictive maintenance system.

A decision threshold of **0.30** was selected to prioritize failure detection.

The system therefore classifies a machine as:

- **NORMAL** when failure probability < 0.30
- **FAILURE** when failure probability >= 0.30

Risk levels are defined as:

- **LOW:** failure probability < 0.15
- **MEDIUM:** 0.15 <= failure probability < 0.30
- **HIGH:** failure probability >= 0.30

---

## Dashboard Features

The Streamlit dashboard provides an interactive interface for entering machine operating parameters and obtaining a failure-risk assessment.

### Input Parameters

Users can enter:

- Machine Type
- Air Temperature [K]
- Process Temperature [K]
- Rotational Speed [rpm]
- Torque [Nm]
- Tool Wear [min]

### Prediction Output

The dashboard provides:

- Failure probability
- Machine failure classification
- Risk level
- Operational recommendation

The dashboard also displays dataset statistics and model performance information.

---

## Operational Recommendation

If the predicted failure probability reaches the selected failure threshold, the dashboard reports a potential failure condition and recommends:

**Schedule inspection / preventive maintenance.**

For lower-risk predictions, the system recommends:

**Continue normal operation and monitoring.**

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Random Forest
- Streamlit
- Git & GitHub

---

## Project Structure

```text
medical-equipment-analytics/
|
|-- data/
|   `-- ai4i2020.csv
|
|-- 01_load_data.py
|-- 02_data_quality.py
|-- 03_eda.py
|-- 04_visualization.py
|-- 05_model.py
|-- 06_feature_importance.py
|-- 07_model_improvement.py
|-- 08_model_comparison.py
|-- 09_final_evaluation.py
|-- 10_threshold_tuning.py
|-- 11_model_interpretation.py
|-- 12_predict_new_machine.py
|-- 13_dashboard.py
|-- 14_feature_importance.py
|
|-- confusion_matrix.png
|-- feature_importance.png
|-- feature_importance_results.csv
|-- precision_recall_curve.png
|-- roc_curve.png
|-- speed_failure.png
|-- threshold_optimization.png
|-- toolwear_failure.png
|-- torque_failure.png
|
|-- requirements.txt
|-- .gitignore
`-- README.md
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/taneshasharma01-tech/medical-equipment-analytics.git
cd medical-equipment-analytics
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit dashboard

```bash
python -m streamlit run 13_dashboard.py
```

The dashboard will open in your browser.

---

## Project Workflow

```text
Data
  |
  v
Data Quality Assessment
  |
  v
Exploratory Data Analysis
  |
  v
Visualization
  |
  v
Preprocessing
  |
  v
Random Forest Model
  |
  v
Feature Importance
  |
  v
Model Improvement
  |
  v
Model Comparison
  |
  v
Final Evaluation
  |
  v
Threshold Optimization
  |
  v
Model Interpretation
  |
  v
Interactive Streamlit Dashboard
  |
  v
Deployment
```

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for predictive maintenance using machine operating parameters.

The developed Random Forest model provides a high ROC-AUC of **0.9721** and achieves **83.82% failure recall** at the selected decision threshold of 0.30.

The Streamlit dashboard converts the trained model into an interactive prediction system that can estimate machine failure probability and provide risk-based maintenance recommendations.

---

## Live Application

**Streamlit Dashboard:**  
https://medical-equipment-analytics-7fg3xc8qeme69mj4jenwbhw.streamlit.app/

**GitHub Repository:**  
https://github.com/taneshasharma01-tech/medical-equipment-analytics