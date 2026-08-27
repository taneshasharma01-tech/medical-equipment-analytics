# Predictive Maintenance System for Medical Equipment

## 📌 Project Overview

This project develops a machine learning-based predictive maintenance system for detecting potential machine failures using sensor-based operating parameters.

The system uses a **Random Forest Classifier** to predict whether a machine is likely to experience failure based on parameters such as temperature, rotational speed, torque, tool wear, and machine type.

A **Streamlit dashboard** is also developed to provide an interactive interface for real-time machine failure prediction.

---

## 🎯 Objective

The main objective of this project is to:

- Predict machine failure before it occurs.
- Identify important machine operating parameters associated with failure.
- Optimize the classification threshold to improve failure detection.
- Provide an interpretable machine-learning-based maintenance decision system.
- Develop an interactive dashboard for practical use.

---

## 📊 Dataset

The project uses the **AI4I 2020 Predictive Maintenance Dataset**.

### Dataset size

- Total samples: **10,000**
- Total original features: **14**
- Target variable: **Machine failure**

### Main input parameters

- Machine Type
- Air temperature [K]
- Process temperature [K]
- Rotational speed [rpm]
- Torque [Nm]
- Tool wear [min]

The dataset also contains failure-mode indicators such as:

- TWF — Tool Wear Failure
- HDF — Heat Dissipation Failure
- PWF — Power Failure
- OSF — Overstrain Failure
- RNF — Random Failure

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Random Forest
- Git/GitHub

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Loading
     ↓
Data Quality Analysis
     ↓
Exploratory Data Analysis
     ↓
Data Visualization
     ↓
Random Forest Model
     ↓
Model Improvement
     ↓
Model Comparison
     ↓
Final Model Evaluation
     ↓
Threshold Optimization
     ↓
Model Interpretation
     ↓
New Machine Prediction
     ↓
Streamlit Dashboard