import pandas as pd

# Load dataset
df = pd.read_csv("data/ai4i2020.csv")

print("===== DATASET OVERVIEW =====")
print("Shape:", df.shape)

print("\n===== DATA TYPES =====")
print(df.dtypes)

print("\n===== MISSING VALUES =====")
print(df.isnull().sum())

print("\n===== DUPLICATE ROWS =====")
print("Number of duplicates:", df.duplicated().sum())

print("\n===== MACHINE FAILURE =====")
print(df["Machine failure"].value_counts())

print("\n===== MACHINE FAILURE (%) =====")
print(df["Machine failure"].value_counts(normalize=True) * 100)

print("\n===== NUMERICAL SUMMARY =====")
print(df.describe())