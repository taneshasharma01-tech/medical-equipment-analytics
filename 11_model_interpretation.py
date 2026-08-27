# ============================================================
# 11_model_interpretation.py
# Interpret final Random Forest performance
# ============================================================

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# FINAL CONFUSION MATRIX
# Threshold = 0.30
# ============================================================

cm = [[1882, 50],
      [14, 54]]

tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]

# ============================================================
# CALCULATE METRICS
# ============================================================

accuracy = (tp + tn) / (tp + tn + fp + fn)

precision = tp / (tp + fp)

recall = tp / (tp + fn)

specificity = tn / (tn + fp)

fpr = fp / (fp + tn)

fnr = fn / (fn + tp)

npv = tn / (tn + fn)

f1 = 2 * (precision * recall) / (precision + recall)

# ============================================================
# DISPLAY RESULTS
# ============================================================

print("=" * 60)
print("FINAL MODEL INTERPRETATION")
print("=" * 60)

print("\nThreshold: 0.30")

print("\n===== CONFUSION MATRIX =====")
print("True Negatives  (TN):", tn)
print("False Positives (FP):", fp)
print("False Negatives (FN):", fn)
print("True Positives  (TP):", tp)

print("\n===== PERFORMANCE METRICS =====")

print(f"Accuracy:             {accuracy:.4f}")
print(f"Failure Precision:    {precision:.4f}")
print(f"Failure Recall:       {recall:.4f}")
print(f"Failure F1-score:     {f1:.4f}")
print(f"Specificity:          {specificity:.4f}")
print(f"False Positive Rate:  {fpr:.4f}")
print(f"False Negative Rate:  {fnr:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")

# ============================================================
# PERCENTAGES
# ============================================================

print("\n===== PERCENTAGE INTERPRETATION =====")

print(f"Failure detection rate: {recall * 100:.2f}%")

print(f"Missed failure rate:     {fnr * 100:.2f}%")

print(f"Normal machine correctly identified: "
      f"{specificity * 100:.2f}%")

print(f"False alarm rate:        {fpr * 100:.2f}%")

# ============================================================
# OPERATIONAL INTERPRETATION
# ============================================================

print("\n===== OPERATIONAL INTERPRETATION =====")

print(
    f"\nOut of {tp + fn} actual machine failures, "
    f"the model detected {tp}."
)

print(
    f"The model missed {fn} actual failures."
)

print(
    f"Out of {tn + fp} normal machines, "
    f"{tn} were correctly identified as normal."
)

print(
    f"The model generated {fp} false alarms."
)

print(
    "\nA threshold of 0.30 prioritizes failure detection "
    "over precision."
)

print(
    "This reduces missed failures compared with the "
    "default 0.50 threshold, but increases false alarms."
)

print("\n" + "=" * 60)
print("INTERPRETATION COMPLETE")
print("=" * 60)