import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

# Load preprocessed data (run from the data/ directory so this file is found)
CSV = "bmi_with_classes.csv"
df = pd.read_csv(CSV)

# Features: Use original measurements (no BMI leakage)
X = df[["Height (cm)", "Weight (kg)"]]
y = df["BMI_Class2"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Train classifier
clf = LogisticRegression(max_iter=1000, solver="liblinear")
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
try:
    auc = roc_auc_score(y_test, y_proba)
except Exception:
    auc = float("nan")

print(f"Accuracy: {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall: {rec:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"ROC AUC: {auc:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Detailed report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=3, zero_division=0))

