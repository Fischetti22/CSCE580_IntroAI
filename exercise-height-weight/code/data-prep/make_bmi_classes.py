import pandas as pd

SHEET_CSV = "DataSample-WeightHeight - Sheet1.csv"

# Read directly (your headers are already in row 2 of the file)
df = pd.read_csv(SHEET_CSV, header=1)

# Clean rows (remove zeros/negatives)
df = df[(df["Height (cm)"] > 0) & (df["Weight (kg)"] > 0)].copy()

# Compute BMI (recompute to be safe)
df["Height_m"] = df["Height (cm)"] / 100.0
df["BMI_calc"] = df["Weight (kg)"] / (df["Height_m"] ** 2)

# Two-class label
df["BMI_Class2"] = (df["BMI_calc"] >= 25).astype(int)

print(df[["Height (cm)", "Weight (kg)", "BMI_calc", "BMI_Class2"]].head())

# Save processed file
df.to_csv("bmi_with_classes.csv", index=False)
print("Saved -> bmi_with_classes.csv")

