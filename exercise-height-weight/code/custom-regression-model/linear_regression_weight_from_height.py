import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load the cleaned file from the first script
SHEET_CSV = "bmi_with_classes.csv"
df = pd.read_csv(SHEET_CSV)

# Features and labels
X = df[["Height (cm)"]]
y = df["Weight (kg)"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

linreg = LinearRegression().fit(X_train, y_train)
y_pred_test = linreg.predict(X_test)

print(f"Intercept: {linreg.intercept_:.3f}")
print(f"Slope: {linreg.coef_[0]:.3f}")
print(f"R^2 (test): {r2_score(y_test, y_pred_test):.3f}")
print(f"MAE (test): {mean_absolute_error(y_test, y_pred_test):.3f} kg")
print(f"RMSE (test): {np.sqrt(mean_squared_error(y_test, y_pred_test)):.3f} kg")

# Plot regression
plt.scatter(df["Height (cm)"], df["Weight (kg)"], alpha=0.5, label="Actual")
xs = np.linspace(df["Height (cm)"].min(), df["Height (cm)"].max(), 200).reshape(-1,1)
ys = linreg.predict(xs)
plt.plot(xs, ys, label="Linear fit")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.legend()
plt.title("Linear Regression: Weight vs Height")
plt.show()

