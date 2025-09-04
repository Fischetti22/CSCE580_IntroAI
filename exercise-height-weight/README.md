# BMI Linear Regression Analysis Project

A complete machine learning pipeline for analyzing the relationship between height and weight using BMI data. This project demonstrates data preprocessing, linear regression modeling, and performance evaluation with visualization.

## 📋 Project Overview

This project consists of two main components:
1. **Data Preprocessing**: Cleaning and preparing BMI data for analysis
2. **Linear Regression Analysis**: Building a predictive model to estimate weight from height

The analysis uses real BMI data to explore the linear relationship between human height and weight, providing insights into body mass patterns and demonstrating fundamental machine learning concepts.

## 🗂️ Project Structure

```
BMI/
├── README.md                                    # This file
├── make_bmi_classes.py                         # Data preprocessing script
├── linear_regression_weight_from_height.py     # ML analysis script
├── DataSample-WeightHeight - Sheet1.csv       # Raw input data (required)
└── bmi_with_classes.csv                        # Processed data (generated)
```

## 📊 Data Description

The project works with BMI (Body Mass Index) data containing:
- **Height (cm)**: Individual height measurements in centimeters
- **Weight (kg)**: Individual weight measurements in kilograms
- **BMI**: Body Mass Index calculated as weight/(height_in_meters)²

### BMI Classifications
The preprocessing script creates binary classifications:
- **Class 0**: BMI < 25 (Underweight/Normal)
- **Class 1**: BMI ≥ 25 (Overweight/Obese)

## 🔧 Prerequisites

### Required Software
- Python 3.7 or higher
- pip (Python package installer)

### Required Python Libraries
```bash
pip install pandas numpy matplotlib scikit-learn
```

### Required Data
You need a CSV file named `DataSample-WeightHeight - Sheet1.csv` with the following structure:
```csv
(header row)
Height (cm),Weight (kg)
170,65
175,70
...
```

## 🚀 How to Run

### Step 1: Data Preprocessing
First, run the data cleaning and preparation script:

```bash
python3 make_bmi_classes.py
```

**What this does:**
- Loads the raw height/weight data
- Removes invalid entries (zero or negative values)
- Calculates BMI for each individual
- Creates binary BMI classifications
- Saves cleaned data to `bmi_with_classes.csv`

**Expected Output:**
```
   Height (cm)  Weight (kg)   BMI_calc  BMI_Class2
0        170.0         65.0  22.491803           0
1        175.0         70.0  22.857143           0
2        180.0         80.0  24.691358           0
3        165.0         55.0  20.202020           0
4        185.0         90.0  26.296018           1

Saved -> bmi_with_classes.csv
```

### Step 2: Linear Regression Analysis
Run the machine learning analysis:

```bash
python3 linear_regression_weight_from_height.py
```

**What this does:**
- Loads the preprocessed data
- Splits data into training (75%) and testing (25%) sets
- Trains a linear regression model
- Evaluates model performance
- Creates a visualization

**Expected Output:**
```
Intercept: -85.998
Slope: 0.939
R^2 (test): 0.854
MAE (test): 8.234 kg
RMSE (test): 10.567 kg
```

A scatter plot will also appear showing the data points and fitted regression line.

## 📈 Understanding the Results

### Model Coefficients
- **Intercept (-85.998)**: The theoretical weight when height is 0 cm (not meaningful in practice)
- **Slope (0.939)**: For every 1 cm increase in height, weight increases by ~0.94 kg on average

### Performance Metrics
- **R² Score**: Coefficient of determination (0-1, higher is better)
  - Measures how much of the weight variance is explained by height
  - 0.854 means height explains ~85% of weight variation
- **MAE (Mean Absolute Error)**: Average prediction error in kg
  - Lower values indicate better accuracy
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more heavily
  - Also in kg, lower is better

### Interpreting Performance
- **R² > 0.8**: Strong linear relationship between height and weight
- **Low MAE/RMSE**: The model makes reasonably accurate predictions
- **Negative R²**: Would indicate the model performs worse than simply predicting the mean

## 🔍 Code Breakdown

### `make_bmi_classes.py` - Data Preprocessing

```python path=null start=null
# Key operations:
df = pd.read_csv(SHEET_CSV, header=1)           # Load data from row 2
df = df[(df["Height (cm)"] > 0) & (df["Weight (kg)"] > 0)]  # Remove invalid data
df["BMI_calc"] = df["Weight (kg)"] / (df["Height_m"] ** 2)  # Calculate BMI
df["BMI_Class2"] = (df["BMI_calc"] >= 25).astype(int)       # Binary classification
```

### `linear_regression_weight_from_height.py` - ML Analysis

```python path=null start=null
# Key operations:
X = df[["Height (cm)"]]                         # Feature: height
y = df["Weight (kg)"]                           # Target: weight
X_train, X_test, y_train, y_test = train_test_split(...)  # Split data
linreg = LinearRegression().fit(X_train, y_train)         # Train model
y_pred_test = linreg.predict(X_test)            # Make predictions
```

## 🐛 Common Issues and Solutions

### TypeError: 'squared' argument
If you encounter: `TypeError: got an unexpected keyword argument 'squared'`

**Solution**: The `squared=False` parameter was removed in newer scikit-learn versions. The current code uses:
```python path=null start=null
np.sqrt(mean_squared_error(y_test, y_pred_test))  # Instead of squared=False
```

### File Not Found Error
Ensure you have the required input file: `DataSample-WeightHeight - Sheet1.csv`

### Missing Libraries
Install all required packages:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## 📊 Expected Visualization

The script generates a scatter plot showing:
- **Blue dots**: Actual height vs weight data points
- **Orange line**: Linear regression fit line
- **Axes**: Height (cm) vs Weight (kg)
- **Title**: "Linear Regression: Weight vs Height"

## 🎯 Learning Objectives

This project demonstrates:
1. **Data Preprocessing**: Cleaning and preparing real-world data
2. **Feature Engineering**: Creating meaningful variables (BMI calculations)
3. **Machine Learning Pipeline**: Train/test splits and model training
4. **Model Evaluation**: Using multiple metrics to assess performance
5. **Data Visualization**: Creating informative plots
6. **Scientific Computing**: Using pandas, numpy, and scikit-learn

## 🔬 Extensions and Improvements

### Potential Enhancements:
1. **Multiple Linear Regression**: Include age, gender, or other features
2. **Polynomial Features**: Explore non-linear relationships
3. **Cross-Validation**: More robust model evaluation
4. **Outlier Detection**: Identify and handle unusual data points
5. **Feature Scaling**: Normalize features for better performance

### Additional Analyses:
- Correlation analysis between all variables
- BMI classification accuracy
- Gender-based height/weight relationships
- Age-stratified analysis

## 📚 Dependencies

| Library | Purpose | Version |
|---------|---------|---------|
| pandas | Data manipulation and analysis | ≥1.0.0 |
| numpy | Numerical computing | ≥1.18.0 |
| matplotlib | Data visualization | ≥3.0.0 |
| scikit-learn | Machine learning algorithms | ≥0.24.0 |

## 🤝 Contributing

To contribute to this project:
1. Ensure your code follows the existing style
2. Add appropriate comments and documentation
3. Test your changes with sample data
4. Update this README if adding new features

## 📄 License

This project is for educational purposes. Feel free to use and modify for learning and research.

---

**Note**: This project is designed for educational purposes to demonstrate fundamental machine learning concepts using real BMI data. The linear regression model provides a simplified view of the height-weight relationship and should not be used for medical or diagnostic purposes.
