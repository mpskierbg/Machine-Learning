# Predicting California Housing Prices with Random Forest Regression

This project demonstrates how to build and evaluate a machine learning model to predict housing prices using the California Housing dataset from scikit-learn. The primary goal is to show the workflow of establishing a baseline model and then improving upon it with an more advanced algorithm.

## Project Overview

The process involves several key steps in a typical machine learning workflow:

* **Data Loading:** The dataset is loaded directly from scikit-learn.
* **Data Splitting:** The data is split into training and testing sets to ensure an unbiased evaluation of the model.
* **Feature Scaling:** `StandardScaler` is used to standardize the features, which is a crucial preprocessing step for many algorithms.
* **Modeling:** We first establish a baseline using a simple `LinearRegression` model and then train a more powerful `RandomForestRegressor` model.
* **Evaluation:** The models are evaluated using the R-squared (R²) score and Mean Squared Error (MSE) to measure their performance.

## The Dataset

The model is trained on the California Housing dataset. The target variable is the median house value for California districts, expressed in hundreds of thousands of dollars ($100,000).

The features include metrics such as:

* `MedInc`: Median income in block group
* `HouseAge`: Median house age in block group
* `AveRooms`: Average number of rooms per household
* `AveBedrms`: Average number of bedrooms per household
* `Population`: Block group population
* `AveOccup`: Average number of household members
* `Latitude`: Block group latitude
* `Longitude`: Block group longitude

## Results: From Baseline to Final Model

Our initial `LinearRegression` model established a baseline performance with an R-squared score of **0.58**.

By switching to a `RandomForestRegressor`, which can capture more complex, non-linear patterns in the data, we achieved a significant improvement.

**Final Model Performance:**

* **R-Squared Score:** 0.81
* **Mean Squared Error:** 0.26

The R-squared score indicates that our final model successfully explains **81%** of the variability in California housing prices, a dramatic improvement over the baseline.

## How to Run the Code

To replicate this project, you can run the Python script below. Ensure you have the necessary libraries installed.

### Dependencies

* `scikit-learn`
* `pandas`

You can install them using pip:

```bash
pip install scikit-learn pandas
```

Python Script

```
# Import necessary libraries
from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# 1. Fetch the dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# 2. Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Instantiate and train the RandomForestRegressor model
# n_jobs=-1 uses all available CPU cores to speed up training.
model = RandomForestRegressor(random_state=42, n_jobs=-1)
model.fit(X_train_scaled, y_train)

print("Training complete!")

# 5. Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation (Random Forest) ---")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R2) Score: {r2:.2f}")
```