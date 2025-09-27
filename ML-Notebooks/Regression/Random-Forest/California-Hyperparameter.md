# Optimizing the Random Forest Model with Hyperparameter Tuning

## What is Hyperparameter Tuning?

**Hyperparameters** are the "settings" of a machine learning model that are set *before* the training process begins. Unlike the model's internal parameters (which are learned from the data), we have to choose the best hyperparameters ourselves. ⚙️

The goal of tuning is to systematically search for the combination of hyperparameter values that results in the best model performance.

---

## Method: Randomized Search Cross-Validation

For this project, we used `RandomizedSearchCV` from scikit-learn. This technique is highly effective for several reasons:

* **Efficiency:** Instead of trying every single possible combination (which can be computationally impossible), it samples a fixed number of random combinations from a defined search space.
* **Robustness:** It uses cross-validation to evaluate each combination, which gives a more reliable estimate of performance than a single train-test split.

---

## Tuning Process

### 1. Defining the Parameter Grid
We defined a search space for some of the most influential hyperparameters of the Random Forest model:

* `n_estimators`: The number of trees in the forest.
* `max_depth`: The maximum depth of each tree.
* `min_samples_split`: The minimum number of samples required to split a node.
* `min_samples_leaf`: The minimum number of samples required at a leaf node.
* `max_features`: The number of features to consider when looking for the best split.

### 2. Running the Search
We configured `RandomizedSearchCV` to try 100 different combinations of these parameters, using 3-fold cross-validation for each iteration.

### 3. The Code
Here is the Python script used for the tuning process:

```python
# Import necessary libraries
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- Setup (Same as before) ---
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the grid of hyperparameters to search
param_grid = {
    'n_estimators': [int(x) for x in np.linspace(start=100, stop=1200, num=12)],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [int(x) for x in np.linspace(5, 30, num=6)],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate the base model and the search object
rf = RandomForestRegressor(random_state=42)
rf_random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_grid,
                                      n_iter=100, cv=3, verbose=2,
                                      random_state=42, n_jobs=-1)

# 3. Fit the search to the data
rf_random_search.fit(X_train_scaled, y_train)

# 4. Evaluate the best model found
best_model = rf_random_search.best_estimator_
y_pred = best_model.predict(X_test_scaled)
r2 = r2_score(y_test, y_pred)

print(f"Best Parameters: {rf_random_search.best_params_}")
print(f"Tuned R-squared Score: {r2:.2f}")
```

## Results: Performance Improvement 🏆

The hyperparameter tuning process successfully found a better set of parameters than the default ones.

* Before Tuning (Default Model):
    * R-Squared Score: 0.81

* After Tuning (Optimized Model):
    * R-Squared Score: 0.82

This demonstrates how tuning can squeeze out additional performance from an already strong model. The best parameters found were: ```{'n_estimators': 600, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 25}.```