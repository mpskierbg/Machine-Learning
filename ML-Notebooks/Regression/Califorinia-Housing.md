Predicting California Housing Prices with Linear Regression

This project demonstrates a simple linear regression model to predict housing prices in California. It uses the popular Scikit-learn library to fetch the dataset, train a model, and evaluate its performance.
Project Overview

The script performs the following steps:

    Loads Data: Fetches the California Housing dataset, which is readily available in Scikit-learn.

    Splits Data: Divides the dataset into a training set (for teaching the model) and a testing set (for evaluating its performance on unseen data).

    Trains Model: Initializes a LinearRegression model and trains it using the training data.

    Makes Predictions: Uses the trained model to predict housing prices on the test set.

    Evaluates Performance: Calculates and prints key performance metrics—Mean Squared Error (MSE) and the R-squared (R2) score—to assess the model's accuracy.

    Displays Results: Shows a sample of the actual prices alongside the model's predicted prices for comparison.

Code Breakdown
1. Imports

The script begins by importing necessary libraries:

    fetch_california_housing: A function to load the dataset.

    pandas: Used for data manipulation and to display the final predictions in a clean format.

    train_test_split: A function to split the data into training and testing sets.

    LinearRegression: The machine learning algorithm we use for this task.

    mean_squared_error & r2_score: Metrics to evaluate the model's performance.

2. Data Loading and Preparation

# Fetch the dataset and load it as a pandas DataFrame
housing = fetch_california_housing(as_frame=True)

# Assign features to X and target to y
X = housing.data
y = housing.target

Here, we load the dataset. The features (like median income, house age, etc.) are assigned to X, and the target variable (the median house value) is assigned to y.
3. Splitting the Data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

We split our data into two parts:

    80% for training: The model will learn from this data.

    20% for testing: We'll use this data to see how well our model performs on data it has never seen before.
    random_state=42 ensures that the split is the same every time we run the code, making our results reproducible.

4. Model Training

# 1. Instantiate the model
model = LinearRegression()

# 2. Fit the model to the training data
model.fit(X_train, y_train)

A LinearRegression model object is created and then trained using the .fit() method on our training data (X_train and y_train).
5. Prediction and Evaluation

# 3. Make predictions on the unseen test data
y_pred = model.predict(X_test)

# 4. Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

    The .predict() method is called on the test features (X_test) to get the model's predictions (y_pred).

    We then compare these predictions to the actual target values (y_test) to calculate our evaluation metrics:

        Mean Squared Error (MSE): The average of the squared differences between the predicted and actual values. A lower MSE is better.

        R-squared (R2) Score: Represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s). It ranges from 0 to 1, with a higher value indicating a better fit.

How to Run This Code

    Prerequisites: Ensure you have Python and the following libraries installed:

        Scikit-learn

        Pandas

    You can install them using pip:

    pip install scikit-learn pandas

    Execution: Save the code as a Python file (e.g., predict_housing.py) and run it from your terminal:

    python predict_housing.py

Example Output

The script will print the evaluation metrics and a sample of the predictions:

--- Model Evaluation ---
Mean Squared Error (MSE): 0.56
R-squared (R2) Score: 0.58

--- Sample Predictions ---
       Actual Price  Predicted Price
20046         0.477         0.952 predicted price
3024          0.458         1.134 predicted price
15663         5.000         2.564 predicted price
20484         2.186         2.842 predicted price
9814          2.780         2.628 predicted price
