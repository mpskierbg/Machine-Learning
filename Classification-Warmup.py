# 1. Import Libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 2. Load Data
iris = load_iris()
X = iris.data    # The features (sepal length, petal width, etc.)
y = iris.target  # The target (the flower species: 0, 1, or 2)

# 3. Split the Data (The Crucial Step!)
# We use 80% for training and 20% for testing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Initialize and Train the Model
model = DecisionTreeClassifier(random_state=42)
print("Training the Decision Tree Model...")
model.fit(X_train, y_train)
print("Training complete.")

# 5. Make Predictions
y_pred = model.predict(X_test)

# 6. Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)

print("-" * 30)
print(f"Model Predictions on Test Data: {y_pred}")
print(f"True Species of Test Data:    {y_test}")
print(f"Calculated Accuracy:          {accuracy * 100:.2f}%")
print("-" * 30)
