🎯 Classification Project 2: K-Nearest Neighbors (KNN)

I. Project Goal & Data

    Goal: Classify tumors as either Malignant (cancerous) or Benign (non-cancerous) based on cell characteristics.

    Model Used: K-Nearest Neighbors (KNN) Classifier.

    New Concept: Hyperparameter Tuning (specifically optimizing the value of K).

    Dataset: load_breast_cancer (Scikit-learn built-in).

II. The KNN Model & The Problem

Unlike the Decision Tree, the KNN model classifies a new data point by looking at the K nearest data points around it and taking a "majority vote." The crucial decision is choosing the optimal value for K (the hyperparameter n_neighbors).
Setting (K)	Conceptual Result	Accuracy Observed
K=1	High Variance (Overfitting): Model is too complex, memorizing noise.	[Insert High K=1 Accuracy Score Here]
K=20	High Bias (Underfitting): Model is too simple, averaging over too many points.	[Insert Low K=20 Accuracy Score Here]
Optimal K	Ideal Balance: Generalizes well to unseen data.	[Insert Best Accuracy Score Here]

III. Best Practice: Hyperparameter Tuning

Relying on a single K value is poor practice. The professional solution is to use a Grid Search to automatically test a range of values and find the K that minimizes error on the test data.

1. The Bias-Variance Tradeoff

The search for the best K is a search for the best tradeoff between:

    Bias: Errors due to a too-simple model (High K).

    Variance: Errors due to a too-complex model (Low K).

2. Grid Search Implementation (Conceptual)

Instead of manually changing K, a GridSearchCV object automates the process:

    Define a Grid: List all the K values to test (e.g., [1, 3, 5, 7, ..., 21]).

    Cross-Validation: The Grid Search repeatedly trains and tests the model with these values on different splits of the training data.

    Result: The output is the best_params_ and best_score_, giving the optimal K value to use in the final model.