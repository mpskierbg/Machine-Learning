🍇 Classification Project 3: SVM & Data Preprocessing

I. Project Goal & Data

    Goal: Identify which of three chemical cultivars (types) a wine sample belongs to.

    Model Used: Support Vector Classifier (SVC), a powerful but scale-sensitive model.

    New Concept: Data Preprocessing (Standard Scaling).

    Dataset: load_wine (Scikit-learn built-in).

II. The Critical New Step: Standard Scaling

The Wine dataset features (like Alcohol vs. Proline) are on vastly different numerical scales (e.g., 12 vs. 1600). The SVC model performs poorly if these large-scale differences are not normalized, as the features with huge numbers unfairly dominate the decision-making process.
Scenario	Model	Accuracy Observed	Conceptual Takeaway
Without Scaling	SVC	[Insert LOW Accuracy Score Here]	The model suffers from feature dominance (low accuracy).
With Scaling	SVC	[Insert HIGH Accuracy Score Here]	Scaling is essential for distance-based models like SVM and KNN.

The Scaling Workflow

To prevent data leakage (biasing the model with test data information), the scaling is applied in two distinct steps:

    Fit (Learning): The StandardScaler is fitted ONLY on the X_train data to learn the mean and standard deviation for each feature.

    Transform (Applying): The learned rules are then applied to BOTH the X_train and the held-out X_test data.