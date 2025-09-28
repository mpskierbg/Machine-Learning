# Dimensionality Reduction with Principal Component Analysis (PCA) 🎲

This project demonstrates one of the most fundamental unsupervised learning techniques: **Principal Component Analysis (PCA)**. It addresses the **"curse of dimensionality"**—a common problem in machine learning where datasets have too many features (dimensions), making them slow to process, difficult to visualize, and prone to overfitting.

PCA is a **dimensionality reduction** method that transforms a large set of features into a smaller one while preserving as much of the original data's variance (or "information") as possible.

---

## Project Goal

The objective of this project is to take the scikit-learn Handwritten Digits dataset, which has `64 features` (from an 8x8 pixel image for each digit), and compress it down to just `2 principal components`.

This will allow us to visualize the high-dimensional data on a simple 2D scatter plot and see if the natural groupings of the digits are preserved.



---

## Project Workflow

### 1. Load the Data
We start by loading the digits dataset. The feature matrix `X` has a shape of `(1797, 64)`, meaning there are 1797 images, each described by 64 pixel values.

### 2. Scale the Features
**PCA is highly sensitive to the scale of the features.** It finds the directions of maximum variance, so if one feature has a much larger scale than others (e.g., values from 0-1000 vs. 0-1), PCA will be biased towards that feature. We use `StandardScaler` to normalize the data, giving every feature a mean of 0 and a standard deviation of 1.

### 3. Apply PCA
We instantiate the `PCA` model from scikit-learn and specify our target number of dimensions with `n_components=2`. The `.fit_transform()` method then calculates the principal components from the scaled data and projects the data onto them, reducing the feature count from 64 to 2.

### 4. Visualize the Results
The final step is to create a scatter plot of our new 2-dimensional data.



* The **x-axis** represents the First Principal Component.
* The **y-axis** represents the Second Principal Component.
* Crucially, we color each point using the true digit labels (`y`). This is done only for visualization to verify our results. **The PCA algorithm itself never saw these labels.**

The resulting plot shows distinct clusters for each digit, proving that PCA successfully captured the essential structure of the data in just two dimensions.

---

## The Code

Here is the complete Python script for this analysis:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the high demonsional data
digits = load_digits()
X = digits.data
y = digits.target

print(f"Original data shape: {X.shape}")

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)

# Fit PCA on the scaled data and transform it
X_pca = pca.fit_transform(X_scaled)

print(f"Data shape after PCA: {X_pca.shape}")

# Visualize the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)

plt.title('PCA of Handwritten Digits Dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(handles=scatter.legend_elements()[0], labels=list(digits.target_names))
plt.grid(True)
plt.show()
```
