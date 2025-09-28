# Finding the Optimal Number of Clusters with the Elbow Method 📈

This project demonstrates a crucial technique in **unsupervised learning** called the **Elbow Method**. While the K-Means algorithm is powerful for finding groups in data, it requires us to specify the number of clusters ("K") beforehand. In real-world scenarios, we often don't know the optimal K. The Elbow Method provides a data-driven way to estimate it.

---

## What is Inertia?

The Elbow Method relies on a metric called **inertia**. Inertia is the sum of the squared distances from each data point to the center of its assigned cluster. A lower inertia value means the clusters are more dense and well-defined.

---

## Project Workflow

The goal is to find the value of "K" where adding another cluster no longer provides a significant decrease in inertia.

### 1. Generate Synthetic Data
We begin by creating the same synthetic 2D dataset with 4 distinct clusters using `make_blobs`. This allows us to verify if the Elbow Method can correctly identify the "ground truth" number of clusters we built into the data.

### 2. Looping Through K Values
We create a loop that runs the K-Means algorithm for a range of K values (from 1 to 10). For each value of K, we:

* Instantiate and fit a `KMeans` model.
* Extract the `inertia_` attribute from the fitted model.
* Store this inertia value in a list.

### 3. Plotting the Elbow Curve
After the loop completes, we plot the number of clusters (K) on the x-axis and their corresponding inertia values on the y-axis. The resulting graph typically looks like an arm bending at the elbow.



The **"elbow"** of the curve—the point where the rate of decrease in inertia sharply flattens—is the best estimate for the optimal number of clusters. For our data, this elbow is clearly visible at `K=4`, confirming that the method worked successfully.

### 4. The Code
Here is the complete Python script used for this demonstration:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Generate Synthetic data
X, y_true = make_blobs(n_samples=200, centers=4, cluster_std=0.8, random_state=42)

# Use the elbow method to find the optimal K
inertia_values = []

# Test K from 1 to 10
k_range = range(1, 11)

for k in k_range:
    # Instantiate and fit KMeans model for teh current k
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    # Append the movel's intertia attribute to our list
    inertia_values.append(kmeans.inertia_)

# Plot the elow curve
plt.figure(figsize=(10, 6))
plt.scatter(k_range, inertia_values, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Intertia')
plt.xticks(k_range)
plt.grid(True)
plt.show()
```