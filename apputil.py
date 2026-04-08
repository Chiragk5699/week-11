import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from time import time

# -----------------------------
# Exercise 2 (Part 1 & 2)
# Load diamonds dataset and keep only numeric columns
# -----------------------------
diamonds = sns.load_dataset("diamonds")

# Select only numeric columns (7 columns expected)
diamonds_numeric = diamonds.select_dtypes(include=[np.number])


# -----------------------------
# Exercise 1
# -----------------------------
def kmeans(X, k):
    """
    Perform k-means clustering using sklearn.

    Parameters:
        X (np.array): numerical data of shape (n_samples, n_features)
        k (int): number of clusters

    Returns:
        centroids (np.array): shape (k, n_features)
        labels (np.array): shape (n_samples,)
    """
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(X)

    centroids = model.cluster_centers_
    labels = model.labels_

    return centroids, labels


# -----------------------------
# Exercise 2 (Part 3)
# -----------------------------
def kmeans_diamonds(n, k):
    """
    Run kmeans on first n rows of diamonds numeric dataset.

    Parameters:
        n (int): number of rows
        k (int): number of clusters

    Returns:
        centroids, labels
    """
    X = diamonds_numeric.iloc[:n].values
    return kmeans(X, k)


# -----------------------------
# Exercise 3
# -----------------------------
def kmeans_timer(n, k, n_iter=5):
    """
    Run kmeans_diamonds multiple times and return average runtime.

    Parameters:
        n (int): number of rows
        k (int): number of clusters
        n_iter (int): number of repetitions

    Returns:
        average_time (float): average runtime in seconds
    """
    times = []

    for _ in range(n_iter):
        start = time()
        _ = kmeans_diamonds(n, k)
        elapsed = time() - start
        times.append(elapsed)

    return np.mean(times)

# -----------------------------
# Bonus Exercise
# -----------------------------