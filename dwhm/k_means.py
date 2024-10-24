import matplotlib.pyplot as plt
import numpy as np


class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tolerance: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def fit(self, X: np.ndarray) -> None:
        # Randomly initialize cluster centers (centroids)
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for i in range(self.max_iters):
            # Assign clusters based on the closest centroid
            self.labels = self._assign_clusters(X)

            # Compute new centroids by taking the mean of the points in each cluster
            new_centroids = np.array(
                [X[self.labels == j].mean(axis=0) for j in range(self.k)]
            )

            # Check for convergence (if centroids do not change significantly)
            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        # Calculate distances between points and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Assign clusters to new data points based on the closest centroid
        return self._assign_clusters(X)


# 1D Example
print("1D K-Means Clustering")
X_1D = np.array([1, 2, 3, 8, 9, 10, 25, 30, 35]).reshape(-1, 1)
kmeans_1d = KMeans(k=3)
kmeans_1d.fit(X_1D)

print(f"1D Centroids: {kmeans_1d.centroids.ravel()}")
print(f"1D Labels: {kmeans_1d.labels}")

# Plotting 1D Clusters
plt.figure(figsize=(6, 4))
for i, centroid in enumerate(kmeans_1d.centroids):
    cluster = X_1D[kmeans_1d.labels == i]
    plt.scatter(cluster, np.zeros_like(cluster), label=f"Cluster {i+1}")
    plt.scatter(centroid, 0, color="red", marker="x", s=100)
plt.title("1D K-Means Clustering")
plt.xlabel("Data Points")
plt.legend()
plt.show()


# 2D Example
print("\n2D K-Means Clustering")
X_2D = np.array(
    [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], [20, 2], [20, 4], [20, 0]]
)
kmeans_2d = KMeans(k=3)
kmeans_2d.fit(X_2D)

print(f"2D Centroids: {kmeans_2d.centroids}")
print(f"2D Labels: {kmeans_2d.labels}")

# Plotting 2D Clusters
plt.figure(figsize=(6, 6))
colors = ["blue", "green", "orange"]
for i, centroid in enumerate(kmeans_2d.centroids):
    cluster = X_2D[kmeans_2d.labels == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f"Cluster {i+1}", color=colors[i])
    plt.scatter(centroid[0], centroid[1], color="red", marker="x", s=100)
plt.title("2D K-Means Clustering")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()
