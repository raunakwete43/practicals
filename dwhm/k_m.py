import numpy as np


class KMeans:
    def __init__(self, k: int, max_iters: int = 100, tolerance: float = 1e-4) -> None:
        self.k = k
        self.max_iters = max_iters
        self.tolerance = tolerance

    def fit(self, X: np.ndarray):
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]

        for i in range(self.max_iters):
            self.labels = self._assign_clusters(X)

            new_centroids = np.array(
                [X[self.labels == j].mean(axis=0) for j in range(self.k)]
            )

            if np.all(np.abs(new_centroids - self.centroids) < self.tolerance):
                break

            self.centroids = new_centroids

    def _assign_clusters(self, X: np.ndarray):
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1)

    def predict(self, X: np.ndarray):
        return self._assign_clusters(X)


X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
# X = np.array(
#     [[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0], [20, 2], [20, 4], [20, 0]]
# )
kmeans = KMeans(k=3)
kmeans.fit(X)

print(f"Centroids = {kmeans.centroids}")
print(f"Labels = {kmeans.labels}")

for idx, label in enumerate(kmeans.labels):
    print(f"{X[idx]} point assigned to cluster {label}")
