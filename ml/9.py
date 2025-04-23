from re import S
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset (2D)
X = np.array(
    [
        [2.5, 2.4],
        [0.5, 0.7],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2.0, 1.6],
        [1.0, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ]
)


X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

cov_matrix = np.cov(X_centered.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
sorted_indices = np.argsort(eigenvalues)[::-1]

principal_component = eigenvectors[:, sorted_indices[0]]


X_pca = X_centered @ principal_component.reshape(-1, 1)

X_reconstructed = X_pca @ principal_component.reshape(1, -1) + X_mean

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], label="Original Data")

pc_line = np.array([X_mean - 2 * principal_component, X_mean + 2 * principal_component])
plt.plot(
    pc_line[:, 0], pc_line[:, 1], color="red", label="Principal Component", linewidth=2
)
plt.scatter(
    X_reconstructed[:, 0], X_reconstructed[:, 1], label="Reconstructed Data", alpha=0.5
)
for i in range(X.shape[0]):
    plt.plot(
        [X[i, 0], X_reconstructed[i, 0]],
        [X[i, 1], X_reconstructed[i, 1]],
        "k--",
        alpha=0.5,
    )
plt.title("PCA: Original and Reconstructed Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.show()
