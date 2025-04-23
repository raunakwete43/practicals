import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs, make_classification

X, y = make_blobs(n_samples=100, centers=2, random_state=6, cluster_std=1.5)

svm = SVC(kernel="linear", C=1.0)
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]

x0 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x1 = -(w[0] * x0 + b) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap="autumn")
plt.plot(x0, x1, "k-", label="Hyperplane")

plt.scatter(
    svm.support_vectors_[:, 0],
    svm.support_vectors_[:, 1],
    s=100,
    facecolors="none",
    edgecolors="k",
    label="Support Vectors",
)

plt.title("SVM with Linear Kernel")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()


X, y = make_classification(
    n_features=2, n_classes=2, n_informative=2, n_redundant=0, random_state=42
)

svm = SVC(kernel="linear")
svm.fit(X, y)

w = svm.coef_[0]
b = svm.intercept_[0]


x0 = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
x1 = -(w[0] * x0 + b) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap="autumn")
plt.plot(x0, x1, "k-", label="Hyperplane")

plt.show()
