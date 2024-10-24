import numpy as np

X = np.arange(1, 10, 1)
y = X * 2 + 3

X_mean = np.mean(X)
y_mean = np.mean(y)

num = np.sum((X - X_mean) * (y - y_mean))
deno = np.sum((X - X_mean) ** 2)

m = num / deno
b = y_mean - m * X_mean


print(f"Slope(m) = {m}, Intercept(b) = {b}")
