# Linear Regression

import matplotlib.pyplot as plt
from scipy import stats
import numpy as np

X = np.random.randn(100)
y = 2 * X + np.random.randn(100) * 0.5
plt.scatter(X, y, alpha=0.5)


slope, intercept, r_value, p_value, std_err = stats.linregress(X, y)
print(f"Slope: {slope}, Intercept: {intercept}, R-squared: {r_value**2}")

plt.plot(X, slope * X + intercept, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
