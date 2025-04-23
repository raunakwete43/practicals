import numpy as np

# Input patterns for AND gate (including bias)
X = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
y = np.array([0, 0, 0, 1])

# Initialize weights randomly
weights = np.random.rand(3)

# Training
learning_rate = 0.1
epochs = 10

print("Training the network...")
for _ in range(epochs):
    for i in range(len(X)):
        weights += learning_rate * X[i] * y[i]

print("\nTesting the network...")
print("Input\t\tOutput")
for i in range(len(X)):
    prediction = 1 if np.dot(X[i], weights) > 0 else 0
    print(f"{X[i][1:]} -> {prediction}")


print(weights)
