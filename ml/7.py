import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None

    def fit(self, X, y):
        self.weights = np.random.rand(X.shape[1])

        for _ in range(self.epochs):
            for i in range(len(X)):
                self.weights += self.learning_rate * X[i] * (y[i] - self.predict(X[i]))

    def predict(self, X):
        return 1 if np.dot(X, self.weights) > 0 else 0


inputs = np.array([[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
outputs = np.array([0, 0, 0, 1])

and_gate = Perceptron(learning_rate=0.1, epochs=10)
and_gate.fit(inputs, outputs)

for i in range(len(inputs)):
    prediction = and_gate.predict(inputs[i])
    print(f"Input: {inputs[i][1:]} -> Output: {prediction}")
