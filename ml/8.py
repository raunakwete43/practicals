# Backpropagation
import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

input_size = 2
output_size = 1
hidden_size = 3

W1 = np.random.uniform(size=(input_size, hidden_size))
b1 = np.random.uniform(size=(1, hidden_size))
W2 = np.random.uniform(size=(hidden_size, output_size))
b2 = np.random.uniform(size=(1, output_size))

epochs = 10000
lr = 0.1


for epoch in range(epochs):
    hidden_input = np.dot(x, W1) + b1
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + b2
    final_output = sigmoid(final_input)

    error = y - final_output
    error_2 = error * sigmoid_derivative(final_output)

    error_1 = np.dot(error_2, W2.T) * sigmoid_derivative(hidden_output)

    W2 += np.dot(hidden_output.T, error_2) * lr
    b2 += np.sum(error_2, axis=0, keepdims=True) * lr
    W1 += np.dot(x.T, error_1) * lr
    b1 += np.sum(error_1, axis=0, keepdims=True) * lr

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error))}")

print("Final output after training:")
print(final_output)
