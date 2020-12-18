import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derive(x):
    return x * (1 - x)


training_inputs = np.array([[-1, -1, -1],
                            [-1, -1, 1],
                            [-1, 1, -1],
                            [-1, 1, 1],
                            [1, -1, -1],
                            [1, -1, 1],
                            [1, 1, -1],
                            [1, 1, 1]])

# training_inputs = np.array([[0, 0, 0],
#                             [0, 0, 1],
#                             [0, 1, 0],
#                             [0, 1, 1],
#                             [1, 0, 0],
#                             [1, 0, 1],
#                             [1, 1, 0],
#                             [1, 1, 1]])

training_outputs = np.array([[-1, -1, -1, 1, -1, 1, 1, 1]]).T
# training_outputs = np.array([[0, 0, 0, 1, 0, 1, 1, 1]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3, 1)) - 1

print('Starting weights:')
print(synaptic_weights)


for iteration in range(1):
    input_l = training_inputs

    outputs = sigmoid(np.dot(input_l, synaptic_weights))

    error = training_outputs - outputs

    adjustments = error * sigmoid_derive(outputs)

    synaptic_weights += np.dot(input_l.T, adjustments)


print('Synaptics weights after:')
print(synaptic_weights)

print('Output after:')
print(outputs)
