import numpy as np
from sklearn.datasets import fetch_openml       ## for loading the mnist data

import sys

def load_reshape_data():
    # load data
    mnist = fetch_openml("mnist_784", version=1)
    pixels, targets = mnist.data, mnist.target

    # reshape data
    pixels = np.array(pixels)
    targets = np.array(targets)

    data = np.column_stack((targets, pixels))
    m, n = data.shape
    np.random.shuffle(data)

    data_dev = data[0:1000]
    data_train = data[1000:]
    y_train = data_train[:, 0].astype(int)
    x_train = data_train[:, 1:].astype(float) / 255.
    return x_train, y_train, data, m, n, data_dev, data_train

def starting_weights_and_biases():
    w1 = np.random.rand(784, 10) - 0.5
    b1 = np.random.rand(10) - 0.5
    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10) - 0.5
    return w1, b1, w2, b2

def rectified_linear_unit(z):
    return np.maximum(0, z)

def rectified_linear_unit_der(z):
    return z>0

def softMax(z):
    shifted = z - np.max(z, axis=1, keepdims=True)
    exp_vals = np.exp(shifted)
    return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

def forward(network_parameters, data):
    w1, b1, w2, b2, = network_parameters
    z1 = data.dot(w1) + b1
    a1 = rectified_linear_unit(z1)
    z2 = a1.dot(w2) + b2
    a2 = softMax(z2)
    activation_parameters = [z1, a1, z2, a2]
    return activation_parameters

def solution_matix(solutions):
    matrix = np.zeros((solutions.size, int(solutions.max()+1)))
    matrix[np.arange(solutions.size), solutions] = 1
    return matrix

def backward(init_parameters, activation_parameters, x, solutions):

    m = x.shape[0]

    w1, b1, w2, b2 = init_parameters
    z1, a1, z2, a2 = activation_parameters
    solutions_matrix = solution_matix(solutions)
    cost = a2 - solutions_matrix
    dw2 = (1/m) * (cost.T.dot(a1)).T
    db2 = 1 / m * np.sum(cost)
    dz1 = (w2.dot(cost.T)).T * rectified_linear_unit_der(z1)
    dw1 = 1 / m * (dz1.T.dot(x)).T
    db1 = 1 / m * np.sum(dz1)
    return dw1, db1, dw2, db2
    
def updating_weights_and_biases(params, updates, alpha):
    w1, b1, w2, b2 = params
    dw1, db1, dw2, db2 = updates
    w1_updated = w1 - alpha * dw1
    b1_updated = b1 - alpha * db1
    w2_updated = w2 - alpha * dw2
    b2_updated = b2 - alpha * db2
    return w1_updated, b1_updated, w2_updated, b2_updated

def model_training(x, y, iterations, alpha):
    weights_and_biases = starting_weights_and_biases()
    for i in range(iterations):
        activations = forward(weights_and_biases, x)
        changes = backward(weights_and_biases, activations, x, y)
        weights_and_biases = updating_weights_and_biases(weights_and_biases, changes, alpha=alpha)
        if i%10 == 0:
            print("######################################")
            print("iteration: ", i)
            print("######################################")
            prediction = np.argmax(activations[3], axis=1)
            print("predictions: ", prediction, "solutions: ", y)
            accuracy = np.sum(prediction == y) / y.size
            print("accruracy: ", accuracy)
    return weights_and_biases

def main():

    x_train, y_train, _, _, _, _, _ = load_reshape_data()
    w1_final, b1_final, w2_final, b2_final = model_training(x_train, y_train, 400, 0.1)
    return 0

if __name__ == "__main__":
    sys.exit(main())