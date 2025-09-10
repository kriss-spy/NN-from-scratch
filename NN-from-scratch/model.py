# my_neural_net/model.py
import sys
import numpy as np


def sigmoid(x):
    """sigmoid function

    Args:
        x (float): input number

    Returns:
        float: sigmoid function output
    """
    return 1 / (1 + np.exp(-x))


def relu(x):
    return x if x > 0 else 0


class unit:
    def __init__(self, nweights, activation):
        self.weights = np.zeros(nweights)
        self.bias = None  # no bias for now
        self.activation = activation

    def forward(self, values):
        result = np.dot(values, self.weights)
        if self.activation == "linear":
            return result
        elif self.activation == "sigmoid":
            return sigmoid(result)

        elif self.activation == "relu":
            return relu(result)

        else:
            print(f"[ERROR] unsupported activation function {self.activation}")
            sys.exit()


class layer:
    # fully connected layer
    def __init__(self, size, prev_size, activation):
        self.size = size
        self.arr = [unit(prev_size, activation) for _ in range(self.size)]

    def forward(self, values):
        return [unit.forward(values) for unit in self.arr]


class MLP:
    def __init__(self, nlayer):
        self.layers = list()
        self.nlayer = nlayer

    def add_layer(self, layer):
        if len(self.layers) >= self.nlayer:
            print(f"[ERROR] exceeded model nlayer {self.nlayer}\n")
            sys.exit()
        self.layers.append(layer)

    def pop_layer(self, layer):
        try:
            self.layers.pop()
        except IndexError:
            print("[ERROR] layers empty, cannot pop layer\n")

    def fit(self, X, Y):
        pass

    def predict(self, X):
        results = X
        for i in range(self.nlayer):
            results = self.layers[i].forward(results)
        return results
