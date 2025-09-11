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
    """relu function

    Args:
        x (float): input number

    Returns:
        float: x if x > 0 else 0
    """
    return x if x > 0 else 0


class unit:
    def __init__(self, nweights, activation):
        self.nweights = nweights
        self.weights = np.random.rand(self.nweights)
        self.bias = None  # no bias for now
        self.activation = activation

    def forward(self, values):
        """forward input values to get unit output

        Args:
            values (List[float]): input values

        Returns:
            float: unit output
        """
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


class fc_layer:
    """fully connected layer"""

    def __init__(self, size, prev_size, activation):
        self.size = size
        self.arr = [unit(prev_size, activation) for _ in range(self.size)]

    def forward(self, values):
        """get layer output from input values

        Args:
            values (List[float]): input values

        Returns:
            List[float]: layer output values
        """
        return [unit.forward(values) for unit in self.arr]

    def print_info(self):
        print(
            f"""fully connected layer
size: {self.size}
"""
        )


def squared_loss(y_pred, y_label):
    return 0.5 * (y_pred - y_label) ** 2


class MLP:
    """Multi-Layer Perceptron
    two layer for now
    """

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

    def fit(self, X, Y, learning_rate):
        y_pred, values = self.predict(X)
        layer1 = self.layers[0]
        layer2 = self.layers[1]
        for j in range(layer2.size):
            layer2.arr[0].weights[j] -= learning_rate * (y_pred - Y) * values[1][j]
        for j in range(layer1.size):
            for k in range(layer1.arr[j].nweights):
                layer1.arr[j].weights[k] -= (
                    learning_rate
                    * (y_pred - Y)
                    * layer2.arr[0].weights[j]
                    * values[1][j]
                    * (1 - values[1][j])
                    * X[k]
                )

    def predict(self, X, logging=False):
        results = X
        values = [results]  # including inputs, intermediate values, outputs
        if logging:
            print("predict start\n")
        for i in range(self.nlayer):
            results = self.layers[i].forward(results)
            values.append(results)
            if logging:
                print(f"{i}: {results}\n")
        return results[0], values

    def print_info(self):
        print(f"nlayer: {self.nlayer}\n")
        for i in range(self.nlayer):
            self.layers[i].print_info()
            print()
