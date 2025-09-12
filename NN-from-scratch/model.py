# my_neural_net/model.py
import sys
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """sigmoid function

    Args:
        x (float): input number

    Returns:
        float: sigmoid function output
    """
    return 1 / (1 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    """relu function

    Args:
        x (float): input number

    Returns:
        float: x if x > 0 else 0
    """
    return x if x > 0 else 0


def d_relu(x):
    return 1 if x > 0 else 0


class unit:
    def __init__(self, nweights: int, activation):
        self.nweights = nweights
        self.weights = np.random.rand(self.nweights, 1)  # column vector
        self.bias = None  # no bias for now
        self.activation = activation

    def step(self, values):
        """forward input values to get unit output

        Args:
            values (ndarray[ndarray[float]]): input values

        Returns:
            float: unit output
        """
        result = np.dot(values.T, self.weights)
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
        self.units = [unit(prev_size, activation) for _ in range(self.size)]

    def step(self, values):
        """get layer output from input values

        Args:
            values (ndarray[ndarray[float]]): input values as column vector

        Returns:
            ndarray[ndarray[float]]: layer output values as column vector
        """
        return np.array([unit.step(values) for unit in self.units]).reshape(
            -1, 1
        )  # column vector

    def print_info(self):
        print(
            f"""fully connected layer
size: {self.size}
"""
        )


def squared_loss(y_pred, y_label):
    return 0.5 * (y_pred - y_label) ** 2


def d_squared_loss(y_pred, y_label):
    return y_pred - y_label


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
            layer2.units[0].weights[j] -= (
                learning_rate * d_squared_loss(y_pred, Y) * values[1][j][0]
            )
        for j in range(layer1.units[j].nweights):
            for k in range(layer1.size):
                layer1.units[k].weights[j][0] -= (
                    learning_rate
                    * d_squared_loss(y_pred, Y)
                    * layer2.units[0].weights[k]
                    * values[1][k][0]
                    * (1 - values[1][k][0])
                    * X[j]
                )

    def train(self, X, Y, epoch, learning_rate, visualizing=False):
        input_size = len(X)

        print("train start\n")
        results = np.zeros(epoch)
        for i in range(epoch):
            self.fit(X, Y, learning_rate)
            result, values = self.predict(X)
            # print(result)
            results[i] = result

        turns = [i for i in range(1, epoch + 1)]
        training_errors = 0.5 * (results - Y) ** 2
        for i in range(epoch):
            print(training_errors[i])
        if visualizing:
            plt.plot(turns, training_errors)
            plt.xlabel("epoch")
            plt.ylabel("error")
            plt.title("training error")
            plt.show()

    def predict(self, X, logging=False):
        results = X
        values = [X]  # values includes inputs, intermediate values, outputs
        if logging:
            print("predict start\n")
        for i in range(self.nlayer):
            results = self.layers[i].step(results)
            values.append(results)
            if logging:
                print(f"{i}: {results}\n")
        return results[0][0], values

    def print_info(self):
        print(f"nlayer: {self.nlayer}\n")
        for i in range(self.nlayer):
            self.layers[i].print_info()
            print()
