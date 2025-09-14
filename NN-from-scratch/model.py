# my_neural_net/model.py
import sys
import numpy as np
import matplotlib.pyplot as plt


def X_scaler(data):
    # scale to standard normal distribution
    mean = np.mean(data, axis=1, keepdims=True)
    var = np.var(data)
    return (data - mean) / np.sqrt(var), mean, var

def Y_scaler(data):
    # scale to standard normal distribution
    mean = np.mean(data)
    var = np.var(data)
    return (data - mean) / np.sqrt(var), mean, var

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
            values (ndarray): input values of (n, m)

        Returns:
            ndarray: unit output of (1, m)
        """
        result = np.dot(values.T, self.weights).T
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
            values (ndarray): input values of (prev_size, m)

        Returns:
            ndarray: layer output values of (size, m)
        """
        # return np.array([unit.step(values) for unit in self.units]).reshape(-1, 1)
        outputs = np.empty((0, values.shape[1]))
        for unit in self.units:
            outputs = np.vstack((outputs, unit.step(values)))
        return outputs

    def print_info(self):
        print(
            f"""fully connected layer
size: {self.size}
"""
        )


def squared_loss(y_pred, y_label):
    """squared loss function
    also support stochastic GD

    Args:
        y_pred (ndarray): numpy array of shape (n, 1)
        y_label (ndarray): numpy array of shape (n, 1)

    Returns:
        float: float valued squared loss
    """
    return np.sum(0.5 * (y_pred - y_label) ** 2)


def d_squared_loss(y_pred, y_label):
    """gradient of squared loss to y_pred
    also support stochastic GD
    Args:
         y_pred (ndarray): numpy array of shape (n, 1)
         y_label (ndarray): numpy array of shape (n, 1)

     Returns:
         ndarray: (n, 1) array of partial derivatives
    """
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

    def fit(self, X, y_label, learning_rate, gd_method):
        y_pred, values = self.predict(X)
        layer1 = self.layers[0]
        layer2 = self.layers[1]

        if gd_method == "stochastic":
            for j in range(layer2.size):
                layer2.units[0].weights[j] -= (
                    learning_rate * d_squared_loss(y_pred, y_label) * values[1][j][0]
                )
            for j in range(layer1.units[j].nweights):
                for k in range(layer1.size):
                    layer1.units[k].weights[j][0] -= (
                        learning_rate
                        * d_squared_loss(y_pred, y_label)
                        * layer2.units[0].weights[k]
                        * values[1][k][0]
                        * (1 - values[1][k][0])
                        * X[j]
                    )
        elif gd_method == "batch":
            input_size = X.shape[1]
            for j in range(layer2.size):
                gradient = 0
                for i in range(input_size):
                    gradient += d_squared_loss(y_pred[i], y_label[i]) * values[1][j][0]
                layer2.units[0].weights[j][0] -= learning_rate * gradient
            for j in range(layer1.units[j].nweights):
                for k in range(layer1.size):
                    gradient = 0
                    for i in range(input_size):
                        gradient += (
                            d_squared_loss(y_pred[i], y_label[i])
                            * layer2.units[0].weights[k][0]
                            * values[1][k][i]
                            * (1 - values[1][k][i])
                            * X[j][i]
                        )
                    layer1.units[k].weights[j][0] -= learning_rate * gradient
        else:
            print(f"gradient descent method {gd_method} unsupported")
            sys.exit()

    def train(self, X, Y, epoch, learning_rate, visualizing=False, gd_method="batch"):
        input_size = X.shape[1]

        print("train start\n")
        results = np.zeros((epoch, input_size))
        for i in range(epoch):
            self.fit(X, Y, learning_rate, gd_method)
            result, values = self.predict(X)
            # print(result)
            results[i] = result

        turns = [i for i in range(1, epoch + 1)]
        training_errors = np.sum(0.5 * (results - Y.reshape(1, -1)) ** 2, axis=1)
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
        return results[0], values

    def print_info(self):
        print(f"nlayer: {self.nlayer}\n")
        for i in range(self.nlayer):
            self.layers[i].print_info()
            print()
