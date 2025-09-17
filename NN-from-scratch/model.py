# my_neural_net/model.py
import sys
import numpy as np
import matplotlib.pyplot as plt


def X_scaler(data):
    # scale to standard normal distribution
    mean = np.mean(data, axis=1, keepdims=True)
    var = np.var(data, axis=1, keepdims=True)
    return (data - mean) / np.sqrt(var), mean, var


def Y_scaler(data):
    # scale to standard normal distribution
    mean = np.mean(data)
    var = np.var(data)
    return (data - mean + 0.5) / np.sqrt(var), mean, var


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


def check_weight_explode(model):
    v = 100
    for layer in model.layers:
        if np.max(layer.weight_mat) > v:
            print(f"pamameter explosion detected!")
            breakpoint()


class unit:
    def __init__(self, nweights: int, activation_fn):
        self.nweights = nweights
        self.weights = np.random.rand(self.nweights, 1)  # column vector
        self.bias = None  # no bias for now
        self.activation_fn = activation_fn

    def step(self, values):
        """forward input values to get unit output

        Args:
            values (ndarray): input values of (n, m)

        Returns:
            ndarray: unit output of (1, m)
        """
        result = np.dot(values.T, self.weights).T
        if self.activation_fn == "linear":
            return result
        elif self.activation_fn == "sigmoid":
            return sigmoid(result)

        elif self.activation_fn == "relu":
            return relu(result)

        else:
            print(f"[ERROR] unsupported activation function {self.activation_fn}")
            sys.exit()


class fc_layer:
    """fully connected layer"""

    def __init__(self, size, prev_size, activation_fn):
        self.nunit = size
        self.ninput = prev_size
        # self.units = [unit(prev_size, activation_fn) for _ in range(self.nunit)]
        self.weight_mat = np.random.rand(self.ninput, self.nunit) * 0.1
        # self.weight_mat = np.zeros((self.ninput, self.nunit))
        self.bias_mat = None  # TODO implement bias
        self.activation_fn = activation_fn

    def step(self, values):
        """get layer output from input values

        Args:
            values (ndarray): input values of (input_size, m)

        Returns:
            ndarray: layer output values of (nunit, m)
        """
        # Linear pre-activation
        z = self.weight_mat.T @ values  # (nunit, m)
        # Apply activation
        if self.activation_fn == "linear":
            return z
        elif self.activation_fn == "sigmoid":
            return sigmoid(z)
        elif self.activation_fn == "relu":
            return relu(z)
        else:
            print(f"[ERROR] unsupported activation function {self.activation_fn}")
            sys.exit()

    def print_info(self):
        print(
            f"""fully connected layer
size: {self.nunit}
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

    def fit(self, X, y_label, learning_rate, gd_method, check_explode):
        y_pred, values = self.predict(X)
        layer1 = self.layers[0]
        layer2 = self.layers[1]

        if gd_method == "stochastic":
            pass  # fix stochastic GD

        elif gd_method == "batch":
            input_size = X.shape[1]

            # layer 1 weights udpate
            layer1.weight_mat -= (
                learning_rate
                * X
                @ (
                    (values[1].T @ (1 - values[1]))
                    @ (y_pred.reshape(-1, 1) - y_label.reshape(-1, 1))
                )
                @ layer2.weight_mat.T
            )

            # layer 2 weights update
            layer2.weight_mat -= (
                learning_rate * values[1] @ (y_pred - y_label)
            ).reshape(-1, 1)

            if check_explode:
                check_weight_explode(self)

        else:
            print(f"gradient descent method {gd_method} unsupported")
            sys.exit()

    def train(
        self,
        X,
        Y,
        epoch,
        learning_rate,
        visualizing=False,
        gd_method="batch",
        error_goal=1.0,
        check_explode=False,
    ):
        input_size = X.shape[1]

        print("train start\n")
        results = np.zeros((epoch, input_size))
        finished = 0
        while finished < epoch:
            self.fit(X, Y, learning_rate, gd_method, check_explode)
            result, values = self.predict(X)
            finished += 1

            if squared_loss(result, Y) <= error_goal:
                break

            print(f"{finished}/{epoch}: {squared_loss(result, Y.reshape(1, -1))}")
            results[finished - 1] = result

        turns = [i for i in range(1, finished + 1)]
        results = results[0:finished]
        training_errors = np.sum(0.5 * (results - Y.reshape(1, -1)) ** 2, axis=1)
        # for i in range(epoch):
        # print(training_errors[i])
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
                print(f"{i+1}: {results}\n")
        return results[0], values

    def print_info(self):
        print(f"nlayer: {self.nlayer}\n")
        for i in range(self.nlayer):
            self.layers[i].print_info()
            print()
