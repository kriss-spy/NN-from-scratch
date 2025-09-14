# my_neural_net/train.py
import numpy as np
import pandas as pd
from .model import MLP, fc_layer, squared_loss, X_scaler, Y_scaler


def test():

    input_size = 5
    X = np.random.rand(input_size, 1)
    Y = 10.0
    # two layer fully connected MLP
    print("test start\n")
    print(f"X shape: {X.shape}")

    model = MLP(2)
    model.add_layer(fc_layer(6, input_size, "sigmoid"))
    model.add_layer(fc_layer(1, 6, "linear"))  # buggy if chosen sigmoid

    model.train(X, Y, 20, 0.1, True, "stochastic")

    result, values = model.predict(X)
    print("final error")
    print(squared_loss(result, Y))


def boston_housing_test():
    input_size = 3
    hidden_size = 6
    alpha = 0.01
    # preprocessing
    dataset_path = (
        "datasets/bostonhousing.csv"  # what is the right path when running as module
    )
    df = pd.read_csv(dataset_path)

    X = np.array(df[["RM", "LSTAT", "PTRATIO"]]).T
    Y = np.array(df["MEDV"])

    X, X_mean, Y_var = X_scaler(X)
    Y, Y_mean, Y_var = Y_scaler(Y)

    # two layer fully connected MLP
    print("test start\n")
    print(f"X shape: {X.shape}")  # (3, 489)
    print(f"Y shape: {Y.shape}")  # (489, )

    # print(f"X: {X}\n")
    model = MLP(2)
    model.add_layer(fc_layer(hidden_size, input_size, "sigmoid"))
    model.add_layer(fc_layer(1, hidden_size, "linear"))  # buggy if chosen sigmoid

    model.train(X, Y, 100, learning_rate=alpha, visualizing=True, gd_method="batch")

    result, values = model.predict(X)
    print("final error")
    print(squared_loss(result, Y))


# test()
boston_housing_test()
