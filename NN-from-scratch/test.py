# my_neural_net/train.py
import numpy as np
import pandas as pd
from .model import MLP, fc_layer, squared_loss


def test():

    input_size = 5
    X = np.random.rand(input_size, 1)
    Y = 10.0
    # two layer fully connected MLP
    print("test start\n")
    print(f"X: {X}\n")
    model = MLP(2)
    model.add_layer(fc_layer(6, input_size, "sigmoid"))
    model.add_layer(fc_layer(1, 6, "linear"))  # buggy if chosen sigmoid

    model.train(X, Y, 20, 0.1, True)

    result, values = model.predict(X)
    print("final error")
    print(squared_loss(result, Y))


# test()


def boston_housing_test():
    input_size = 3
    hidden_size = 6
    alpha = 0.1
    # preprocessing
    dataset_path = (
        "datasets/bostonhousing.csv"  # what is the right path when running as module
    )
    df = pd.read_csv(dataset_path)

    X = df[["RM", "LSTAT", "PTRATIO"]]
    Y = df["MEDV"]

    # two layer fully connected MLP
    print("test start\n")
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    # print(f"X: {X}\n")
    model = MLP(2)
    model.add_layer(fc_layer(hidden_size, input_size, "sigmoid"))
    model.add_layer(fc_layer(1, hidden_size, "linear"))  # buggy if chosen sigmoid

    model.train(X, Y, 20, learning_rate=alpha, visualizing=True)

    result, values = model.predict(X)
    print("final error")
    print(squared_loss(result, Y))


boston_housing_test()
