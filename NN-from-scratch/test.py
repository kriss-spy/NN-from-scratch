# my_neural_net/train.py
import numpy as np
import pandas as pd
from .model import MLP, fc_layer, squared_loss, X_scaler, Y_scaler


def test():
    n = 100
    input_size = 5
    hidden_size = 5
    X = np.random.rand(input_size, n)
    Y = np.array([0.5])  # no regularization, so between 0 and 1

    # two layer fully connected MLP
    print("test start\n")
    print(f"X shape: {X.shape}")  # (3, 489)
    print(f"Y shape: {Y.shape}")  # (489, )

    model = MLP(2)
    model.add_layer(fc_layer(hidden_size, input_size, "sigmoid"))
    model.add_layer(fc_layer(1, hidden_size, "sigmoid"))

    model.train(X, Y, 20, 0.001, True, "batch")

    result, values = model.predict(X)
    print("final error")
    print(squared_loss(result, Y))


def test_and_gate():
    """Test MLP on AND logic gate data with lower learning rate and error curve"""
    # AND gate dataset
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])  # shape (2, 4)
    Y = np.array([0, 0, 0, 1])  # shape (4,)

    # Build MLP: 2 inputs, 2 hidden units, 1 output
    mlp = MLP(nlayer=2)
    mlp.add_layer(fc_layer(size=2, prev_size=2, activation_fn="sigmoid"))
    mlp.add_layer(fc_layer(size=1, prev_size=2, activation_fn="sigmoid"))

    print("Predictions before training:")
    preds, _ = mlp.predict(X)
    print(np.round(preds, 3))

    # Lower learning rate, more epochs
    epochs = 3000
    lr = 0.01
    input_size = X.shape[1]
    results = np.zeros((epochs, input_size))
    errors = np.zeros(epochs)
    for i in range(epochs):
        mlp.fit(X, Y, lr, "batch", False)
        result, _ = mlp.predict(X)
        results[i] = result
        errors[i] = np.sum(0.5 * (result - Y.reshape(1, -1)) ** 2)
        if i % 500 == 0:
            print(f"{i}/{epochs}: error={errors[i]}")

    print("Predictions after training:")
    preds, _ = mlp.predict(X)
    print(np.round(preds, 3))
    print("Expected:", Y)
    # Plot error curve
    import matplotlib.pyplot as plt

    plt.plot(range(epochs), errors)
    plt.xlabel("epoch")
    plt.ylabel("training error")
    plt.title("AND gate training error curve")
    plt.show()


def boston_housing_test():
    input_size = 3
    hidden_size = 3
    alpha = 0.00001
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
    model.add_layer(fc_layer(1, hidden_size, "sigmoid"))  # buggy if chosen sigmoid

    model.train(
        X,
        Y,
        500,
        learning_rate=alpha,
        visualizing=True,
        gd_method="batch",
        check_explode=True,
    )

    result, values = model.predict(X)
    print("final error")
    print(squared_loss(result, Y))


# test()
# test_and_gate()
boston_housing_test()
