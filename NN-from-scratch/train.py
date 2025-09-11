# my_neural_net/train.py
import numpy as np
from .model import MLP, fc_layer

input_size = 5
X = np.random.random(input_size)
Y = 11.0


def test():
    # two layer fully connected MLP
    print("test start\n")
    print(f"X: {X}\n")
    model = MLP(2)
    model.add_layer(fc_layer(6, input_size, "sigmoid"))
    model.add_layer(fc_layer(1, 6, "sigmoid"))
    # model.print_info()

    results, values = model.predict(X)
    print(results)

    model.fit(X, Y, 0.1)
    results, values = model.predict(X)
    print(results)

    model.fit(X, Y, 0.1)
    results, values = model.predict(X)
    print(results)


# test()


def train(epoch, learning_rate):
    # two layer fully connected MLP
    print(f"X: {X}\n")
    model = MLP(nlayer=2)
    model.add_layer(fc_layer(size=6, prev_size=input_size, activation="sigmoid"))
    model.add_layer(fc_layer(size=1, prev_size=6, activation="linear"))
    # model.print_info()

    print("train start\n")

    for i in range(epoch):
        model.fit(X, Y, learning_rate)
        results, values = model.predict(X)
        print(results)

    return model


train(50, 0.1)
