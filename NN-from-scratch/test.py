# my_neural_net/train.py
import numpy as np
from .model import MLP, fc_layer, squared_loss

input_size = 5
X = np.random.rand(input_size, 1)
Y = 10.0


def test():
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


test()
