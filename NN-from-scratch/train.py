# my_neural_net/train.py
import numpy as np
from .model import MLP, layer

input_size = 5
X = np.random.random(input_size)


def test():
    print("test start\n")
    model = MLP(2)
    model.add_layer(layer(6, input_size, "sigmoid"))
    model.add_layer(layer(1, 6, "sigmoid"))
    results = model.predict(X)
    print(results)


test()
