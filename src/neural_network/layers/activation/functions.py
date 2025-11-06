import numpy as np
from .layer import Activation

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(x):
            sigmoid_x = sigmoid(x)
            return sigmoid_x * (1 - sigmoid_x)

        super().__init__(sigmoid, sigmoid_derivative)
