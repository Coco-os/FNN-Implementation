import numpy as np
from ..layer import Layer


class Activation(Layer):
    def __init__(self, activation, activation_derivative):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_derivative
        self.optimizer = None

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, optimizer):
        return np.multiply(output_gradient, self.activation_prime(self.input))
