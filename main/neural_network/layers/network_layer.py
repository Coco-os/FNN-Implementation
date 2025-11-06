import numpy as np
from .layer import Layer

class FullyConnectedLayer(Layer):
    def __init__(self, input_size, output_size, optimizer):
        super().__init__()
        self.optimizer = optimizer
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(-1, 1, (output_size, input_size))
        self.bias = np.random.uniform(-1, 1, (output_size, 1))

        self.optimizer.initialize(self.weights.shape, self.bias.shape)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, optim):
        weights_gradient = np.dot(output_gradient, self.input.T)

        self.weights, self.bias = optim.update(
            self.weights, self.bias, weights_gradient, output_gradient
        )

        return np.dot(self.weights.T, output_gradient)

