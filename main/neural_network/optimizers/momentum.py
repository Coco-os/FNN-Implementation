import numpy as np

class Momentum:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.v_weights = None
        self.v_bias = None

    def initialize(self, weights_shape, bias_shape):
        self.v_weights = np.zeros(weights_shape)
        self.v_bias = np.zeros(bias_shape)

    def update(self, weights, bias, weights_gradient, output_gradient):
        self.v_weights = self.momentum * self.v_weights - self.learning_rate * weights_gradient
        self.v_bias = self.momentum * self.v_bias - self.learning_rate * output_gradient

        weights += self.v_weights
        bias += self.v_bias

        return weights, bias
