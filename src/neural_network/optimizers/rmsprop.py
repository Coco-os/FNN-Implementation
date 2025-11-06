import numpy as np

class RMSProp:
    def __init__(self, learning_rate=0.001, gamma=0.9, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.s_weights = None
        self.s_bias = None

    def initialize(self, weights_shape, bias_shape):
        self.s_weights = np.zeros(weights_shape)
        self.s_bias = np.zeros(bias_shape)

    def update(self, weights, bias, weights_gradient, output_gradient):
        self.s_weights = self.gamma * self.s_weights + (1 - self.gamma) * (weights_gradient ** 2)
        self.s_bias = self.gamma * self.s_bias + (1 - self.gamma) * (output_gradient ** 2)

        weights -= self.learning_rate * weights_gradient / (np.sqrt(self.s_weights) + self.epsilon)
        bias -= self.learning_rate * output_gradient / (np.sqrt(self.s_bias) + self.epsilon)

        return weights, bias