import numpy as np

class Adam:
    def __init__(self, learning_rate=0.001, gamma_v=0.9, gamma_s=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.gamma_v = gamma_v
        self.gamma_s = gamma_s
        self.epsilon = epsilon
        self.t = 0

        self.v_weights = None
        self.s_weights = None
        self.v_bias = None
        self.s_bias = None

    def initialize(self, weights_shape, bias_shape):
        self.v_weights = np.zeros(weights_shape)
        self.s_weights = np.zeros(weights_shape)
        self.v_bias = np.zeros(bias_shape)
        self.s_bias = np.zeros(bias_shape)

    def update(self, weights, bias, weights_gradient, output_gradient):
        self.t += 1

        self.v_weights = self.gamma_v * self.v_weights + (1 - self.gamma_v) * weights_gradient
        self.s_weights = self.gamma_s * self.s_weights + (1 - self.gamma_s) * (weights_gradient ** 2)

        self.v_bias = self.gamma_v * self.v_bias + (1 - self.gamma_v) * output_gradient
        self.s_bias = self.gamma_s * self.s_bias + (1 - self.gamma_s) * (output_gradient ** 2)

        v_weights_corrected = self.v_weights / (1 - self.gamma_v ** self.t)
        s_weights_corrected = self.s_weights / (1 - self.gamma_s ** self.t)

        v_bias_corrected = self.v_bias / (1 - self.gamma_v ** self.t)
        s_bias_corrected = self.s_bias / (1 - self.gamma_s ** self.t)

        weights -= self.learning_rate * v_weights_corrected / (np.sqrt(s_weights_corrected) + self.epsilon)
        bias -= self.learning_rate * v_bias_corrected / (np.sqrt(s_bias_corrected) + self.epsilon)

        return weights, bias
