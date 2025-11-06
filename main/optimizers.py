import numpy as np

from main import FullyConnectedLayer


class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def initialize(self, weights_shape, bias_shape):
        pass

    def update(self, weights, bias, weights_gradient, output_gradient):
        weights -= self.learning_rate * weights_gradient
        bias -= self.learning_rate * output_gradient
        return weights, bias


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


class Adagrad:
    def __init__(self, learning_rate=0.01, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.s_weights = None
        self.s_bias = None

    def initialize(self, weights_shape, bias_shape):
        self.s_weights = np.zeros(weights_shape)
        self.s_bias = np.zeros(bias_shape)

    def update(self, weights, bias, weights_gradient, output_gradient):
        self.s_weights += weights_gradient ** 2
        self.s_bias += output_gradient ** 2

        weights -= self.learning_rate * weights_gradient / (np.sqrt(self.s_weights) + self.epsilon)
        bias -= self.learning_rate * output_gradient / (np.sqrt(self.s_bias) + self.epsilon)

        return weights, bias


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


def initialize_optimizer(nn, optimizers):
    i = 0

    for layer in nn:
        if isinstance(layer, FullyConnectedLayer):
            optimizers[i].initialize(layer.weights.shape, layer.bias.shape)
            i += 1
