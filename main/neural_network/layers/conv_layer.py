import numpy as np
from scipy import signal
from .layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape, kernel_size, depth, optim):
        super().__init__()
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.kernels_shape)
        self.bias = np.random.randn(*self.output_shape)
        self.optim = optim

        self.optim.initialize(self.weights.shape, self.bias.shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.bias)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.weights[i, j], mode="valid")

        return self.output

    def backward(self, output_gradient, optim):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                input_gradient[j] += signal.convolve2d(output_gradient[i], self.weights[i, j], "full")

        self.weights, self.bias = optim.update(
            self.weights, self.bias, kernels_gradient, output_gradient
        )

        return input_gradient
