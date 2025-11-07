import numpy as np
from main.neural_network.optimizers.interface.base import BaseOptimizer, Array, Shape

class Adagrad(BaseOptimizer):
    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8) -> None:
        super().__init__(learning_rate)
        self.epsilon = float(epsilon)

        self.s_weights: Array | None = None
        self.s_bias: Array | None = None

    def initialize(self, weights_shape: Shape, bias_shape: Shape) -> None:
        self.s_weights = np.zeros(weights_shape, dtype=float)
        self.s_bias = np.zeros(bias_shape, dtype=float)

    def reset(self) -> None:
        self.s_weights = None
        self.s_bias = None

    def update(
        self,
        weights: Array,
        bias: Array,
        dW: Array,
        dB: Array
    ) -> tuple[Array, Array]:

        assert self.s_weights is not None and self.s_bias is not None

        self.s_weights += dW ** 2
        self.s_bias += dB ** 2

        weights = weights - self.learning_rate * dW / (np.sqrt(self.s_weights) + self.epsilon)
        bias    = bias    - self.learning_rate * dB / (np.sqrt(self.s_bias) + self.epsilon)

        return weights, bias
