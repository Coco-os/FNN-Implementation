from .base import BaseOptimizer, Array, Shape

class SGD(BaseOptimizer):
    def __init__(self, learning_rate: float = 1e-2) -> None:
        super().__init__(learning_rate)

    def initialize(self, weights_shape: Shape, bias_shape: Shape) -> None:
        pass

    def update(self, weights: Array, bias: Array, dW: Array, dB: Array) -> tuple[Array, Array]:
        weights = weights - self.learning_rate * dW
        bias    = bias    - self.learning_rate * dB
        return weights, bias
