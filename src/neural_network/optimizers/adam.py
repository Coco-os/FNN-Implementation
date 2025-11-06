import numpy as np
from .base import BaseOptimizer, Array, Shape

class Adam(BaseOptimizer):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.t = 0
        self.m_w: Array | None = None
        self.v_w: Array | None = None
        self.m_b: Array | None = None
        self.v_b: Array | None = None

    def initialize(self, weights_shape: Shape, bias_shape: Shape) -> None:
        self.m_w = np.zeros(weights_shape, dtype=float)
        self.v_w = np.zeros(weights_shape, dtype=float)
        self.m_b = np.zeros(bias_shape, dtype=float)
        self.v_b = np.zeros(bias_shape, dtype=float)
        self.t = 0

    def reset(self) -> None:
        self.m_w = self.v_w = self.m_b = self.v_b = None
        self.t = 0

    def update(self, weights: Array, bias: Array, dW: Array, dB: Array) -> tuple[Array, Array]:
        assert self.m_w is not None and self.v_w is not None
        assert self.m_b is not None and self.v_b is not None

        self.t += 1

        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * dW
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (dW ** 2)

        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * dB
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (dB ** 2)

        m_w_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_w_hat = self.v_w / (1 - self.beta2 ** self.t)
        m_b_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_b_hat = self.v_b / (1 - self.beta2 ** self.t)

        weights = weights - self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
        bias    = bias    - self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        return weights, bias
