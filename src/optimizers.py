import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.001):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer, dW, db):
        """Update weights and biases for a given layer."""
        pass


class Adam(Optimizer):
    """Adam optimizer (Adaptive Moment Estimation)."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m_W = {}
        self.v_W = {}
        self.m_b = {}
        self.v_b = {}
        self.t = {}

    def update(self, layer, dW, db):
        layer_id = id(layer)

        # Initialize moment estimates
        if layer_id not in self.m_W:
            self.m_W[layer_id] = np.zeros_like(dW)
            self.v_W[layer_id] = np.zeros_like(dW)
            self.m_b[layer_id] = np.zeros_like(db)
            self.v_b[layer_id] = np.zeros_like(db)
            self.t[layer_id] = 0

        self.t[layer_id] += 1
        t = self.t[layer_id]

        # Update biased first and second moment estimates
        self.m_W[layer_id] = self.beta1 * self.m_W[layer_id] + (1 - self.beta1) * dW
        self.v_W[layer_id] = self.beta2 * self.v_W[layer_id] + (1 - self.beta2) * (dW ** 2)
        self.m_b[layer_id] = self.beta1 * self.m_b[layer_id] + (1 - self.beta1) * db
        self.v_b[layer_id] = self.beta2 * self.v_b[layer_id] + (1 - self.beta2) * (db ** 2)

        # Bias correction
        m_W_hat = self.m_W[layer_id] / (1 - self.beta1 ** t)
        v_W_hat = self.v_W[layer_id] / (1 - self.beta2 ** t)
        m_b_hat = self.m_b[layer_id] / (1 - self.beta1 ** t)
        v_b_hat = self.v_b[layer_id] / (1 - self.beta2 ** t)

        # Parameter updates
        layer.W -= self.learning_rate * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
        layer.b -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)


class SGD_Momentum(Optimizer):
    """Stochastic Gradient Descent with Momentum."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.v_W = {}
        self.v_b = {}

    def update(self, layer, dW, db):
        layer_id = id(layer)

        if layer_id not in self.v_W:
            self.v_W[layer_id] = np.zeros_like(dW)
            self.v_b[layer_id] = np.zeros_like(db)

        # Momentum update
        self.v_W[layer_id] = self.momentum * self.v_W[layer_id] - self.learning_rate * dW
        self.v_b[layer_id] = self.momentum * self.v_b[layer_id] - self.learning_rate * db

        # Update parameters
        layer.W += self.v_W[layer_id]
        layer.b += self.v_b[layer_id]


class RMSProp(Optimizer):
    """RMSProp optimizer."""

    def __init__(self, learning_rate: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(learning_rate)
        self.beta = beta
        self.epsilon = epsilon
        self.s_W = {}
        self.s_b = {}

    def update(self, layer, dW, db):
        layer_id = id(layer)

        if layer_id not in self.s_W:
            self.s_W[layer_id] = np.zeros_like(dW)
            self.s_b[layer_id] = np.zeros_like(db)

        # Compute moving average of squared gradients
        self.s_W[layer_id] = self.beta * self.s_W[layer_id] + (1 - self.beta) * (dW ** 2)
        self.s_b[layer_id] = self.beta * self.s_b[layer_id] + (1 - self.beta) * (db ** 2)

        # Update parameters
        layer.W -= self.learning_rate * dW / (np.sqrt(self.s_W[layer_id]) + self.epsilon)
        layer.b -= self.learning_rate * db / (np.sqrt(self.s_b[layer_id]) + self.epsilon)

