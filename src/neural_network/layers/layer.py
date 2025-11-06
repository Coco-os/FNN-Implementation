import numpy as np
from typing import Callable


# ------------------------------
# Activation Functions
# ------------------------------

class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self(x)
        return s * (1 - s)


# ------------------------------
# Weight Initialization
# ------------------------------

class WeightInitializers:
    """Collection of weight initialization methods."""

    @staticmethod
    def he(n_input: int, n_output: int) -> np.ndarray:
        """He initialization (for ReLU-type activations)."""
        return np.random.randn(n_output, n_input) * np.sqrt(2.0 / n_input)

    @staticmethod
    def glorot(n_input: int, n_output: int) -> np.ndarray:
        """Glorot/Xavier initialization (for sigmoid or tanh)."""
        limit = np.sqrt(6.0 / (n_input + n_output))
        return np.random.uniform(-limit, limit, (n_output, n_input))


# ------------------------------
# Layer Class
# ------------------------------

class Layer:
    """
    Represents a fully connected neural network layer.

    Parameters:
        input_size: number of input features.
        output_size: number of neurons in the layer.
        activation_function: activation function instance (must have .forward() and .derivative()).
        initializer: weight initialization function (e.g., WeightInitializers.he or WeightInitializers.glorot).
    """
    def __init__(self, input_size: int, output_size: int, activation_function, initializer):

        self.input_size = input_size
        self.output_size = output_size
        self.initializer = initializer
        self.activation_function = activation_function

        # Initialize weights and bias using the provided initializer
        self.W = initializer(input_size, output_size)
        self.b = np.zeros((output_size, 1))



        # Cache for forward/backward passes
        self.Z = None
        self.A = None
        self.X = None
        self.dW = None
        self.db = None

    # ------------------------------
    # Forward and Backward Pass
    # ------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.X = X  # cache input for backward
        self.Z = np.dot(self.W, X) + self.b
        self.A = self.activation_function.forward(self.Z)
        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.

        Parameters:
            dA: Gradient of the loss w.r.t. the output of this layer.

        Returns:
            dX: Gradient of the loss w.r.t. the input of this layer.
        """
        if self.X is None or self.Z is None:
            raise ValueError("Forward pass must be called before backward.")

        # Gradient w.r.t. Z
        dZ = dA * self.activation_function.derivative(self.Z)

        # Gradients w.r.t. weights and bias
        self.dW = np.dot(dZ, self.X.T) / self.X.shape[1]
        self.db = np.mean(dZ, axis=1, keepdims=True)

        # Gradient w.r.t. input to pass to previous layer
        dX = np.dot(self.W.T, dZ)
        return dX

    # ------------------------------
    # Utility Methods
    # ------------------------------

    def __repr__(self):
        init_name = self.initializer.__name__ if callable(self.initializer) else str(self.initializer)
        return f"Layer(input={self.input_size}, output={self.output_size}, activation={self.activation_function.__class__.__name__}, initializer={init_name})"