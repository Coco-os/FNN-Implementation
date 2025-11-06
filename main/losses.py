import numpy as np


class LossFunction:
    """
    Collection of loss functions and their derivatives
    for training neural networks.
    """

    # ------------------------------
    # Mean Squared Error (MSE)
    # ------------------------------
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute Mean Squared Error (MSE) loss.
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mse_derivative(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Derivative of MSE with respect to predictions (y_pred).
        """
        return 2 * (y_pred - y_true) / y_true.size

    # ------------------------------
    # Categorical Cross-Entropy (CCE)
    # ------------------------------
    @staticmethod
    def categorical_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> float:
        """
        Compute the categorical cross-entropy loss.

        y_true: one-hot encoded true labels (shape: [batch_size, num_classes])
        y_pred: predicted probabilities (after softmax)
        """
        # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    @staticmethod
    def categorical_cross_entropy_derivative(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
        """
        Derivative of categorical cross-entropy with respect to y_pred.
        """
        y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
        return - (y_true / y_pred) / y_true.shape[0]
