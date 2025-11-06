import numpy as np


class CategoricalCrossEntropyLoss:

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        """
        Compute categorical cross-entropy loss.

        y_true: one-hot encoded true labels, shape (num_classes,)
        y_pred: predicted probabilities (after softmax), shape (num_classes,)
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        """
        Derivative of categorical cross-entropy w.r.t. the input of softmax.
        """
        return y_pred - y_true
