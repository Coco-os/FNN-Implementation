import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.sum(y_true * np.log(y_pred))

    @staticmethod
    def derivative(y_true, y_pred):
        return y_pred - y_true
