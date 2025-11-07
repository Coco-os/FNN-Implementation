import numpy as np

class MSELoss:

    def __init__(self):
        pass

    def __call__(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    @staticmethod
    def derivative(y_true, y_pred):
        return 2 * (y_pred - y_true) / np.size(y_true)
