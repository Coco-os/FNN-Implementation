
class SGD:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def initialize(self, weights_shape, bias_shape):
        pass

    def update(self, weights, bias, weights_gradient, output_gradient):
        weights -= self.learning_rate * weights_gradient
        bias -= self.learning_rate * output_gradient
        return weights, bias
