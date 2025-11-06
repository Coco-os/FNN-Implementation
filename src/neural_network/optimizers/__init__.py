from .sgd import SGD
from .adam import Adam
from .rmsprop import RMSProp
from .adagrad import Adagrad
from .momentum import Momentum

__all__ = ["SGD", "Adam", "RMSProp", "Adagrad", "Momentum"]

def initialize_for_network(network, optimizers):
    layers = getattr(network, "layers", network)  # acepta red iterable
    idx = 0
    for layer in layers:
        if hasattr(layer, "weights") and hasattr(layer, "bias"):
            optimizers[idx].initialize(layer.weights.shape, layer.bias.shape)
            idx += 1
    return optimizers
