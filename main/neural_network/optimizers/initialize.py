from main.neural_network.layers.network_layer import FullyConnectedLayer

def initialize_optimizer(nn, optimizers):
    i = 0

    for layer in nn:
        if isinstance(layer, FullyConnectedLayer):
            optimizers[i].initialize(layer.weights.shape, layer.bias.shape)
            i += 1
