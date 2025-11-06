
def clone_optimizer(opt):
    cls = opt.__class__
    kwargs = {}

    for name in ("learning_rate", "lr", "beta1", "beta2", "epsilon", "momentum"):
        if hasattr(opt, name):
            kwargs[name] = getattr(opt, name)

    try:
        return cls(**kwargs)
    except TypeError:
        return cls()

def initialize_optimizer(nn, optimizers):
    i = 0

    for layer in nn:
        has_dense_params = hasattr(layer, "weights") and hasattr(layer, "bias")
        has_conv_params  = hasattr(layer, "kernels") and hasattr(layer, "biases")

        if has_dense_params or has_conv_params:
            if i >= len(optimizers):
                raise ValueError(
                    f"No hay suficientes optimizadores: se necesitan al menos {i+1}, "
                    f"pero se recibieron {len(optimizers)}."
                )

            layer.optimizer = clone_optimizer(optimizers[i])

            if has_dense_params:
                layer.optimizer.initialize(layer.weights.shape, layer.bias.shape)
            else:
                layer.optimizer.initialize(layer.kernels.shape, layer.biases.shape)

            i += 1
        else:
            layer.optimizer = None
