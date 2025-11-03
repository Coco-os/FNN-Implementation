import numpy as np
from typing import Type, Dict, Any
import abc
from layers import Layer, ComplexLayer
from network import Neuron

OPTIMIZERS: Dict[str, Type[abc.ABC]] = {}


def register_optimizer(name: str):
    """Decorator to register optimizer classes with a given name."""

    def decorator(cls: Type[abc.ABC]) -> Type[abc.ABC]:
        OPTIMIZERS[name.lower()] = cls
        return cls

    return decorator


def optimizer(name: str) -> Type[abc.ABC]:
    """Factory function to get an optimizer class (or instance) by name."""
    cls = OPTIMIZERS.get(name.lower())
    if cls is None:
        raise ValueError(
            f"Optimizer '{name}' not found. Registered: {list(OPTIMIZERS.keys())}"
        )
    return cls


def create_optimizer(name: str, **kwargs: Dict[str, Any]) -> abc.ABC:
    """Factory function to create an optimizer instance by name."""
    cls = optimizer(name)
    return cls(**kwargs)


class Optimizer(abc.ABC):
    @abc.abstractmethod
    def update(
        self,
        layer: Layer | ComplexLayer,
        gradients: np.ndarray | list[np.ndarray],
        learning_rate: float,
    ) -> None:
        pass


@register_optimizer("adam")
class AdamOptimizer(Optimizer):
    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m: Dict[int, np.ndarray | list[np.ndarray]] = {}
        self.v: Dict[int, np.ndarray | list[np.ndarray]] = {}
        self.t: Dict[int, int] = {}

    def update(
        self,
        layer: Layer | ComplexLayer,
        gradients: np.ndarray | list[np.ndarray],
        learning_rate: float,
    ) -> None:
        layer_id = id(layer)
        if layer_id not in self.m:
            if isinstance(layer, Layer):
                self.m[layer_id] = np.zeros_like(layer.W)
                self.v[layer_id] = np.zeros_like(layer.W)
            else:
                self.m[layer_id] = [
                    np.zeros_like(neuron.weights) for neuron in layer.neurons
                ]
                self.v[layer_id] = [
                    np.zeros_like(neuron.weights) for neuron in layer.neurons
                ]
            self.t[layer_id] = 0

        self.t[layer_id] += 1
        t = self.t[layer_id]

        if isinstance(layer, Layer):
            g = gradients
            m = self.m[layer_id]
            v = self.v[layer_id]

            m[:] = self.beta1 * m + (1 - self.beta1) * g
            v[:] = self.beta2 * v + (1 - self.beta2) * (g**2)

            m_hat = m / (1 - self.beta1**t)
            v_hat = v / (1 - self.beta2**t)

            layer.W -= learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        else:
            for i, neuron in enumerate(layer.neurons):
                g = gradients[i]
                m = self.m[layer_id][i]
                v = self.v[layer_id][i]

                m[:] = self.beta1 * m + (1 - self.beta1) * g
                v[:] = self.beta2 * v + (1 - self.beta2) * (g**2)

                m_hat = m / (1 - self.beta1**t)
                v_hat = v / (1 - self.beta2**t)
                neuron.weights = neuron.weights - learning_rate * m_hat / (
                    np.sqrt(v_hat) + self.epsilon
                )
