import numpy as np
from typing import Callable, Optional
from numpy.typing import NDArray


class Layer:
    def __init__(
        self,
        layer_id: str,
        weights: NDArray[np.float64],
        bias: NDArray[np.float64],
        activation_function: Callable[
            [NDArray[np.float64]], NDArray[np.float64]
        ],
    ):
        self.layer_id = layer_id
        self.W = weights  # shape: (n_out, n_in)
        self.b = bias  # shape: (n_out,)
        self.activation = activation_function

    def forward(self, input_values: list[float]) -> NDArray[np.float64]:
        x = np.array(input_values, dtype=np.float64)
        z = np.dot(self.W, x) + self.b
        return self.activation(z)

    def backward(self, delta_out: np.ndarray) -> np.ndarray:
        self.grad_W = np.outer(delta_out, self.input)
        self.grad_b = delta_out
        delta_prev = np.dot(self.W.T, delta_out)
        return delta_prev


class ComplexLayer:
    def __init__(self, layer_id: str, neurons: list[Neuron]):
        self.layer_id = layer_id
        self.neurons = neurons

    def forward(
        self, input_values: Optional[list[float]] = None
    ) -> list[float]:
        outputs = []
        for neuron in self.neurons:
            if input_values is not None:
                output = neuron.forward(input_values)
            else:
                output = neuron.forward()
            outputs.append(output)
        return outputs

    def backward(self, delta_out: list[float]) -> list[float]:
        delta_prev_total = np.zeros(len(self.neurons[0].input_values))
        for neuron, delta in zip(self.neurons, delta_out):
            delta_prev = neuron.backward(delta)
            delta_prev_total += delta_prev
        return delta_prev_total

    def __repr__(self):
        return f"Layer({self.layer_id}, Neurons: {len(self.neurons)})"
