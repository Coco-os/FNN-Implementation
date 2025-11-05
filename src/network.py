import numpy as np
from typing import Callable, Optional, List
from numpy.typing import NDArray


class Neuron:
    def __init__(
        self,
        neuron_id: str,
        activation_function: Callable[[float], float],
        weights: Optional[List[float]] = None,
        inputs: Optional[List["Neuron"]] = None,
        bias: float = 0.0,
    ):
        self.neuron_id = neuron_id
        self.activation_function = activation_function
        self.weights = (
            np.array(weights, dtype=np.float64) if weights else np.array([])
        )
        self.inputs = inputs if inputs else []
        self.outputs: List["Neuron"] = []
        self.bias = bias

        for n in self.inputs:
            n.add_output(self)

    def add_input(self, neuron: "Neuron"):
        self.inputs.append(neuron)

    def add_output(self, neuron: "Neuron"):
        self.outputs.append(neuron)

    def forward(self, input_values: Optional[List[float]] = None) -> float:
        if self.inputs:
            input_values = np.array(
                [n.output_value for n in self.inputs], dtype=np.float64
            )
        elif input_values is None:
            raise ValueError("Input neuron requires explicit input_values.")

        self.input_values = np.array(input_values, dtype=np.float64)
        z = np.dot(self.weights, self.input_values) + self.bias
        self.output_value = self.activation_function(z)
        return self.output_value

    def backward(self, gradient: float) -> np.ndarray:
        """Returns gradient w.r.t inputs (delta for previous layer)"""
        self.grad_weights = gradient * self.input_values
        self.grad_bias = gradient
        delta_prev = self.weights * gradient
        return delta_prev

    def __repr__(self):
        return f"Neuron({self.neuron_id})"


class ComplexLayer:
    def __init__(self, layer_id: str, neurons: List[Neuron]):
        self.layer_id = layer_id
        self.neurons = neurons

    def forward(self, input_values: Optional[List[float]] = None) -> np.ndarray:
        outputs = []
        for neuron in self.neurons:
            if input_values is not None:
                out = neuron.forward(input_values)
            else:
                out = neuron.forward()
            outputs.append(out)
        return np.array(outputs)

    def backward(self, delta_out: np.ndarray) -> np.ndarray:
        delta_prev_total = np.zeros(len(self.neurons[0].input_values))
        for neuron, delta in zip(self.neurons, delta_out):
            delta_prev_total += neuron.backward(delta)
        return delta_prev_total

    def __repr__(self):
        return f"Layer({self.layer_id}, Neurons: {len(self.neurons)})"


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
        self.W = weights
        self.b = bias
        self.activation = activation_function

    def forward(self, input_values: np.ndarray) -> np.ndarray:
        self.input = np.array(input_values, dtype=np.float64)
        self.output = self.activation(np.dot(self.W, self.input) + self.b)
        return self.output

    def backward(self, delta_out: np.ndarray) -> np.ndarray:
        self.grad_W = np.outer(delta_out, self.input)
        self.grad_b = delta_out
        delta_prev = np.dot(self.W.T, delta_out)
        return delta_prev

    def __repr__(self):
        return f"Layer({self.layer_id}, Weights shape: {self.W.shape})"


class FeedforwardNetwork:
    def __init__(self, layers: List[ComplexLayer] | List[Layer]):
        self.layers = layers

    def forward(self, input_values: np.ndarray) -> np.ndarray:
        current_values = np.array(input_values, dtype=np.float64)
        for layer in self.layers:
            current_values = layer.forward(current_values)
        return current_values

    def backward(self, grad_output: np.ndarray) -> None:
        delta = grad_output
        for layer in reversed(self.layers):
            delta = layer.backward(delta)

    def __repr__(self):
        return f"FeedforwardNetwork(Layers: {len(self.layers)})"
