import numpy as np
from typing import Callable, Optional
from numpy.typing import NDArray


class Neuron:
    def __init__(
        self,
        neuron_id: str,
        activation_function: Callable[[float], float],
        weights: list[float],
        inputs: Optional[list["Neuron"]] = None,
        bias: float = 0.0,
    ):
        self.neuron_id = neuron_id
        self.activation_function = activation_function
        self.weights = weights
        self.inputs = inputs if inputs is not None else []
        self.outputs: list["Neuron"] = []
        self.bias = bias

        for n in self.inputs:
            n.add_output(self)

    def add_input(self, neuron: "Neuron"):
        self.inputs.append(neuron)

    def add_output(self, neuron: "Neuron"):
        self.outputs.append(neuron)

    def update_weights(self, new_weights: list[float]):
        if len(new_weights) != len(self.inputs):
            raise ValueError(
                "Length of new_weights must match number of inputs."
            )
        self.weights = new_weights

    def update_bias(self, new_bias: float):
        self.bias = new_bias

    def forward(self, input_values: Optional[list[float]] = None) -> float:
        if self.inputs:
            input_values = [n.output_value for n in self.inputs]
        elif input_values is None:
            raise ValueError("Input neuron requires explicit input_values.")

        z = sum(w * x for w, x in zip(self.weights, input_values)) + self.bias
        self.output_value = self.activation_function(z)
        return self.output_value

    def __repr__(self):
        return f"Neuron({self.neuron_id})"


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

    def __repr__(self):
        return f"Layer({self.layer_id}, Neurons: {len(self.neurons)})"
