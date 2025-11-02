from typing import Callable


class Neuron:
    def __init__(
        self,
        neuron_id: str,
        activation_function: Callable[[float], float],
        bias: float = 0.0,
        weights: list[float] | None = None,
    ):
        self.neuron_id = neuron_id
        self.activation_function = activation_function
        self.inputs: list["Neuron"] = []
        self.outputs: list["Neuron"] = []
        self.bias = bias
        self.weights = weights if weights is not None else []

    def add_input(self, neuron: "Neuron"):
        self.inputs.append(neuron)

    def add_output(self, neuron: "Neuron"):
        self.outputs.append(neuron)

    def activate(self, input_values: list[float]) -> float:
        z = sum(w * x for w, x in zip(self.weights, input_values)) + self.bias
        self.output_value = self.activation_function(z)
        return self.output_value
