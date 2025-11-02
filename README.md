# FNN Implementation

Implementation of a simple Feedforward Neural Network (FNN) using a modular `Neuron` class.

---

## Classes

### Neuron

The `Neuron` class serves as the fundamental building block of the feedforward neural network.  
Each neuron performs a weighted sum of its inputs, adds a bias term, applies an activation function, and produces an output value.

---

### Initialization Parameters

| Parameter             | Type                       | Description                                                                    |
| --------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| `neuron_id`           | `str`                      | Unique identifier for the neuron.                                              |
| `activation_function` | `Callable[[float], float]` | Activation function used to compute the neuron’s output.                       |
| `weights`             | `list[float]`              | List of weights assigned to each input.                                        |
| `inputs`              | `list[Neuron]` or `None`   | List of upstream neurons connected to this neuron. Optional for input neurons. |
| `bias`                | `float`                    | Bias term added to the weighted sum. Default is `0.0`.                         |

---

### Attributes

| Attribute             | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `neuron_id`           | String identifier for the neuron instance.                  |
| `activation_function` | Callable function applied to the weighted sum.              |
| `weights`             | The current weights applied to each input connection.       |
| `inputs`              | Neurons that feed into this neuron (upstream).              |
| `outputs`             | Neurons that receive signals from this neuron (downstream). |
| `bias`                | The current bias value of the neuron.                       |

---

### Methods

| Method                        | Description                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `add_input(neuron)`           | Adds an input neuron after initialization, enabling flexible network architecture.                       |
| `add_output(neuron)`          | Adds a downstream connection from this neuron to another neuron.                                         |
| `update_weights(new_weights)` | Updates the neuron’s weights. The new list length must match the number of inputs.                       |
| `update_bias(new_bias)`       | Updates the bias term.                                                                                   |
| `forward(input_values=None)`  | Computes the neuron’s output by applying the activation function to the weighted sum of inputs and bias. |
| `__repr__()`                  | Returns a string representation of the neuron (its ID).                                                  |

---

### Example

```python
import math
from neuron import Neuron

# Define an activation function
sigmoid = lambda x: 1 / (1 + math.exp(-x))

# Create an input neuron (no upstream neurons)
input_neuron = Neuron("I1", activation_function=lambda x: x, weights=[])

# Create a hidden neuron connected to the input neuron
hidden_neuron = Neuron("H1", activation_function=sigmoid, inputs=[input_neuron], weights=[0.8], bias=0.2)

# Forward pass
input_neuron.forward([1.0])
output = hidden_neuron.forward()

print(output)
```
