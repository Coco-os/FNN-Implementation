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

# Layer and ComplexLayer Implementation

```

# Implementation of fully connected layers for a feedforward neural network using both **matrix-based** and **neuron-based** approaches.

---

## Classes

### Layer

The `Layer` class represents a traditional fully connected neural network layer.
It performs a linear transformation (weighted sum + bias) on its inputs and applies an activation function to produce outputs.

---

### Initialization Parameters

| Parameter             | Type                                                   | Description                                        |
| --------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| `layer_id`            | `str`                                                  | Unique identifier for the layer.                   |
| `weights`             | `NDArray[np.float64]`                                  | Weight matrix of shape `(n_out, n_in)`.            |
| `bias`                | `NDArray[np.float64]`                                  | Bias vector of shape `(n_out,)`.                   |
| `activation_function` | `Callable[[NDArray[np.float64]], NDArray[np.float64]]` | Activation function applied to the layer’s output. |

---

### Attributes

| Attribute    | Description                                            |
| ------------ | ------------------------------------------------------ |
| `layer_id`   | Unique string identifier for the layer.                |
| `W`          | Weight matrix connecting inputs to outputs.            |
| `b`          | Bias vector added to the weighted sum.                 |
| `activation` | Activation function applied elementwise to the output. |

---

### Methods

| Method                  | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| `forward(input_values)` | Computes the layer’s output by applying `activation(Wx + b)` to the input vector. |

---

### Example

```python
import numpy as np

# Activation function

relu = lambda x: np.maximum(0, x)

# Define weights and biases

weights = np.array([[0.5, -0.2], [0.3, 0.8]])
bias = np.array([0.1, -0.1])

# Create a layer

layer = Layer("L1", weights=weights, bias=bias, activation_function=relu)

# Forward pass

output = layer.forward([1.0, 2.0])
print(output)
```

---

### ComplexLayer

The `ComplexLayer` class represents a layer composed of individual `Neuron` instances.
This allows flexible neuron-level architectures while maintaining a layer abstraction.

---

### Initialization Parameters

| Parameter  | Type           | Description                              |
| ---------- | -------------- | ---------------------------------------- |
| `layer_id` | `str`          | Unique identifier for the layer.         |
| `neurons`  | `list[Neuron]` | List of neurons contained in this layer. |

---

### Attributes

| Attribute  | Description                            |
| ---------- | -------------------------------------- |
| `layer_id` | Layer identifier                       |
| `neurons`  | List of `Neuron` objects in this layer |

---

### Methods

| Method                  | Description                                                                                  |
| ----------------------- | -------------------------------------------------------------------------------------------- |
| `forward(input_values)` | Computes the output of all neurons in the layer, optionally using the provided input vector. |
| `__repr__()`            | Returns a string representation of the layer, including its ID and number of neurons.        |

---

### Example

```python
# Assume Neuron class is already imported

# Create input neurons

input1 = Neuron(
    "I1",
    activation_function=lambda x: x,
    weights=[])
input2 = Neuron(
    "I2",
    activation_function=lambda x: 2x,
    weights=[])

# Create hidden neurons

hidden1 = Neuron(
    "H1",
    activation_function=lambda x: 1/(1+np.exp(-x)),
    inputs=[input1, input2],
    weights=[0.5, -0.3],
    bias=0.1)
hidden2 = Neuron(
    "H2",
    activation_function=lambda x: np.tanh(x),
    inputs=[input1, input2],
    weights=[-0.8, 0.7],
    bias=-0.2
)
# Create complex layer

complex_layer = ComplexLayer("HL1", neurons=[hidden1, hidden2])

# Forward pass

output = complex_layer.forward([1.0, 2.0])
print(output)
```
