# 🧠 Feedforward Neural Network (FNN)

Implementation of a modular **Feedforward Neural Network (FNN)** built from individual neurons, supporting both **matrix-based** and **neuron-based** layers.  
Includes full forward propagation, partial backpropagation support, optimizers, and dataset utilities.

---

## Overview

This project demonstrates a **from-scratch** neural network built using object-oriented Python, emphasizing:

- Modular design (`Neuron`, `Layer`, `ComplexLayer`, `FeedforwardNetwork`)
- Extendable backpropagation logic
- Optimizer support (Adam)
- Mini-batch training
- Dataset management and splitting

---

## Classes

---

### Neuron

The `Neuron` class represents the fundamental processing unit of the network.  
Each neuron computes a weighted sum of its inputs, adds a bias, applies an activation function, and produces an output value.

---

#### **Initialization Parameters**

| Parameter             | Type                       | Description                                                                    |
| --------------------- | -------------------------- | ------------------------------------------------------------------------------ |
| `neuron_id`           | `str`                      | Unique identifier for the neuron.                                              |
| `activation_function` | `Callable[[float], float]` | Activation function used to compute the neuron’s output.                       |
| `weights`             | `list[float]`              | List of weights assigned to each input.                                        |
| `inputs`              | `list[Neuron]` or `None`   | List of upstream neurons connected to this neuron. Optional for input neurons. |
| `bias`                | `float`                    | Bias term added to the weighted sum. Default is `0.0`.                         |

---

#### **Attributes**

| Attribute             | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `neuron_id`           | String identifier for the neuron instance.                  |
| `activation_function` | Callable function applied to the weighted sum.              |
| `weights`             | The current weights applied to each input connection.       |
| `inputs`              | Neurons that feed into this neuron (upstream).              |
| `outputs`             | Neurons that receive signals from this neuron (downstream). |
| `bias`                | The current bias value of the neuron.                       |

---

#### **Methods**

| Method                        | Description                                                                                              |
| ----------------------------- | -------------------------------------------------------------------------------------------------------- |
| `add_input(neuron)`           | Adds an input neuron after initialization.                                                               |
| `add_output(neuron)`          | Adds a downstream connection to another neuron.                                                          |
| `update_weights(new_weights)` | Updates the neuron’s weights (must match input length).                                                  |
| `update_bias(new_bias)`       | Updates the neuron’s bias.                                                                               |
| `forward(input_values=None)`  | Computes the neuron’s output by applying the activation function to the weighted sum of inputs and bias. |
| `backward(gradient)`          | Computes partial gradients of weights and bias during backpropagation.                                   |
| `__repr__()`                  | Returns a readable string representation of the neuron.                                                  |

---

#### **Example**

```python
    import math
    from neuron import Neuron

    sigmoid = lambda x: 1 / (1 + math.exp(-x))

    input_neuron = Neuron("I1", activation_function=lambda x: x, weights=[])
    hidden_neuron = Neuron("H1", activation_function=sigmoid, inputs=[input_neuron], weights=[0.8], bias=0.2)

    input_neuron.forward([1.0])
    output = hidden_neuron.forward()

    print(output)

```

---

## Layer and ComplexLayer

Implementation of fully connected layers for a feedforward neural network using both **matrix-based** (`Layer`) and **neuron-based** (`ComplexLayer`) approaches.

---

### Layer

The `Layer` class represents a fully connected neural layer that performs:

$$z = W x + b$$

followed by an activation function.

---

#### **Initialization Parameters**

| Parameter             | Type                                                   | Description                                        |
| --------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| `layer_id`            | `str`                                                  | Unique identifier for the layer.                   |
| `weights`             | `NDArray[np.float64]`                                  | Weight matrix of shape `(n_out, n_in)`.            |
| `bias`                | `NDArray[np.float64]`                                  | Bias vector of shape `(n_out,)`.                   |
| `activation_function` | `Callable[[NDArray[np.float64]], NDArray[np.float64]]` | Activation function applied to the layer’s output. |

---

#### **Attributes**

| Attribute    | Description                                            |
| ------------ | ------------------------------------------------------ |
| `layer_id`   | Unique string identifier for the layer.                |
| `W`          | Weight matrix connecting inputs to outputs.            |
| `b`          | Bias vector added to the weighted sum.                 |
| `activation` | Activation function applied elementwise to the output. |

---

#### **Methods**

| Method                  | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| `forward(input_values)` | Computes the layer’s output by applying `activation(Wx + b)` to the input vector. |
| `backward(delta_out)`   | Performs the backward pass and computes gradients for weights and biases.         |

---

#### **Example**

```python
    import numpy as np

    relu = lambda x: np.maximum(0, x)

    weights = np.array([[0.5, -0.2], [0.3, 0.8]])
    bias = np.array([0.1, -0.1])

    layer = Layer("L1", weights=weights, bias=bias, activation_function=relu)
    output = layer.forward([1.0, 2.0])
    print(output)
```

---

### ComplexLayer

The `ComplexLayer` class represents a layer composed of individual `Neuron` instances, offering a more granular neuron-based design.

---

#### **Initialization Parameters**

| Parameter  | Type           | Description                              |
| ---------- | -------------- | ---------------------------------------- |
| `layer_id` | `str`          | Unique identifier for the layer.         |
| `neurons`  | `list[Neuron]` | List of neurons contained in this layer. |

---

#### **Methods**

| Method                  | Description                                                                         |
| ----------------------- | ----------------------------------------------------------------------------------- |
| `forward(input_values)` | Computes the output of all neurons in the layer.                                    |
| `backward(delta_out)`   | Performs backward propagation through all neurons and returns cumulative gradients. |

---

#### **Example**

```python
    import numpy as np
    from neuron import Neuron
    from layers import ComplexLayer

    input1 = Neuron("I1", activation_function=lambda x: x, weights=[])
    input2 = Neuron("I2", activation_function=lambda x: x, weights=[])

    hidden1 = Neuron("H1", activation_function=lambda x: 1/(1+np.exp(-x)), inputs=[input1, input2], weights=[0.5, -0.3], bias=0.1)
    hidden2 = Neuron("H2", activation_function=lambda x: np.tanh(x), inputs=[input1, input2], weights=[-0.8, 0.7], bias=-0.2)

    complex_layer = ComplexLayer("HL1", neurons=[hidden1, hidden2])
    output = complex_layer.forward([1.0, 2.0])
    print(output)
```

---

## FeedforwardNetwork

The `FeedforwardNetwork` class manages sequential layer execution for both forward and backward passes.

| Method               | Description                                                        |
| -------------------- | ------------------------------------------------------------------ |
| `forward(inputs)`    | Propagates inputs through all layers and returns final outputs.    |
| `backward(grad_out)` | Performs backpropagation starting from the output layer backwards. |

---

#### **Example**

```python
    from network import FeedforwardNetwork
    from layers import Layer
    import numpy as np

    relu = lambda x: np.maximum(0, x)

    layer1 = Layer("L1", np.random.randn(4, 3), np.zeros(4), relu)
    layer2 = Layer("L2", np.random.randn(2, 4), np.zeros(2), relu)

    net = FeedforwardNetwork([layer1, layer2])
    output = net.forward([1.0, 0.5, -1.2])
    print(output)
```

---

## Optimizers

The project includes an **Adam optimizer** implementation with adaptive learning rates and bias correction.

| Parameter | Default | Description                                            |
| --------- | ------- | ------------------------------------------------------ |
| `beta1`   | 0.9     | Exponential decay rate for the first moment estimate.  |
| `beta2`   | 0.999   | Exponential decay rate for the second moment estimate. |
| `epsilon` | 1e-8    | Small constant for numerical stability.                |

### Example

```python
    optimizer = AdamOptimizer()
    optimizer.update(layer, gradients, grad_bias, learning_rate=0.001)
```

---

## Mini-Batch Training

Supports variable-sized **mini-batches** (e.g., 32, 64, 128) for training.  
Batches are processed sequentially, and if the dataset size is not divisible by the batch size, the final batch is handled automatically.

---

## Dataset Splitting

Utility functions allow partitioning data into **training**, **validation**, and **test** sets with customizable ratios and random seed control.

Example ratios:

- 70% training
- 15% validation
- 15% testing

Optional `random_seed` ensures reproducibility.
