# FNN-Implementation

Implementation of a feedforward neural network

## Classes

### Neuron

Our feed forward network works using this class as a basis. It has 5 initialization parameters, and 6 attributes:
Neuron_id: A string the servers as identifier for the Neuron object.
Activation_function: A callable method that represent the activation
function the Neuron object will use
Weights: A list of weights, asigned to the inputs.
Input: A list of Neurons that connect to the Neuron upstream.
Outputs: A list of Neurons that connect to the Neuron downstream.
Bias: The ammount of bias passed to the Neuron.

It also has 6 methods:

Add Input: It lets us add input neurons after class creation for flexible architecture.
Add Output: It lets us add output neurons after class creation for flexible architecture.
Update Weights:It lets us update weights after declaration. Necesary for the feedforward architecture.
Update bias: It lets us update bias after declaration. Necesary for the feedforward architecture.
Forward: Calculates and returns the output of the Neuron.
**repr**: Returns the identity of the neuron
