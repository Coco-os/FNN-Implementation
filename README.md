# FNN-Implementation

Implementation of a feedforward neural network

## Classes

### Neuron

Our feed forward network works using this class as a basis. It has 5 initialization parameters, and 6 attributes: \n
Neuron_id: A string the servers as identifier for the Neuron object.\n
Activation_function: A callable method that represent the activation
function the Neuron object will use\n
Weights: A list of weights, asigned t the inputs.\n
Input: A list of Neurons that connect to the Neuron upstream.\n
Outputs: A list of Neurons that connect to the Neuron downstream.\n
Bias: The ammount of bias passed to the Neuron.\n

It also has 6 methods:

Add Input: It lets us add input neurons after class creation for flexible architecture.\n
Add Output: It lets us add output neurons after class creation for flexible architecture.\n
Update Weights:It lets us update weights after declaration. Necesary for the feedforward architecture.\n
Update bias: It lets us update bias after declaration. Necesary for the feedforward architecture.\n
Forward: Calculates and returns the output of the Neuron.\n
\_\_repr\_\_: Returns the identity of the neuron\n
