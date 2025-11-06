import numpy as np
from IPython.core.display_functions import clear_output

from src.neural_network.layers.network_layer import FullyConnectedLayer
from src.neural_network.optimizers.initialize import initialize_optimizer


def predict(network, x):
    y_pred = x
    for layer in network:
        y_pred = layer.forward(y_pred)
    return y_pred

def progress_bar(iteration, total, length=20):
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    return f"[{bar}]"


def train(network, loss, x_train, y_train, x_val=None, y_val=None, epochs=100):
    history = []

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            output = predict(network, x)

            error += loss(y, output)

            grad = clip_gradient(loss.derivative(y, output))

            for layer in reversed(network):
                grad = clip_gradient(layer.backward(grad, layer.optimizer))
                if np.any(np.isinf(grad)):
                    print("Gradient is inf")

        error = error / len(x_train)
        history.append(error)

        val_accuracy_message = ""

        if x_val is not None:
            val_accuracy = validation_accuracy(network, x_val, y_val)
            val_accuracy_message = f", Validation accuracy: {round(val_accuracy * 100, 2)}%"

        if e % 10 == 0:
            bar = progress_bar(e, epochs)
            print(f"{bar} Epoch: {e}, Loss: {error}{val_accuracy_message}")

        clear_output(wait=True)

    print(f"Final Loss = {error}")
    return history


def test_nn(nn, x_test, y_test, loss=None):
    correct_predictions = 0
    total_predictions = len(x_test)

    y_pred = []
    y_true = []

    for x, y in zip(x_test, y_test):
        output = predict(nn, x)
        predicted_class = np.argmax(output)
        true_class = np.argmax(y)
        y_true.append(true_class)
        y_pred.append(predicted_class)

        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions

    loss_message = ""

    if loss is not None:
        error = test_loss(nn, loss, x_test, y_test)
        loss_message = f'Loss: {error}, '
    print(f'{loss_message}Accuracy: {round(accuracy * 100, 2)}%')
    return y_true, y_pred


def test_loss(nn, loss, x_test, y_test):
    error = 0
    for x, y in zip(x_test, y_test):
        output = predict(nn, x)

        error += loss(y, output)
    error /= len(x_test)
    print(f"Loss in test = {error}")
    return error


def validation_accuracy(nn, x_val, y_val):
    correct_predictions = 0
    total_predictions = len(x_val)

    for x, y in zip(x_val, y_val):
        output = predict(nn, x)
        predicted_class = np.argmax(output)
        true_class = np.argmax(y)

        if predicted_class == true_class:
            correct_predictions += 1

    return correct_predictions / total_predictions


def clip_gradient(grad, threshold=3.0):
    norm = np.linalg.norm(grad)
    if norm > threshold:
        grad = grad * (threshold / norm)
    return grad


def create_nn(neurons_layer, activation_function, optimizers):
    nn = []
    for i in range(len(neurons_layer) - 1):
        nn.append(FullyConnectedLayer(neurons_layer[i], neurons_layer[i + 1], optimizers[i]))
        if i < len(activation_function):
            nn.append(activation_function[i])

    initialize_optimizer(nn, optimizers)
    return nn
