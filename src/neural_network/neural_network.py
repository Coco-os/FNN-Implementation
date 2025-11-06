import numpy as np

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


