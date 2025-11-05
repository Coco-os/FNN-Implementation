import matplotlib.pyplot as plt
from src.layer import Layer
import numpy as np
from utils import create_mini_batches

class NeuralNetwork:
    def __init__(
            self,
            input_size: int,
            layers_num_neurons: list[int],
            layers_activation_functions: list,
            layers_initializers,
            optimizer,
            loss_function
    ):
        if len(layers_num_neurons) != len(layers_activation_functions):
            raise ValueError("layers_config and activations must have the same length")

        if not isinstance(layers_initializers, list):
            initializers = [layers_initializers] * len(layers_num_neurons)
        else:
            if len(layers_initializers) != len(layers_num_neurons):
                raise ValueError("If passing a list of initializers, it must match layers_config length")
            initializers = layers_initializers

        self.layers = []
        self.optimizer = optimizer
        self.loss_function = loss_function
        prev_size = input_size

        for neurons, act_instance, init in zip(layers_num_neurons, layers_activation_functions, initializers):
            self.layers.append(Layer(
                input_size=prev_size,
                output_size=neurons,
                activation_function=act_instance,
                initializer=init
            ))
            prev_size = neurons

        # Store losses for plotting
        self.loss_history = []

    def __repr__(self) -> str:
        init_name = self.loss_function.__name__ if callable(self.loss_function) else str(self.loss_function)
        desc = "NeuralNetwork(\n"
        for i, layer in enumerate(self.layers):
            desc += f"  Layer {i + 1}: {repr(layer)}\n"
        desc += f"  Optimizer: {self.optimizer.__class__.__name__}\n"
        desc += f"  Loss Function: {init_name}\n"
        desc += ")"
        return desc

    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Backpropagation using the derivative of the loss function.
        """
        # Compute gradient of loss w.r.t output
        dA = self.loss_function.derivative(y_true, y_pred)

        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            dA = layer.backward(dA)
            self.optimizer.update(layer, layer.dW, layer.db)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int):
        """
        Train the network using mini-batches and the specified loss function.
        """
        try:
            self.loss_history = []

            for epoch in range(epochs):
                total_loss = 0.0
                batches = list(create_mini_batches(X_train, y_train, batch_size=batch_size))
                for X_batch, y_batch in batches:
                    y_pred = self.forward(X_batch)
                    # Compute loss using the loss function
                    loss = self.loss_function.forward(y_batch, y_pred)
                    total_loss += loss

                    # Backpropagation
                    self.backward(y_batch, y_pred)

                avg_loss = total_loss / len(batches)
                self.loss_history.append(avg_loss)

                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f}")
        except Exception as e:
            print(f"[ERROR] {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=0)

    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == np.argmax(y_true, axis=0))
        print(f"Accuracy: {accuracy * 100:.2f}%")
        return float(accuracy)

    def plot_losses(self):
        """Plot training loss per epoch."""
        if not self.loss_history:
            print("[INFO] No loss history found. Train the model first.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(self.loss_history) + 1), self.loss_history, marker='o')
        plt.title("Training Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.grid(True)
        plt.show()
