import matplotlib.pyplot as plt
import numpy as np

def dispersion_graph(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5, label='Test')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs. Predicted Values')
    plt.legend()
    plt.show()


def learning_curve(loss_during_train):
    epochs = np.arange(len(loss_during_train))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss_during_train, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()
