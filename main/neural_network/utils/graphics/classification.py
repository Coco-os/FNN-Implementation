import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def display_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.title(title)
    plt.xlabel('Predicciones')
    plt.ylabel('Verdaderos')
    plt.show()


def display_images(x, y, num_images=10, pred=None):
    plt.figure(figsize=(10, 5))

    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap="gray")
        if pred is not None:
            plt.title(f"Label: {np.argmax(y[i])}\nPrediction: {pred[i]}")
        else:
            plt.title(f"Label: {np.argmax(y[i])}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()
