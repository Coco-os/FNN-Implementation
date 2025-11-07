import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def display_confusion_matrix(y_true, y_pred, labels, title):
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="pink_r", colorbar=False)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Predicciones", fontsize=12)
    ax.set_ylabel("Verdaderos", fontsize=12)

    plt.grid(False)
    plt.tight_layout()
    plt.show()


def display_images(x, y, num_images=10, pred=None):
    plt.figure(figsize=(10, 5))

    for i in range(num_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x[i].reshape(28, 28), cmap="gray")
        if pred is not None:
            plt.title(f"Real: {np.argmax(y[i])}\nPred: {pred[i]}", fontsize=10, color="#E26BAF")
        else:
            plt.title(f"Real: {np.argmax(y[i])}", fontsize=10, color="#E26BAF")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
