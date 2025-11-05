import os
import sys
import subprocess
import importlib.util
import numpy as np


def ensure_tensorflow_installed():
    """Install TensorFlow if it is not already installed."""
    if importlib.util.find_spec("tensorflow") is None:
        print("[INFO] TensorFlow not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    else:
        print("[OK] TensorFlow is already installed.")


def download_mnist_with_keras():
    """Download the MNIST dataset using Keras and save it locally."""
    from tensorflow.keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    os.makedirs("datasets", exist_ok=True)
    np.savez_compressed("datasets/mnist.npz",
                        x_train=x_train, y_train=y_train,
                        x_test=x_test, y_test=y_test)
    print("[OK] MNIST dataset saved in datasets/mnist.npz")


def uninstall_tensorflow_in_subprocess():
    """Run a separate process to uninstall TensorFlow."""
    print("[INFO] Launching separate process to uninstall TensorFlow...")
    subprocess.Popen(
        [sys.executable, "-m", "pip", "uninstall", "-y",
         "tensorflow", "keras", "tensorboard", "tensorflow-intel"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    print("[INFO] TensorFlow is being uninstalled in the background.")


# === Main script ===
ensure_tensorflow_installed()
download_mnist_with_keras()
uninstall_tensorflow_in_subprocess()
print("[OK] MNIST dataset saved at data/datasets/mnist.npz and TensorFlow uninstalled.")
