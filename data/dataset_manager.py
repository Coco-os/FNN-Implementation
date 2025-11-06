import os
from main.utils import train_val_test_split
import numpy as np
from sklearn.datasets import load_iris
import sys
import subprocess
import importlib.util
from tensorflow.keras.datasets import mnist

def download_iris_dataset():
    """Download the IRIS dataset and save it locally as NumPy arrays."""
    print("[INFO] Downloading the IRIS dataset...")

    # Load the dataset from scikit-learn
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names

    # Ensure the directory exists
    out_dir = os.path.join("..", "data", "datasets")
    os.makedirs(out_dir, exist_ok=True)

    # Full output path
    out_path = os.path.join(out_dir, "iris.npz")

    np.savez_compressed(
        out_path,
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
    )

    print(f"[OK] IRIS dataset saved at {out_path}")


def ensure_tensorflow_installed():
    """Install TensorFlow if it is not already installed."""
    if importlib.util.find_spec("tensorflow") is None:
        print("[INFO] TensorFlow not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    else:
        print("[OK] TensorFlow is already installed.")


def download_mnist_with_keras():
    """Download the MNIST dataset using Keras and save it locally."""
    print("[INFO] Downloading the MNIST dataset...")

    # Load MNIST from Keras
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Define the correct output directory and create it if missing
    out_dir = os.path.join("..", "data", "datasets")
    os.makedirs(out_dir, exist_ok=True)

    # Full output path
    out_path = os.path.join(out_dir, "mnist.npz")

    # Save compressed dataset
    np.savez_compressed(
        out_path,
        x_train=x_train, y_train=y_train,
        x_test=x_test, y_test=y_test
    )

    print(f"[OK] MNIST dataset saved at {out_path}")


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


def download_mnist_dataset():
    ensure_tensorflow_installed()
    download_mnist_with_keras()
    uninstall_tensorflow_in_subprocess()
    print("[OK] MNIST dataset saved at data/datasets/mnist.npz and TensorFlow uninstalled.")


def preprocess_dataset(X_data, y_data):
    # Normalize if it's image data (heuristic)
    if X_data.max() > 1:
        X_data = X_data.astype(np.float32) / 255.0

    # Flatten images if necessary (e.g., MNIST 28x28 -> 784)
    if len(X_data.shape) > 2:
        X_data = X_data.reshape(X_data.shape[0], -1)

    return X_data, y_data


def split_and_preprocess_dataset(file_path: str, random_seed: int):
    full_dataset = np.load(file_path)
    keys = list(full_dataset.keys())

    # Try to detect which dataset structure we're dealing with
    if "X" in keys and "y" in keys:
        X_data = full_dataset["X"]
        y_data = full_dataset["y"]

    elif {"x_train", "y_train", "x_test", "y_test"}.issubset(keys):
        # Combine train and test into one large set before splitting
        X_data = np.concatenate([full_dataset["x_train"], full_dataset["x_test"]], axis=0)
        y_data = np.concatenate([full_dataset["y_train"], full_dataset["y_test"]], axis=0)

    else:
        raise KeyError(
            f"Could not find suitable keys for X/y in {file_path}. Found keys: {keys}"
        )

    if X_data.max() > 1 or len(X_data.shape) > 2:
        preprocess_dataset(X_data, y_data)

    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(
        X_data, y_data, train_size=0.7, val_size=0.15, test_size=0.15, random_seed=random_seed
    )

    base = os.path.basename(file_path)
    dataset_name, _ = os.path.splitext(base)

    print("")
    print("[OK] Dataset split successfully")
    print(f"------------- {dataset_name} dataset shape -------------")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("")

    return X_train, y_train, X_val, y_val, X_test, y_test



