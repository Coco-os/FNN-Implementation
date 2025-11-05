import os
import numpy as np
from sklearn.datasets import load_iris

def download_iris_dataset():
    """Download the IRIS dataset and save it locally as NumPy arrays."""
    print("[INFO] Downloading the IRIS dataset...")

    # Load the dataset from scikit-learn
    iris = load_iris()
    X = iris.data          # Features
    y = iris.target        # Labels
    feature_names = iris.feature_names
    target_names = iris.target_names

    os.makedirs("datasets", exist_ok=True)
    out_path = "datasets/iris.npz"

    np.savez_compressed(out_path,
                        X=X,
                        y=y,
                        feature_names=feature_names,
                        target_names=target_names)

    print(f"[OK] IRIS dataset saved at {out_path}")



download_iris_dataset()
