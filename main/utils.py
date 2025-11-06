from typing import Iterator, Tuple
import numpy as np


def create_mini_batches(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = True):
    """
    X: shape (n_features, n_samples)
    y: shape (n_samples,)  <- class labels
    """
    num_samples = X.shape[1]  # samples are in axis 1
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[:, batch_indices], y[batch_indices]



def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int | None = None,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:

    assert (
        train_size + val_size + test_size == 1.0
    ), "Las proporciones deben sumar 1.0"

    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if random_seed is not None:
        np.random.seed(random_seed)

    np.random.shuffle(indices)

    train_end = int(train_size * num_samples)
    val_end = train_end + int(val_size * num_samples)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
