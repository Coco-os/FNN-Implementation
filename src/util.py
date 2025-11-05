import numpy as np
from typing import Tuple, Iterator, Any
from numpy.typing import NDArray


def create_mini_batches(
    X: NDArray[Any], y: NDArray[Any], batch_size: int, shuffle: bool = True
) -> Iterator[Tuple[NDArray[Any], NDArray[Any]]]:
    num_samples = X.shape[0]
    indices = np.arange(num_samples)

    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]
