import numpy as np

def to_categorical(y, num_classes=None, dtype=np.float32):

    y = np.asarray(y)

    if y.ndim == 2 and y.shape[1] > 1 and set(np.unique(y)) <= {0, 1}:
        if num_classes is not None and y.shape[1] != num_classes:
            raise ValueError(f"num_classes={num_classes} no coincide con y.shape[1]={y.shape[1]}")
        return y.astype(dtype, copy=False)

    y = y.reshape(-1)

    if not np.issubdtype(y.dtype, np.integer):
        y = y.astype(int)

    if num_classes is None:
        num_classes = int(y.max()) + 1 if y.size else 0

    if (y < 0).any() or (y >= num_classes).any():
        raise ValueError("Todas las etiquetas deben estar en el rango [0, num_classes).")

    out = np.zeros((y.shape[0], num_classes), dtype=dtype)
    if y.size:
        out[np.arange(y.shape[0]), y] = 1
    return out
