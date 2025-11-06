from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple, Any
import numpy as np

Array = np.ndarray
Shape = Tuple[int, ...]


class BaseOptimizer(ABC):
    def __init__(self, learning_rate: float = 1e-2) -> None:
        self.learning_rate = float(learning_rate)

    @abstractmethod
    def initialize(self, weights_shape: Shape, bias_shape: Shape) -> None:
        raise NotImplementedError

    @abstractmethod
    def update(
        self,
        weights: Array,
        bias: Array,
        dW: Array,
        dB: Array,
    ) -> tuple[Array, Array]:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def on_epoch_start(self, epoch: int) -> None:
        pass

    def on_epoch_end(self, epoch: int) -> None:
        pass

    def state_dict(self) -> dict[str, Any]:
        return {"lr": self.learning_rate}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.learning_rate = float(state.get("lr", self.learning_rate))
