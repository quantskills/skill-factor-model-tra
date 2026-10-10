from __future__ import annotations

import numpy as np


class CausalErrorMemory:
    """Per-stock error history; callers decide when an observation has matured."""

    def __init__(self, stock_count: int, history_length: int, expert_count: int) -> None:
        self.values = np.zeros((stock_count, history_length, expert_count), dtype=np.float32)

    def history(self, stocks: np.ndarray) -> np.ndarray:
        return self.values[stocks].copy()

    def update(self, stocks: np.ndarray, errors: np.ndarray) -> None:
        if len(stocks) != len(errors) or errors.shape[1] != self.values.shape[2]:
            raise ValueError("error-memory update shape mismatch")
        self.values[stocks, :-1, :] = self.values[stocks, 1:, :]
        self.values[stocks, -1, :] = errors
