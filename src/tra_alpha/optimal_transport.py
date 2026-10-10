from __future__ import annotations

import math

import torch


def balanced_assignment(costs: torch.Tensor, temperature: float = 0.1, iterations: int = 50) -> torch.Tensor:
    """Return a row-stochastic, approximately balanced Sinkhorn assignment."""
    if costs.ndim != 2 or len(costs) == 0 or temperature <= 0:
        raise ValueError("costs must be a non-empty rank-2 tensor and temperature must be positive")
    centered = costs - costs.min(dim=-1, keepdim=True).values
    scaled = centered / centered.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
    log_weights = -scaled / temperature
    column_target = math.log(costs.shape[0] / costs.shape[1])
    for _ in range(iterations):
        log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
        log_weights = log_weights + column_target - torch.logsumexp(log_weights, dim=0, keepdim=True)
    return log_weights.softmax(dim=1)
