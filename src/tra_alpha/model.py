from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .optimal_transport import balanced_assignment


class SequenceEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(input_size, hidden_size), nn.SiLU())
        self.recurrent = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape [stocks, history, features]")
        _, hidden = self.recurrent(self.projection(inputs))
        return self.dropout(hidden[-1])


class TemporalRoutingAdaptor(nn.Module):
    """Multi-head predictor with error-memory routing and OT supervision."""

    def __init__(self, input_size: int, settings: dict) -> None:
        super().__init__()
        self.settings = dict(settings)
        hidden = int(settings["hidden_size"])
        self.expert_count = int(settings["experts"])
        self.history_length = int(settings["history_length"])
        self.encoder = SequenceEncoder(input_size, hidden, float(settings.get("dropout", 0.1)))
        self.predictors = nn.Linear(hidden, self.expert_count)
        self.error_encoder = nn.GRU(self.expert_count, int(settings["router_hidden"]), batch_first=True)
        self.router = nn.Linear(hidden + int(settings["router_hidden"]), self.expert_count)

    def forward(self, inputs: torch.Tensor, error_history: torch.Tensor | None = None) -> dict:
        hidden = self.encoder(inputs)
        expert_predictions = self.predictors(hidden)
        if error_history is None:
            error_history = inputs.new_zeros((len(inputs), self.history_length, self.expert_count))
        expected = (len(inputs), self.history_length, self.expert_count)
        if tuple(error_history.shape) != expected:
            raise ValueError(f"error_history must have shape {expected}")
        _, state = self.error_encoder(error_history)
        logits = self.router(torch.cat([hidden, state[-1]], dim=-1))
        probabilities = logits.softmax(dim=-1)
        if self.training:
            routing = F.gumbel_softmax(logits, tau=float(self.settings.get("temperature", 1.0)), hard=False)
        elif self.settings.get("hard_inference", True):
            routing = F.one_hot(probabilities.argmax(dim=-1), self.expert_count).to(probabilities.dtype)
        else:
            routing = probabilities
        return {"prediction": (routing * expert_predictions).sum(dim=-1), "expert_predictions": expert_predictions,
                "probabilities": probabilities, "routing_logits": logits, "history_costs": error_history.mean(dim=1)}

    def objective(self, outputs: dict, labels: torch.Tensor, epoch: int) -> tuple[torch.Tensor, dict]:
        prediction_loss = F.smooth_l1_loss(outputs["prediction"], labels)
        errors = (outputs["expert_predictions"].detach() - labels[:, None]).square()
        mixture = float(self.settings.get("historical_cost_weight", 0.1))
        costs = (1.0 - mixture) * errors + mixture * outputs["history_costs"].detach()
        assignment = balanced_assignment(costs, float(self.settings.get("transport_temperature", 0.1)))
        routing_loss = -(assignment * outputs["routing_logits"].log_softmax(dim=-1)).sum(dim=-1).mean()
        coefficient = float(self.settings.get("transport_weight", 0.1)) * float(self.settings.get("transport_decay", 0.99)) ** epoch
        return prediction_loss + coefficient * routing_loss, {"prediction_loss": float(prediction_loss.detach()),
                                                                "transport_loss": float(routing_loss.detach()),
                                                                "transport_coefficient": coefficient}
