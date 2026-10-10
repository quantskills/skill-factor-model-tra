from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from tra_alpha.config import load_config
from tra_alpha.data import load_panel
from tra_alpha.memory import CausalErrorMemory
from tra_alpha.model import TemporalRoutingAdaptor
from tra_alpha.optimal_transport import balanced_assignment
from tra_alpha.trainer import predict, train, verify


def _config(tmp_path: Path) -> Path:
    rng = np.random.default_rng(6)
    dates = pd.bdate_range("2021-01-04", periods=36).strftime("%Y%m%d").astype(int)
    rows = []
    for date in dates:
        for ticker in range(1, 11):
            first, second = rng.normal(size=2)
            rows.append({"date": int(date), "ticker": ticker, "first": first, "second": second,
                         "label": 0.35 * first - 0.25 * second + rng.normal(scale=0.15)})
    panel_path = tmp_path / "panel.csv"
    pd.DataFrame(rows).to_csv(panel_path, index=False)
    config = {
        "data": {"panel_path": str(panel_path), "date_column": "date", "ticker_column": "ticker",
                 "label_column": "label", "feature_columns": ["first", "second"]},
        "splits": {"train": [int(dates[0]), int(dates[18])], "valid": [int(dates[19]), int(dates[26])],
                   "test": [int(dates[27]), int(dates[-1])]},
        "preprocessing": {"lookback": 4, "standardize_features": True, "standardize_labels": True},
        "model": {"hidden_size": 8, "experts": 2, "router_hidden": 4, "history_length": 2, "dropout": 0.0,
                  "temperature": 1.0, "hard_inference": True, "historical_cost_weight": 0.1,
                  "transport_temperature": 0.2, "transport_weight": 0.1, "transport_decay": 0.99},
        "training": {"epochs": 2, "learning_rate": 0.01, "weight_decay": 0.0, "seed": 4, "device": "cpu",
                     "label_availability_delay": 2},
        "output_root": str(tmp_path / "runs"),
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return path


def test_train_predict_and_verify(tmp_path: Path) -> None:
    run_dir = train(_config(tmp_path))
    signals = predict(run_dir, "test")
    frame = pd.read_parquet(signals)
    assert list(frame.columns) == ["date", "ticker", "signal"]
    assert len(frame) > 0
    assert verify(run_dir)["status"] == "verified"


def test_memory_uses_prior_updates_only() -> None:
    memory = CausalErrorMemory(stock_count=3, history_length=2, expert_count=2)
    before = memory.history(np.array([0]))
    memory.update(np.array([0]), np.array([[1.0, 2.0]], dtype=np.float32))
    after = memory.history(np.array([0]))
    assert np.allclose(before, 0.0)
    assert np.allclose(after[0, -1], [1.0, 2.0])


def test_sinkhorn_assignment_is_balanced() -> None:
    assignment = balanced_assignment(torch.tensor([[0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.9, 0.1]]), temperature=0.2)
    assert torch.allclose(assignment.sum(dim=1), torch.ones(4), atol=1e-5)
    assert torch.allclose(assignment.sum(dim=0), torch.full((2,), 2.0), atol=1e-3)


def test_forward_accepts_history(tmp_path: Path) -> None:
    config = load_config(_config(tmp_path))
    panel = load_panel(config)
    _, values, _ = panel.batch(panel.lookback - 1, require_label=False)
    model = TemporalRoutingAdaptor(values.shape[-1], config["model"])
    output = model(torch.as_tensor(values), torch.zeros((len(values), 2, 2)))
    assert output["prediction"].shape[0] == values.shape[0]
