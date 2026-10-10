from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import load_config
from .data import PanelData, load_panel
from .io import set_seed, timestamped_run_dir, write_json, write_yaml
from .memory import CausalErrorMemory
from .model import TemporalRoutingAdaptor


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device requests CUDA but CUDA is unavailable")
    return device


def _pearson(prediction: np.ndarray, labels: np.ndarray) -> float | None:
    if len(prediction) < 3 or np.std(prediction) < 1e-12 or np.std(labels) < 1e-12:
        return None
    return float(np.corrcoef(prediction, labels)[0, 1])


def _apply_matured(memory: CausalErrorMemory, pending: dict, position: int) -> None:
    for stocks, expert_predictions, labels in pending.pop(position, []):
        memory.update(stocks, np.square(expert_predictions - labels[:, None]))


def _schedule_error(pending: dict, available_position: int, stocks: np.ndarray, expert_predictions: np.ndarray,
                    labels: np.ndarray, delay: int) -> None:
    known = np.isfinite(labels)
    if known.any():
        pending[available_position + delay].append((stocks[known].copy(), expert_predictions[known].copy(), labels[known].copy()))


def _walk(model: TemporalRoutingAdaptor, panel: PanelData, positions: np.ndarray, device: torch.device,
          delay: int, optimizer: torch.optim.Optimizer | None = None, epoch: int = 0) -> dict:
    memory = CausalErrorMemory(len(panel.tickers), model.history_length, model.expert_count)
    pending: dict = defaultdict(list)
    losses: list[float] = []
    correlations: list[float] = []
    model.train(optimizer is not None)
    for position in positions:
        position = int(position)
        _apply_matured(memory, pending, position)
        require_label = optimizer is not None
        stocks, features, train_labels = panel.batch(position, require_label=require_label)
        if len(stocks) < 3:
            continue
        history = torch.as_tensor(memory.history(stocks), device=device)
        inputs = torch.as_tensor(features, device=device)
        if optimizer is not None:
            assert train_labels is not None
            labels = torch.as_tensor(train_labels, device=device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(inputs, history)
            loss, _ = model.objective(outputs, labels, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            expert = outputs["expert_predictions"].detach().cpu().numpy()
            observed = train_labels
        else:
            with torch.no_grad():
                outputs = model(inputs, history)
            expert = outputs["expert_predictions"].detach().cpu().numpy()
            observed = panel.labels[position, stocks]
            score = _pearson(outputs["prediction"].detach().cpu().numpy(), observed)
            if score is not None:
                correlations.append(score)
        _schedule_error(pending, position, stocks, expert, observed, delay)
    result = {"mean_pearson": float(np.mean(correlations)) if correlations else None,
              "valid_dates": len(correlations), "mean_loss": float(np.mean(losses)) if losses else None}
    return result


def train(config_path: str | Path) -> Path:
    config = load_config(config_path)
    panel = load_panel(config)
    train_positions = panel.positions(config["splits"]["train"], require_label=True)
    valid_positions = panel.positions(config["splits"]["valid"], require_label=True)
    if not len(train_positions) or not len(valid_positions):
        raise ValueError("train and valid splits must contain labeled positions after lookback")
    set_seed(int(config["training"]["seed"]))
    device = _device(config["training"]["device"])
    model = TemporalRoutingAdaptor(panel.features.shape[-1], config["model"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["learning_rate"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    delay = int(config["training"]["label_availability_delay"])
    run_dir = timestamped_run_dir(config["output_root"], config.get("run_name"))
    write_yaml(run_dir / "resolved_config.yaml", config)
    write_json(run_dir / "input_manifest.json", {"panel_path": config["data"]["panel_path"],
                                                    "panel_summary": panel.summary(config["splits"]),
                                                    "feature_columns": panel.feature_columns,
                                                    "label_availability_delay": delay})
    metrics_path = run_dir / "training_metrics.jsonl"
    best_score, best_epoch = -float("inf"), -1
    for epoch in range(int(config["training"]["epochs"])):
        train_result = _walk(model, panel, train_positions, device, delay, optimizer=optimizer, epoch=epoch)
        valid_result = _walk(model, panel, valid_positions, device, delay)
        if valid_result["mean_pearson"] is None:
            raise ValueError("validation contains no cross-sections with a defined Pearson IC")
        record = {"epoch": epoch + 1, "train_loss": train_result["mean_loss"],
                  "mean_pearson": valid_result["mean_pearson"], "valid_dates": valid_result["valid_dates"]}
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if record["mean_pearson"] > best_score:
            best_score, best_epoch = record["mean_pearson"], epoch + 1
            torch.save({"model_state": model.state_dict(), "model_settings": config["model"],
                        "input_size": panel.features.shape[-1], "best_epoch": best_epoch,
                        "best_valid_pearson": best_score}, run_dir / "best_checkpoint.pt")
    write_json(run_dir / "validation_summary.json", {"status": "complete", "best_epoch": best_epoch,
                                                        "best_valid_pearson": best_score, "device": str(device)})
    write_json(run_dir / "run_manifest.json", {"status": "complete", "run_dir": str(run_dir),
                                                  "artifacts": ["resolved_config.yaml", "input_manifest.json",
                                                                "training_metrics.jsonl", "best_checkpoint.pt",
                                                                "validation_summary.json"]})
    return run_dir


def _load_run(run_dir: str | Path) -> tuple[dict, PanelData, TemporalRoutingAdaptor, torch.device]:
    root = Path(run_dir).resolve()
    config = load_config(root / "resolved_config.yaml")
    panel = load_panel(config)
    device = _device(config["training"]["device"])
    checkpoint = torch.load(root / "best_checkpoint.pt", map_location=device, weights_only=True)
    model = TemporalRoutingAdaptor(int(checkpoint["input_size"]), checkpoint["model_settings"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, panel, model, device


def predict(run_dir: str | Path, split: str) -> Path:
    if split not in {"train", "valid", "test"}:
        raise ValueError("split must be train, valid, or test")
    config, panel, model, device = _load_run(run_dir)
    positions = panel.positions(config["splits"][split], require_label=False)
    delay = int(config["training"]["label_availability_delay"])
    memory = CausalErrorMemory(len(panel.tickers), model.history_length, model.expert_count)
    pending: dict = defaultdict(list)
    rows = []
    with torch.no_grad():
        for position in positions:
            position = int(position)
            _apply_matured(memory, pending, position)
            stocks, features, _ = panel.batch(position, require_label=False)
            if not len(stocks):
                continue
            outputs = model(torch.as_tensor(features, device=device), torch.as_tensor(memory.history(stocks), device=device))
            signal = outputs["prediction"].detach().cpu().numpy()
            expert = outputs["expert_predictions"].detach().cpu().numpy()
            observed = panel.labels[position, stocks]
            _schedule_error(pending, position, stocks, expert, observed, delay)
            rows.append(pd.DataFrame({"date": int(panel.dates[position]), "ticker": panel.tickers[stocks],
                                      "signal": signal.astype("float64")}))
    if not rows:
        raise ValueError(f"no predictable rows in split {split}")
    destination = Path(run_dir) / f"signals_{split}.parquet"
    output = pd.concat(rows, ignore_index=True)
    output.to_parquet(destination, index=False)
    write_json(Path(run_dir) / f"signals_{split}_summary.json", {"rows": int(len(output)), "dates": int(output.date.nunique()),
                                                                    "path": str(destination), "label_delay": delay})
    return destination


def verify(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    required = ["resolved_config.yaml", "input_manifest.json", "training_metrics.jsonl", "best_checkpoint.pt",
                "validation_summary.json", "run_manifest.json"]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing run artifacts: {', '.join(missing)}")
    summary = json.loads((root / "validation_summary.json").read_text(encoding="utf-8"))
    return {"status": "verified", "run_dir": str(root.resolve()), "best_valid_pearson": summary["best_valid_pearson"]}
