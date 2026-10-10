from __future__ import annotations

import copy
from pathlib import Path

import yaml


DEFAULTS = {
    "preprocessing": {"lookback": 20, "standardize_features": True, "standardize_labels": True},
    "model": {"hidden_size": 64, "experts": 4, "router_hidden": 8, "history_length": 5,
              "dropout": 0.1, "temperature": 1.0, "hard_inference": True,
              "historical_cost_weight": 0.1, "transport_temperature": 0.1,
              "transport_weight": 0.1, "transport_decay": 0.99},
    "training": {"epochs": 20, "learning_rate": 0.001, "weight_decay": 0.0001, "seed": 42,
                 "device": "cpu", "selection_metric": "mean_pearson", "label_availability_delay": 1},
    "output_root": "runs",
}


def _merge(base: dict, override: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict:
    source = Path(path).expanduser().resolve()
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("configuration must be a YAML mapping")
    config = _merge(DEFAULTS, loaded)
    config["_config_path"] = str(source)
    config["_config_dir"] = str(source.parent)
    if "panel_path" not in config.get("data", {}):
        raise ValueError("data.panel_path is required")
    panel_path = Path(config["data"]["panel_path"]).expanduser()
    config["data"]["panel_path"] = str((source.parent / panel_path).resolve() if not panel_path.is_absolute() else panel_path)
    output = Path(config["output_root"]).expanduser()
    config["output_root"] = str((source.parent / output).resolve() if not output.is_absolute() else output)
    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    data = config.get("data", {})
    required = ["date_column", "ticker_column", "label_column", "feature_columns"]
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"missing data settings: {', '.join(missing)}")
    if not isinstance(data["feature_columns"], list) or not data["feature_columns"]:
        raise ValueError("data.feature_columns must be a non-empty list")
    if len(set(data["feature_columns"])) != len(data["feature_columns"]):
        raise ValueError("data.feature_columns contains duplicates")
    splits = config.get("splits", {})
    for name in ("train", "valid", "test"):
        if name not in splits or not isinstance(splits[name], list) or len(splits[name]) != 2:
            raise ValueError(f"splits.{name} must be [start_date, end_date]")
        start, end = map(int, splits[name])
        if start > end:
            raise ValueError(f"splits.{name} start exceeds end")
    ordered = [(int(splits[name][0]), int(splits[name][1])) for name in ("train", "valid", "test")]
    if ordered[0][1] >= ordered[1][0] or ordered[1][1] >= ordered[2][0]:
        raise ValueError("train, valid, and test splits must be strictly chronological and non-overlapping")
    if int(config["preprocessing"]["lookback"]) < 2:
        raise ValueError("preprocessing.lookback must be at least 2")
    if int(config["model"]["experts"]) < 1 or int(config["model"]["history_length"]) < 1:
        raise ValueError("model.experts and model.history_length must be positive")
    if int(config["training"]["epochs"]) < 1 or int(config["training"]["label_availability_delay"]) < 1:
        raise ValueError("training.epochs and label_availability_delay must be positive")
    if config["training"].get("selection_metric") != "mean_pearson":
        raise ValueError("training.selection_metric must be mean_pearson")
