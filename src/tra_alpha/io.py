from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml


def timestamped_run_dir(output_root: str, run_name: str | None = None) -> Path:
    name = run_name or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    destination = Path(output_root) / name
    if destination.exists():
        raise FileExistsError(f"run directory already exists: {destination}")
    destination.mkdir(parents=True)
    return destination


def write_json(path: str | Path, value: dict) -> None:
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")


def write_yaml(path: str | Path, value: dict) -> None:
    Path(path).write_text(yaml.safe_dump(value, allow_unicode=True, sort_keys=False), encoding="utf-8")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
