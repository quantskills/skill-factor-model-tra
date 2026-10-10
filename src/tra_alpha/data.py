from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _read_frame(path: str) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    if source.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(source)
    if source.suffix.lower() == ".csv":
        return pd.read_csv(source)
    raise ValueError("data.panel_path must be a CSV or Parquet file")


def _standardize(values: np.ndarray) -> np.ndarray:
    output = values.copy()
    for date_index in range(values.shape[0]):
        row = values[date_index]
        if row.ndim == 1:
            mask = np.isfinite(row)
            if mask.any():
                std = row[mask].std()
                output[date_index, mask] = 0.0 if std < 1e-12 else (row[mask] - row[mask].mean()) / std
        else:
            for column in range(row.shape[1]):
                mask = np.isfinite(row[:, column])
                if mask.any():
                    std = row[mask, column].std()
                    output[date_index, mask, column] = 0.0 if std < 1e-12 else (row[mask, column] - row[mask, column].mean()) / std
    return output


@dataclass
class PanelData:
    dates: np.ndarray
    tickers: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    lookback: int
    feature_columns: list[str]

    def positions(self, split: list[int], require_label: bool) -> np.ndarray:
        start, end = map(int, split)
        positions = np.flatnonzero((self.dates >= start) & (self.dates <= end))
        positions = positions[positions >= self.lookback - 1]
        if require_label:
            positions = np.array([pos for pos in positions if np.isfinite(self.labels[pos]).any()], dtype=int)
        return positions

    def batch(self, position: int, require_label: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        history = self.features[position - self.lookback + 1:position + 1]
        valid = np.isfinite(history).all(axis=(0, 2))
        if require_label:
            valid &= np.isfinite(self.labels[position])
        stocks = np.flatnonzero(valid)
        values = history[:, stocks, :].transpose(1, 0, 2).astype(np.float32, copy=True)
        labels = self.labels[position, stocks].astype(np.float32, copy=True) if require_label else None
        return stocks, values, labels

    def summary(self, splits: dict) -> dict:
        result = {"dates": int(len(self.dates)), "tickers": int(len(self.tickers)), "feature_count": int(self.features.shape[-1]),
                  "date_start": int(self.dates[0]), "date_end": int(self.dates[-1])}
        for name, bounds in splits.items():
            result[f"{name}_positions"] = int(len(self.positions(bounds, require_label=name != "test")))
        return result


def load_panel(config: dict) -> PanelData:
    data_config = config["data"]
    date_col, ticker_col, label_col = data_config["date_column"], data_config["ticker_column"], data_config["label_column"]
    feature_cols = list(data_config["feature_columns"])
    frame = _read_frame(data_config["panel_path"])
    required = [date_col, ticker_col, *feature_cols]
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"missing panel columns: {', '.join(absent)}")
    if label_col not in frame.columns:
        frame[label_col] = np.nan
    frame = frame[[date_col, ticker_col, *feature_cols, label_col]].copy()
    frame[date_col] = pd.to_numeric(frame[date_col], errors="raise").astype("int64")
    if frame.duplicated([date_col, ticker_col]).any():
        raise ValueError("panel has duplicate (date, ticker) rows")
    for column in [*feature_cols, label_col]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    dates = np.sort(frame[date_col].unique())
    tickers = np.array(sorted(frame[ticker_col].unique().tolist(), key=lambda item: str(item)), dtype=object)
    features = np.full((len(dates), len(tickers), len(feature_cols)), np.nan, dtype=np.float32)
    labels = np.full((len(dates), len(tickers)), np.nan, dtype=np.float32)
    date_lookup = {date: index for index, date in enumerate(dates)}
    ticker_lookup = {ticker: index for index, ticker in enumerate(tickers)}
    for row in frame.itertuples(index=False, name=None):
        date, ticker, *values = row
        features[date_lookup[date], ticker_lookup[ticker], :] = np.asarray(values[:-1], dtype=np.float32)
        labels[date_lookup[date], ticker_lookup[ticker]] = values[-1]
    if config["preprocessing"].get("standardize_features", True):
        features = _standardize(features)
    if config["preprocessing"].get("standardize_labels", True):
        labels = _standardize(labels)
    return PanelData(dates=dates, tickers=tickers, features=features, labels=labels,
                     lookback=int(config["preprocessing"]["lookback"]), feature_columns=feature_cols)
