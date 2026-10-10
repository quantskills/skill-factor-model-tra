# Configuration Reference

Configuration is YAML. Omitted fields use the defaults below; paths are resolved relative to the configuration file.

## Data and splits

| Key | Default | Meaning |
|---|---:|---|
| `data.panel_path` | required | Path to a CSV or Parquet long-form factor panel. |
| `data.date_column` | required | Integer-like `YYYYMMDD` date column. |
| `data.ticker_column` | required | Instrument identifier column. |
| `data.label_column` | required | Training and validation target column. |
| `data.feature_columns` | required | Numeric factor columns used as model features. |
| `splits.train` | required | `[start_date, end_date]` training interval. |
| `splits.valid` | required | Validation interval. |
| `splits.test` | required | Signal-export interval. Splits are chronological and non-overlapping. |

## Preprocessing

| Key | Default | Meaning |
|---|---:|---|
| `preprocessing.lookback` | `20` | Historical trading dates per prediction; minimum `2`. |
| `preprocessing.standardize_features` | `true` | Cross-sectionally standardise each factor on each date. |
| `preprocessing.standardize_labels` | `true` | Cross-sectionally standardise labels on each date. |

## Model

| Key | Default | Meaning |
|---|---:|---|
| `model.hidden_size` | `64` | GRU backbone hidden width. |
| `model.experts` | `4` | Number of prediction heads. |
| `model.router_hidden` | `8` | Error-memory GRU hidden width. |
| `model.history_length` | `5` | Retained historical errors per stock and head. |
| `model.dropout` | `0.1` | Feature-encoder dropout probability. |
| `model.temperature` | `1.0` | Training Gumbel-softmax temperature. |
| `model.hard_inference` | `true` | Uses the maximum-probability head at inference. |
| `model.historical_cost_weight` | `0.1` | Weight combining current prediction error with historical error cost. |
| `model.transport_temperature` | `0.1` | Sinkhorn optimal-transport temperature. |
| `model.transport_weight` | `0.1` | Initial routing auxiliary-loss weight. |
| `model.transport_decay` | `0.99` | Per-epoch decay for the routing auxiliary loss. |

## Training and output

| Key | Default | Meaning |
|---|---:|---|
| `training.epochs` | `20` | Training epochs. |
| `training.learning_rate` | `0.001` | AdamW learning rate. |
| `training.weight_decay` | `0.0001` | AdamW weight decay. |
| `training.seed` | `42` | Random seed. |
| `training.device` | `cpu` | `cpu`, `cuda`, or `auto`; `auto` prefers available CUDA. |
| `training.selection_metric` | `mean_pearson` | Checkpoint selection metric; this release fixes it to validation mean Pearson IC. |
| `training.label_availability_delay` | `1` | Panel dates before an error may enter memory; minimum `1`. |
| `output_root` | `runs` | Root directory for run outputs. |
| `run_name` | timestamp | Optional fixed run directory name; an existing name fails closed. |
