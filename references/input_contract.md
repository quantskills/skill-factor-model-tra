# Input Contract

`data.panel_path` points to one CSV or Parquet table. Each row represents one
stock on one decision date.

```yaml
data:
  panel_path: /path/to/panel.parquet
  date_column: date
  ticker_column: ticker
  label_column: label
  feature_columns: [factor_value, factor_quality]
splits:
  train: [20160104, 20231229]
  valid: [20240102, 20241231]
  test: [20250102, 20261231]
training:
  label_availability_delay: 5
```

`date` is integer-like `YYYYMMDD`; `(date, ticker)` is unique; feature columns
are numeric and explicitly declared. The delay controls when a prediction-head
error may enter the routing memory. An error generated at date `t` is not used
until the configured number of later panel dates have passed.
