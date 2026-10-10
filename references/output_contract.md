# Output Contract

A completed run directory contains `resolved_config.yaml`, `input_manifest.json`,
`training_metrics.jsonl`, `best_checkpoint.pt`, `validation_summary.json`, and
`run_manifest.json`. The optional signal export is `signals_<split>.parquet`.

Signal columns are always:

```text
date,ticker,signal
```

The score is designed for cross-sectional ranking. Portfolio construction and
backtesting are deliberately outside this repository.
