# Temporal Routing Adaptor Alpha Model

A standalone Temporal Routing Adaptor model for synthesising multi-factor
panels into cross-sectional stock signals. It exports portable
`date,ticker,signal` files without a dependency on a specific backtester.

```bash
python -m pip install -e .
python examples/make_synthetic_panel.py --output examples/generated/synthetic_panel.csv
python scripts/run_model.py check --config examples/smoke.yaml
python scripts/run_model.py train --config examples/smoke.yaml
```

## Research Use

**Data source:** users supply factor panels and labels at runtime; the repository contains only synthetic examples. **Assumptions and parameters:** splits, labels, lookback, model dimensions, and training settings are explicit YAML parameters. **Limitations and risk boundary:** output is a cross-sectional ranking signal, not portfolio construction, execution, or a return guarantee. This project is for research and education and does not constitute investment advice.
