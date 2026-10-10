---
name: factor-model-tra
description: Use when training a standalone cross-sectional stock model that synthesizes a user-supplied multi-factor panel into portable ranking signals with validation-based checkpoint selection.
quantSkills:
  schema_version: 2.0.0
  organization: quantskills
  organization_url: https://github.com/quantskills
  repository: skill-factor-model-tra
  repository_url: https://github.com/quantskills/skill-factor-model-tra
  project_type: skill
  license: GPL-3.0-only
  maintainer: wangxueliang
  collection: research-models
  catalog: {category: "06", subcategory: 06.statistical-ml-models}
  workflow: {primary_stage: modeling, workflow_stages: [modeling]}
  tags: [factor-model-tra, factor-model, alpha-model, signal-export]
  requires: []
  summary_zh: "使用 TRA 时序路由模型将用户多因子面板合成为股票横截面预测信号。"
  summary_en: "Synthesizes user-supplied multi-factor panels into cross-sectional stock signals with TRA."
  status: active
  validation_level: runnable
  maintainer_type: community
  platforms: [cursor, claude-code, codex, hermes, openclaw]
  interface:
    mode: structured
    envelope: {name: quantskills-envelope, version: 1.0.0}
    inputs:
      - {profile: factor-panel, version_range: ">=1.0.0 <2.0.0", required: true}
    outputs:
      - {profile: factor-signal, version: 1.0.0}
      - {profile: model-run-manifest, version: 1.0.0}
    adapters: [factor-model-training, factor-signal-export]
---

```json qsh-form
{
  "version": 1,
  "task": {"placeholder": "提供多因子面板、标签可得延迟、时间切分与输出目录", "required": true},
  "fields": [
    {"key": "run_mode", "label": "运行模式", "type": "select", "options": [{"value": "train", "label": "训练"}, {"value": "predict", "label": "导出信号"}]}
  ],
  "prompt_template": "{{task}}\n\n使用 TRA 执行 {{run_mode}}。先校验 YAML 配置与面板数据，再运行对应命令并检查输出清单。"
}
```

# Temporal Routing Adaptor Alpha Model

Use this skill to train a causal routing model on a user-supplied multi-factor
panel. It is self-contained: it does not import Qlib, PandaAI local modules,
other skills, or a bundled backtest engine.

## Workflow

1. Read [the input contract](references/input_contract.md) and prepare one
   CSV or Parquet panel with explicit `date`, `ticker`, feature, and label columns.
2. Start from `examples/smoke.yaml` and configure chronological splits, feature
   columns, model parameters, and the label availability delay.
3. Run validation, training, signal export, and artifact verification:

```bash
python scripts/run_model.py check --config config.yaml
python scripts/run_model.py train --config config.yaml
python scripts/run_model.py predict --run-dir runs/<run_id> --split test
python scripts/run_model.py verify --run-dir runs/<run_id>
```

## Model Boundary

This repository is an independent, non-official TRA reproduction and generic factor-panel adaptation.

TRA uses several prediction heads, a router conditioned on both the feature
representation and causally available historical head errors, and a Sinkhorn
optimal-transport assignment loss. Read
[the method mapping](references/method_mapping.md) for the relation to the
original TRA publication.

## Outputs

Every run writes a resolved configuration, input manifest, training metrics,
best checkpoint, validation summary, and optional `date,ticker,signal` files.
Read [the output contract](references/output_contract.md) before integrating a
signal into another system.

## Research Use

**Data source:** users supply factor panels and labels at runtime; the repository contains only synthetic examples. **Assumptions and parameters:** splits, labels, lookback, model dimensions, and training settings are explicit YAML parameters. **Limitations and risk boundary:** output is a cross-sectional ranking signal, not portfolio construction, execution, or a return guarantee. This project is for research and education and does not constitute investment advice.
