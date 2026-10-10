# Temporal Routing Adaptor Alpha Model

`skill-factor-model-tra` 是一个独立、可安装的 TRA 多因子合成模型。它通过多预测头、因果历史误差记忆和最优传输路由，在用户提供的因子面板上生成股票横截面预测信号。

## 功能

- 读取 CSV 或 Parquet 多因子面板；
- 使用 GRU 编码历史因子序列；
- 以多个预测头学习不同模式，并由路由器完成样本分配；
- 将已实现预测误差按配置延迟写入历史记忆，避免当期标签进入当期决策；
- 使用 Sinkhorn 最优传输辅助损失保持预测头分工；
- 按验证集平均 Pearson IC 选择 checkpoint，并输出标准 `date,ticker,signal` 文件。

## 安装与快速开始

```bash
python -m pip install -e ".[test]"
python examples/make_synthetic_panel.py   --output examples/generated/synthetic_panel.csv
python scripts/run_model.py check --config examples/smoke.yaml
python scripts/run_model.py train --config examples/smoke.yaml
```

训练后导出信号并校验：

```bash
python scripts/run_model.py predict   --run-dir outputs/smoke/<run_id> --split test
python scripts/run_model.py verify   --run-dir outputs/smoke/<run_id>
```

运行合成数据冒烟测试和单元测试：

```bash
python scripts/run_smoke.py
python -m pytest -q
```

## 输入数据与配置

输入是一张用户提供的长表，每行对应一个 `(date, ticker)`：

| 列 | 含义 |
|---|---|
| `date` | 整数日期，格式为 `YYYYMMDD` |
| `ticker` | 股票标识符 |
| `factor_*` | 一个或多个数值因子列 |
| `label` | 训练和验证标签；未来预测行可以为空 |

核心配置包括因子列、时间切分、回看期、预测头数量、误差记忆长度、Sinkhorn 温度和标签可得延迟：

```yaml
data:
  panel_path: /path/to/panel.parquet
  date_column: date
  ticker_column: ticker
  label_column: label
  feature_columns: [factor_value, factor_quality, factor_momentum]
splits:
  train: [20160104, 20231229]
  valid: [20240102, 20241231]
  test: [20250102, 20261231]
preprocessing:
  lookback: 20
model:
  experts: 4
  history_length: 5
  transport_temperature: 0.1
training:
  label_availability_delay: 5
  epochs: 20
  seed: 42
```

完整字段说明见 [输入契约](references/input_contract.md)。

## 输出

每次运行在 `output_root/run_<timestamp>/` 下保存冻结配置、输入清单、训练指标、最佳 checkpoint、验证摘要和信号文件。信号格式固定为：

```text
date,ticker,signal
20250102,600000,0.1837
```

完整产物定义见 [输出契约](references/output_contract.md)。

## 方法与论文参考

本项目参考以下论文的 TRA 路由思想：

> Hengxu Lin, Dong Zhou, Weiqing Liu, Jiang Bian. **Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport.** KDD 2021. DOI: [10.1145/3447548.3467358](https://doi.org/10.1145/3447548.3467358).

这是一个**非官方独立复现与通用因子面板适配版本**，并非论文作者或 Microsoft Qlib 发布的官方实现。它保留多预测头、基于表示与历史误差的路由、Gumbel 路由与 Sinkhorn OT 监督；主干网络、标签定义、数据面板和训练参数由 YAML 配置，因此不以复现论文的具体数值结果为目标。详细映射见 [方法映射说明](references/method_mapping.md)。

论文公开版本：[arXiv](https://arxiv.org/abs/2106.12950)；Qlib 参考目录：[GitHub](https://github.com/microsoft/qlib/tree/main/examples/benchmarks/TRA)。

## 研究使用说明

**数据来源**：运行时由用户提供多因子面板和标签；仓库仅包含合成样例。**假设与参数**：时间切分、标签可得延迟、回看期、预测头数量和 OT 参数均通过 YAML 明确配置。**限制与风险边界**：输出为横截面排序信号，不包含组合构建、交易执行或收益保证。该项目仅用于研究与教育，不构成任何投资建议。

## 许可证

本项目采用 [GPL-3.0-only](LICENSE) 许可证。
