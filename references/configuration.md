# 配置参数说明

配置文件为 YAML。未填写的字段使用下表默认值；所有路径相对于配置文件所在目录解析。

## 数据与时间切分

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `data.panel_path` | 必填 | CSV 或 Parquet 多因子长表路径。 |
| `data.date_column` | 必填 | 整数日期列，格式为 `YYYYMMDD`。 |
| `data.ticker_column` | 必填 | 标的代码列。 |
| `data.label_column` | 必填 | 训练与验证标签列。 |
| `data.feature_columns` | 必填 | 用于建模的数值因子列列表。 |
| `splits.train` | 必填 | `[开始日期, 结束日期]` 训练区间。 |
| `splits.valid` | 必填 | 验证区间。 |
| `splits.test` | 必填 | 信号导出区间。三段区间按时间顺序且不重叠。 |

## 预处理

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `preprocessing.lookback` | `20` | 每次预测使用的历史交易日数量，最小值为 `2`。 |
| `preprocessing.standardize_features` | `true` | 按日期对每个因子做横截面标准化。 |
| `preprocessing.standardize_labels` | `true` | 按日期对标签做横截面标准化。 |

## 模型

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `model.hidden_size` | `64` | GRU 主干网络隐藏维度。 |
| `model.experts` | `4` | 预测头数量。 |
| `model.router_hidden` | `8` | 误差记忆 GRU 隐藏维度。 |
| `model.history_length` | `5` | 每只股票、每个预测头保留的历史误差长度。 |
| `model.dropout` | `0.1` | 特征编码器 dropout 比例。 |
| `model.temperature` | `1.0` | 训练期 Gumbel-softmax 温度。 |
| `model.hard_inference` | `true` | 推理期使用最大路由概率对应的预测头。 |
| `model.historical_cost_weight` | `0.1` | 当前预测误差与历史误差成本的混合权重。 |
| `model.transport_temperature` | `0.1` | Sinkhorn 最优传输温度。 |
| `model.transport_weight` | `0.1` | 路由辅助损失初始权重。 |
| `model.transport_decay` | `0.99` | 每个 epoch 的路由辅助损失衰减系数。 |

## 训练与输出

| 字段 | 默认值 | 说明 |
|---|---:|---|
| `training.epochs` | `20` | 训练 epoch 数。 |
| `training.learning_rate` | `0.001` | AdamW 学习率。 |
| `training.weight_decay` | `0.0001` | AdamW 权重衰减。 |
| `training.seed` | `42` | 随机种子。 |
| `training.device` | `cpu` | `cpu`、`cuda` 或 `auto`。`auto` 优先使用可用 CUDA。 |
| `training.selection_metric` | `mean_pearson` | 验证集 checkpoint 选择指标；当前版本固定为验证集平均 Pearson IC。 |
| `training.label_availability_delay` | `1` | 标签对应的预测头误差进入记忆前需经过的面板日期数量，最小值为 `1`。 |
| `output_root` | `runs` | 运行目录根路径。 |
| `run_name` | 自动时间戳 | 可选固定运行目录名；同名目录已存在时运行会停止。 |
