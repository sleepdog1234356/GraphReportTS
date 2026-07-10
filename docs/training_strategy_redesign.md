# GraphReportTS 正式训练策略修订设计

日期：2026-07-10  
状态：已确认，待实施

## 1. 背景

当前正式管线把六个 official baseline 统一交给同一个训练循环：固定
`AdamW(lr=1e-3, weight_decay=1e-4)`、SmoothL1、无学习率调度，并把早停
patience 设为最大 epoch。该设置能够调用各论文的模型类，但没有保留各来源代码的
优化器、损失和 scheduler 语义。

MIT 已完成日志显示，固定高学习率会产生长时间验证平台和晚期偶然刷新；例如
PatchTST、iTransformer、TimeCMA、TimesNet、DLinear 的最佳验证 checkpoint 分别
出现在第 73、79、54、77、72 epoch。直接把 patience 改为 12 或 15 会在这些最佳点
之前停止，因此不能只通过缩短 patience 解决训练耗时问题。

本次修订恢复 baseline 的来源一致训练机制，为 GraphReportTS 制定适合其多分支结构的
训练策略，并把管线顺序调整为主模型优先。

## 2. 目标与约束

- 保持无未来数据泄露和无历史 SOH 的输入协议不变。
- 保持各模型结构、输入长度、预测长度和数据划分不变。
- baseline 使用来源代码的优化器、回归损失、学习率调度和早停语义。
- 不复制来源代码中可能使用测试集选模的行为；统一只用验证集选择 checkpoint。
- DistilBERT backbone 保持冻结，不把简化文本编码器带回正式训练。
- 主模型、完整模型参考和全部消融使用同一套训练策略。
- 新旧训练协议的产物必须物理隔离，禁止在同一结果目录混用。
- 正式流程仍是完整训练，不使用缩小模型、缩短数据或 smoke 配置代替正式结果。

## 3. Baseline 训练配置

所有 baseline 共享：batch size 128、输入 32 cycles、预测 20 steps、seed 42、相同
train/val/test split，以及禁止历史 SOH 的五维可观测输入。最终测试必须加载验证 MSE
最低的 `best.pt`。

| 模型 | 优化器与损失 | 学习率调度 | 梯度裁剪 | 训练上限与早停 |
| --- | --- | --- | --- | --- |
| PatchTST | Adam，MSE，lr=1e-4 | OneCycleLR，pct_start=0.3，每 batch step | 无 | 100 epoch，patience=20 |
| iTransformer | Adam，MSE，lr=1e-4 | 来源代码 type1 分段衰减，每 epoch step | 无 | 10 epoch，patience=3 |
| TimesNet | Adam，MSE，lr=1e-4 | 来源代码 type1 分段衰减，每 epoch step | 无 | 10 epoch，patience=3 |
| DLinear | Adam，MSE，lr=1e-4 | 来源代码 type1 分段衰减，每 epoch step | 无 | 10 epoch，patience=3 |
| Time-LLM | Adam，MSE，lr=1e-3 | OneCycleLR，pct_start=0.2，每 batch step | 无 | 10 epoch，patience=10 |
| TimeCMA | AdamW，MSE，lr=1e-4，wd=1e-3 | CosineAnnealingLR，T_max=50，eta_min=1e-6，每 epoch step | 5.0 | 100 epoch，patience=50，前 50% epoch 禁止早停 |

模型训练上限不同来自各自来源训练协议，不视为轻量化。适配层只替换数据输入与统一
评估，不重新发明模型内部结构。

每个 baseline profile 需要显式定义以下字段：optimizer、loss、base LR、weight decay、
scheduler、scheduler step 粒度、gradient clip、max epochs、early-stop patience 和
early-stop start epoch。未声明字段不得回落到统一的隐式默认值。

TimeCMA 的来源训练代码包含基于测试指标更新模型的路径；项目适配实现不得保留该行为，
只能依据验证 MSE 更新 `best.pt`。

## 4. GraphReportTS 训练配置

### 4.1 参数分组

DistilBERT backbone 冻结，并持续保持 eval 模式。可训练参数分为两类：

- Core group，lr=1e-3：GraphMapEncoder、InterCycleTemporalEncoder、
  NumericHistoryEncoder、context fuser、context norm 和 relative-step decoder。
- Semantic group，lr=3e-4：HF text projection、GatedSemanticFusion、gate，以及存在时的
  CrossModalFusion。

优化器使用 AdamW。LayerNorm 参数、bias 和 embedding 不做 weight decay；其余参数使用
`weight_decay=1e-4`。全局 gradient clipping 保持 1.0。

### 4.2 学习率与损失调度

- 最大训练轮数为 80。
- 前 5 epoch 对全部参数组执行线性 LR warmup，从目标 LR 的 10% 升至 100%。
- warmup 后使用 ReduceLROnPlateau，监控验证 MSE，`factor=0.5`、`patience=5`。
- Core group 最低 LR 为 1e-5；Semantic group 最低 LR 为 3e-6。
- 主回归损失保持 SmoothL1，checkpoint 选择指标保持验证 MSE。
- align loss 在前 5 epoch 为 0；第 6 至 15 epoch 线性升至 `w_align=0.001`，之后保持。
- 文本 gate 从第 1 epoch 起由回归目标学习；align warmup 不冻结 gate。

该顺序让图、历史和 decoder 先形成可用预测，再逐步引入语义对齐，避免 InfoNCE 在训练
初期压过量级较小的回归损失。

### 4.3 早停与 checkpoint

- 第 20 epoch 之前禁止早停。
- 第 20 epoch 起把 stale counter 从 0 开始计数，使用 patience=20，`min_delta=0`。
- 任意真实降低的验证 MSE 都更新 `best.pt`；早停判断和最佳 checkpoint 保存不得因绝对
  `1e-5` 阈值而忽略小幅改进。
- `last.pt` 必须保存 model、optimizer、scheduler、当前 epoch、各参数组 LR、最佳验证
  MSE 和 stale counter，保证恢复训练时策略连续。
- `best.pt` 只由验证集决定，测试集仅在训练结束后评估一次。

每个 epoch 的 `epoch_history.jsonl` 增加 core LR、semantic LR、align weight、回归损失、
align loss、验证 MSE/MAE/RMSE 和 gate mean/std/min/max。

## 5. 消融训练一致性

所有 battery ablation 沿用主模型的优化器、参数分组、LR warmup、plateau scheduler、
align warmup、早停和 checkpoint 规则。某个消融删除模块时，仅移除对应参数，不改变其余
模块的训练策略。

`no_align_loss`、`no_semantic_alignment`、`no_report_prompt` 和 `no_cross_modal` 的 align
weight 始终为 0；其它消融仍执行相同的 align warmup。这样消融差异只来自被移除的模型
部分，而不是训练配置变化。

## 6. 管线顺序与结果隔离

正式顺序调整为：

1. GraphReportTS 主模型：MIT、CALCE、XJTU。
2. Official baselines：每个数据集依次执行六个模型。
3. Battery ablations：MIT、CALCE、XJTU 的全部既定变体。

新管线使用固定输出根目录 `runs/full_hf_v3_training_strategy_nosoh`。旧目录
`runs/full_hf_v2_nosoh` 保留为旧训练协议参考，但其中指标不得进入新的汇总表。

主管线保持 fail-fast：任一正式训练失败时停止后续阶段。重新启动时只跳过存在完整
`test_metrics.json` 且 `run_config.json` 的训练策略版本匹配当前版本的任务；不允许仅凭
目录存在就跳过。

## 7. 验证方案

- 单元测试每个 baseline profile 的 optimizer、loss、scheduler、epoch 和 patience。
- 单元测试 OneCycle 按 batch step、其它 scheduler 按其来源语义 step。
- 单元测试主模型两类参数组互斥且覆盖全部可训练参数，DistilBERT backbone 无梯度。
- 单元测试 warmup、plateau 降 LR、align warmup、early-stop start epoch 和恢复状态。
- 单元测试 `best.pt` 只依据验证 MSE，且测试指标不参与选模。
- dry-run 验证管线顺序为 main -> baselines -> ablations。
- 远端正式启动前执行一个 batch 的前向、反向和 scheduler 状态检查；该检查不产出正式
  指标，也不上传为独立 smoke 管线。

## 8. 完成标准

- 本地和远端代码一致，远端工作树对应同一 Git commit。
- README 与工作报告记录新训练策略、管线顺序和旧结果失效原因。
- 新输出目录为空或只包含本次策略生成的结果。
- 首个远端主模型进程参数、LR、align weight、冻结参数和 GPU 状态均通过核验。
- 三阶段管线在主模型优先顺序下持续运行，且日志能区分每个训练 profile。
