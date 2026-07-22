# BatteryGTR（电池预测主体）

BatteryGTR 使用最近32轮的58维适配特征进入多尺度 patch 稀疏图，冻结
DistilBERT处理电池专用prompt，`BatterySOHHead`直接预测未来20步SOH。主体不含
linear/Ridge anchor，也不采用`series_context_decomp`。

稳定入口：

```bash
python -m anchoredgtr.battery.train_battery_gtr \
  --dataset mit --cache_dir data/battery/cache/features/mit \
  --output artifacts/battery/battery_gtr/runs
```

双GPU入口为`projects/battery/battery_gtr/run_matrix.sh`。本公开目录仅包含论文主体；对比模型、消融和优化探索代码不在此发布树中。

新运行统一使用 BatteryGTR 模型名并写入 BatteryGTR 目录；参数层名称保持稳定，可严格加载已有主体 checkpoint。
