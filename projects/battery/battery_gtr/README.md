# BatteryGTR（电池预测主体）

电池主体保持原版v2：最近32轮的58维适配特征进入原版patch稀疏图，冻结
DistilBERT处理电池专用prompt，`BatterySOHHead`直接预测未来20步SOH。主体不含
linear/Ridge anchor，也不采用`series_context_decomp`。

稳定入口：

```bash
python -m bstalignment.battery.train_battery_gtr \
  --dataset mit --cache_dir data/battery/cache/features/mit \
  --output artifacts/battery/battery_gtr/runs
```

双GPU入口为`projects/battery/battery_gtr/run_matrix.sh`。本公开目录仅包含论文主体；对比模型、消融和优化探索代码不在此发布树中。

`GraphReportTS-v2` 是该电池主体的历史名称。旧 checkpoint 的参数键、旧结果中的模型名和旧产物路径均保持不变；新运行统一写入 BatteryGTR 目录。
