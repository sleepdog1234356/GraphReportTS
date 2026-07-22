# AnchoredGTR（通用预测主体）

论文主体使用最近36步完整数值历史和此前36步统计prompt，预测未来
24/36/48/60步。稳定入口是：

```bash
python -m anchoredgtr.general.train_anchored_gtr \
  --dataset ETTh2 --horizon 24 \
  --provenance-manifest artifacts/general/anchored_gtr/provenance.json
```

入口从`anchoredgtr.general.strategy_registry`读取经审计的逐单元策略。该注册表是历史最佳正式结果的事后汇总，不表示一次统一预注册实验。新结果统一写入模型名`AnchoredGTR`。

批量双GPU入口为`projects/general/anchored_gtr/run_matrix.sh`。数据默认位于`data/general`，DistilBERT默认位于`hf_models/distilbert-base-uncased`，大文件均保持在数据盘项目根目录内。
