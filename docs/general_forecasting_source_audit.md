# General Forecasting Source Audit

This audit freezes the official repositories and commits used for the six
baseline adapters.  The immutable records are in
`configs/general_forecasting/models.yaml`; raw-data checksums are in
`configs/general_forecasting/datasets.yaml`.  The project deliberately does
not vendor external source trees.  These `.yaml` files use JSON syntax, which
is a YAML-compatible subset, so validation does not depend on an installed
YAML parser.

## Shared protocol overrides

All baseline adapters preserve the source implementation listed below, while
the formal protocol overrides only `seq_len=36`, `pred_len` (96, 192, 336, or
720), dataset/raw and canonical-data paths, M2M `features=M`, encoder/decoder
feature counts, smoke seed 42 or formal seeds 2021/2022/2023, and the run
output path. No adapter substitutes GraphReportTS prompts for an official baseline prompt. Checkpoints are
selected by validation MSE; the shared evaluator reports standardized-space
MSE and MAE.

| Baseline | Official source frozen in manifest | Source implementation and loader | Source-native optimization | Prompt |
| --- | --- | --- | --- | --- |
| PatchTST | `yuqinie98/PatchTST` @ `204c21e` | `PatchTST_supervised/models/PatchTST.py`; `PatchTST_supervised/data_provider/data_factory.py` | Adam + MSE, batch-stepped OneCycleLR, 100 epochs, patience 20 | None |
| iTransformer | `thuml/iTransformer` @ `c2426e6` | `model/iTransformer.py`; `data_provider/data_factory.py` | Adam + MSE, source type-1 epoch decay, 10 epochs, patience 3 | None |
| TimeCMA | `ChenxiLiu-HNU/TimeCMA` @ `223e4ae` | `models/TimeCMA.py`; `data_provider/data_factory.py` | AdamW + MSE, epoch cosine schedule, gradient clip 5.0, 100 epochs; early stopping begins at epoch 50 with patience 50 | Official per-variable timestamp/value/trend prompt in `TimeCMA.py` |
| TimesNet | `thuml/Time-Series-Library` @ `4e938a1` | `models/TimesNet.py`; `data_provider/data_factory.py` | Adam + MSE, source type-1 epoch decay, 10 epochs, patience 3 | None |
| DLinear | `cure-lab/LTSF-Linear` @ `0c11366` | `models/DLinear.py`; `data_provider/data_factory.py` | Adam + MSE, source type-1 epoch decay, 10 epochs, patience 3 | None |
| Time-LLM | `KimMeen/Time-LLM` @ `b13e881` | `models/TimeLLM.py`; `data_provider/data_factory.py` | Adam + MSE, batch-stepped OneCycleLR, 10 epochs, patience 10 | Official per-variable statistics and top-five-lag prompt in `TimeLLM.py` |

For every adapter, source-preserved fields include model architecture,
source-native loss, optimizer, scheduler, early-stopping mechanics, and (for
TimeCMA and Time-LLM) source prompt construction and frozen language-model
path.  The protocol overrides above are recorded with every run so a source
update or a changed comparison contract cannot silently alter a formal table.
