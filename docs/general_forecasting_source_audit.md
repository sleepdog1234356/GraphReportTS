# General Forecasting Source Audit

This audit freezes the official repositories and commits used for the six
baseline adapters.  The immutable records are in
`configs/general_forecasting/models.yaml`; raw-data checksums are in
`configs/general_forecasting/datasets.yaml`.  The project deliberately does
not vendor external source trees.  These `.yaml` files use JSON syntax, which
is a YAML-compatible subset, so validation does not depend on an installed
YAML parser.

## Task 6 direct-source re-audit

Task 6 re-checked every adapter path and profile value over SSH against the
read-only clones at
`/root/autodl-tmp/GraphReportTS/external/{patchtst,itransformer,timecma,timesnet,dlinear,time_llm}`.
Their checked-out short commits are respectively `204c21e`, `c2426e6`,
`223e4ae`, `4e938a1`, `0c11366`, and `b13e881`. Dataset scripts, rather than a
single generalized default, govern dataset/horizon architecture, learning rate,
epoch, and scheduler exceptions.

The corresponding full SHAs re-read on 2026-07-13 are:

| Baseline | Full SHA |
| --- | --- |
| PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` |
| iTransformer | `c2426e68ca13f74aaec08045c5c724d8ad328124` |
| TimeCMA | `223e4ae9364bec3e3a2d8bb39ab6eed2cf510296` |
| TimesNet | `4e938a1767106324dd753b2a44832bf870a0252e` |
| DLinear | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` |
| Time-LLM | `b13e881f86cd0475ce1b72c17110430663334955` |

Formal local validation resolves the manifest revision and `HEAD` to full SHAs,
requires exact equality, and rejects output from
`git status --porcelain --untracked-files=no`. The read-only server DLinear clone
currently reports two tracked deleted `Pyraformer/utils/__pycache__/*.pyc` files;
it was used only for read-only source inspection and would correctly be rejected
as a formal runtime checkout until its owner restores those tracked files.

This direct audit supersedes two earlier generalizations in this document:
PatchTST is not universally OneCycle/patience 20, and TimeCMA is not universally
100 epochs. The exact exceptions below are represented by
`resolve_general_profile` and locked by `tests/test_general_baselines.py`.

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
| PatchTST | `yuqinie98/PatchTST` @ `204c21e` | `PatchTST_supervised/models/PatchTST.py:15-92` (`Model`) | Adam + MSE. ETTm1/2: batch OneCycle, pct_start 0.4, 100 epochs, patience 20. ECL: batch OneCycle, pct_start 0.2, 100 epochs, patience 10. ETTh1/2: source-default type3 epoch decay, 100 epochs, patience 100. Weather: type3, 100 epochs, patience 20. | None |
| iTransformer | `thuml/iTransformer` @ `c2426e6` | `model/iTransformer.py:10-80` (`Model`) | Adam + MSE, type1 epoch decay, 10 epochs, patience 3. ECL uses lr 5e-4/batch 16; the other formal datasets use lr 1e-4/batch 32. | None |
| TimeCMA | `ChenxiLiu-HNU/TimeCMA` @ `223e4ae` | `models/TimeCMA.py:6-102` (`Dual`) | AdamW + MSE, wd 1e-3, epoch cosine with `T_max=min(epochs,50)`, eta_min 1e-6, clip 5.0, patience 50, stop gate at `epochs//2`. ETT/ECL scripts request 999 epochs. Weather requests 20 at H96 and 100 otherwise. | Exact per-variable timestamp/value/trend template in `storage/gen_prompt_emb.py:28-100` |
| TimesNet | `thuml/Time-Series-Library` @ `4e938a1` | `models/TimesNet.py:71-128,201-213` (`Model`) | Adam + MSE, type1 epoch decay, lr 1e-4, patience 3. Default 10 epochs except ETTm1-H336=3 and ETTm2-H192/H720 plus Weather-H192/H720=1. | None |
| DLinear | `cure-lab/LTSF-Linear` @ `0c11366` | `models/DLinear.py:38-87` (`Model`) | Adam + MSE, type1 epoch decay, 10 epochs, patience 3. Dataset/horizon script learning rates range from 1e-4 to 5e-2 and are retained exactly. | None |
| Time-LLM | `KimMeen/Time-LLM` @ `b13e881` | `models/TimeLLM.py:30-264` (`Model`) | Adam + MSE, patience 10. Source scripts select type1, batch OneCycle (`pct_start=0.2`), or cosine (`T_max=20`, eta_min 1e-8) by dataset/horizon, with 10-100 epochs (Weather-H720=15; ETTh2-H720=20). | Exact normalized per-variable min/max/median/trend/top-five-lags prompt at `models/TimeLLM.py:200-264` |

For every adapter, source-preserved fields include model architecture,
source-native loss, optimizer, scheduler, early-stopping mechanics, and (for
TimeCMA and Time-LLM) source prompt construction and frozen language-model
path. The protocol overrides above are recorded with every run so a source
update or a changed comparison contract cannot silently alter a formal table.

## Exact script evidence

- PatchTST dataset architecture and scheduler flags: `PatchTST_supervised/scripts/PatchTST/etth1.sh:8-42`, `etth2.sh:8-42`, `ettm1.sh:8-45`, `ettm2.sh:8-45`, `electricity.sh:8-45`, and `weather.sh:8-43`. Defaults are `run_longExp.py:78-85`; Adam/MSE, OneCycle construction, batch stepping, and non-TST epoch adjustment are `exp/exp_main.py:47-52,112-124,193-212`.
- iTransformer architecture: `scripts/multivariate_forecasting/ETT/iTransformer_ETTh1.sh:13-78`, corresponding ETTh2/ETTm1/ETTm2 scripts at the same blocks, `ECL/iTransformer.sh:13-88`, and `Weather/iTransformer.sh:13-81`. Defaults are `run.py:43-68`; Adam/MSE and epoch adjustment are `experiments/exp_long_term_forecasting.py:32-37,94-177`.
- TimeCMA dataset/horizon settings: `scripts/ETTm1.sh:11-99`, `ETTm2.sh:11-96`, `ETTh1.sh:11-99`, `ETTh2.sh:11-96`, `ECL.sh:11-96`, and `Weather.sh:11-100`. Optimizer, cosine, and clipping are `train.py:51-90`; validation/test checkpoint behavior and delayed stop are `train.py:233-299`. The adapter intentionally replaces the source's test-dependent save branch with validation-MSE-only selection.
- TimesNet architecture/epoch overrides: `scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh:14-102`, corresponding ETTh2/ETTm1/ETTm2 scripts, `ECL_script/TimesNet.sh:14-97`, and `Weather_script/TimesNet.sh:14-102`. Defaults are `run.py:57,90-96`; Adam/MSE and type1 stepping are `exp/exp_long_term_forecasting.py:34-39,88-161` and `utils/tools.py:12-29`.
- DLinear architecture and learning rates: `models/DLinear.py:38-87` and `scripts/EXP-LongForecasting/Linear/{etth1,etth2,ettm1,ettm2,electricity,weather}.sh:9-66`. Defaults are `run_longExp.py:66-72`; Adam/MSE and type1 stepping are `exp/exp_main.py:46-51,112-205`.
- Time-LLM architecture/training: `scripts/TimeLLM_ETTm1.sh:2-124`, `TimeLLM_ETTm2.sh:2-121`, `TimeLLM_ETTh1.sh:2-116`, `TimeLLM_ETTh2.sh:2-120`, `TimeLLM_ECL.sh:2-107`, and `TimeLLM_Weather.sh:2-115`. Defaults and frozen-backbone construction are `run_main.py:55-99` and `models/TimeLLM.py:43-164`; optimizer/scheduler stepping is `run_main.py:145-164,236-264`. Exact descriptions are `dataset/prompt_bank/{ETT,ECL,Weather}.txt`.

## Length-36 and text-model notes

PatchTST and Time-LLM retain source `patch_len=16` and `stride=8`; both remain
valid for a 36-step input, so no patch adjustment is made. iTransformer and
Time-LLM accept decoder arguments but do not consume decoder history in their
forecast implementations (`iTransformer.py:42-80`, `TimeLLM.py:194-255`).
TimesNet stores `label_len` but its long-term forecast uses only encoder values
and marks (`TimesNet.py:80-128`). Consequently no general adapter receives
`label_len=18`; zero is supplied only as a constructor compatibility field for
TimesNet. The Time-LLM source identifies `huggyllama/llama-7b` but does not pin a
model/tokenizer revision, so formal construction requires caller-supplied local
paths and explicit revision provenance and forces local-only loading. Its source
scripts use bf16 mixed precision; the profile records `bf16`.

## Task 6 review-fix evidence

iTransformer and TimesNet both use the shared THUML `time_features` implementation
(`utils/timefeatures.py:34-148`). Their data loaders transpose those features to
`[time, feature]` and return both encoder and decoder markers
(`iTransformer/data_provider/data_loader.py:74-90,164-180,262-278` and
`TimesNet/data_provider/data_loader.py:90-110,192-212,302-322`). The project
baseline collator now reproduces the four hourly columns or five minute columns
from Task 3 timestamps and carries them to the official forward arguments.

Time-LLM local-path redirection is constructor-scoped. Exact original
`from_pretrained` descriptors are restored in `finally`; runtime provenance
records resolved paths, revisions, requested precision, and observed backbone
dtype. TimeCMA early-stop failures accumulate continuously, matching
`train.py:289-299`, while the half-epoch condition gates only the break action.
