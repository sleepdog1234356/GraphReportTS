# Task 6 Report: Source-Consistent General Baseline Adapters

## Status and commits

Complete.

- Design record: `efb4c41` (`Document general baseline adapter design`)
- Implementation: `dfffe59` (`Add official general forecasting baselines`)
- Branch: `codex/general-forecasting-v1`

No training, CUDA work, model download, real embedding generation, or server
mutation was performed.

## Direct pinned-source audit

The six read-only server clones were inspected over SSH at
`connect.westc.seetacloud.com:/root/autodl-tmp/GraphReportTS/external`.
`git rev-parse --short HEAD` returned:

| Repository | Audited commit | Official class evidence | Training/prompt evidence |
| --- | --- | --- | --- |
| PatchTST | `204c21e` | `PatchTST_supervised/models/PatchTST.py:15-92`, `Model` | Dataset scripts under `PatchTST_supervised/scripts/PatchTST`: ETTh1/2 lines 8-42, ETTm1/2 lines 8-45, electricity lines 8-45, weather lines 8-43; defaults `run_longExp.py:78-85`; Adam/MSE and scheduling `exp/exp_main.py:47-52,112-124,193-212` |
| iTransformer | `c2426e6` | `model/iTransformer.py:10-80`, `Model` | `scripts/multivariate_forecasting/ETT/iTransformer_*.sh:13-78`, `ECL/iTransformer.sh:13-88`, `Weather/iTransformer.sh:13-81`; defaults `run.py:43-68`; optimizer/loss `experiments/exp_long_term_forecasting.py:32-37,94-177` |
| TimeCMA | `223e4ae` | `models/TimeCMA.py:6-102`, `Dual` | Horizon blocks in `scripts/{ETTm1,ETTm2,ETTh1,ETTh2,ECL,Weather}.sh`; AdamW/cosine/clip `train.py:51-90`; checkpoint/stop logic `train.py:233-299`; exact prompt/final-token extraction `storage/gen_prompt_emb.py:28-100` |
| TimesNet | `4e938a1` | `models/TimesNet.py:71-128,201-213`, `Model` | Dataset scripts under `scripts/long_term_forecast/{ETT,ECL,Weather}_script`; defaults `run.py:57,90-96`; Adam/MSE/type1 `exp/exp_long_term_forecasting.py:34-39,88-161` and `utils/tools.py:12-29` |
| DLinear | `0c11366` | `models/DLinear.py:38-87`, `Model` | `scripts/EXP-LongForecasting/Linear/{etth1,etth2,ettm1,ettm2,electricity,weather}.sh:9-66`; defaults `run_longExp.py:66-72`; Adam/MSE/type1 `exp/exp_main.py:46-51,112-205` |
| Time-LLM | `b13e881` | `models/TimeLLM.py:30-264`, `Model` | `scripts/TimeLLM_{ETTm1,ETTm2,ETTh1,ETTh2,ECL,Weather}.sh`; defaults `run_main.py:55-99`; Adam/scheduler `run_main.py:145-164,236-264`; exact normalized prompt `models/TimeLLM.py:200-264`; descriptions `dataset/prompt_bank/{ETT,ECL,Weather}.txt` |

The detailed evidence and exact exceptions are retained in
`docs/general_forecasting_source_audit.md`.

### Corrections produced by the binding audit

- PatchTST is not universally OneCycle/patience 20. ETTm1/2 use OneCycle with
  `pct_start=0.4`; ECL uses OneCycle with `pct_start=0.2`; ETTh1/2 use the
  source-default type3 epoch schedule and patience 100; Weather uses type3 and
  patience 20.
- TimeCMA is not universally 100 epochs. The pinned ETT and ECL scripts request
  999 epochs; Weather requests 20 for H96 and 100 for H192/H336/H720. Its stop
  gate is `epochs//2`, with patience 50.
- TimesNet retains the source script epoch exceptions: ETTm1-H336=3;
  ETTm2-H192/H720=1; Weather-H192/H720=1.
- DLinear retains source dataset/horizon learning rates, including ETTh2=0.05
  and ETTm2-H336/H720=0.01.
- Time-LLM retains per-run type1, OneCycle, or cosine behavior and exact epoch
  exceptions. ETTh1-H336 uses cosine (`T_max=20`, `eta_min=1e-8`), while the
  source TST runs use batch OneCycle with `pct_start=0.2`.

## RED evidence

The first command with the shell-default Python 3.14 stopped before collection
because that interpreter has no PyTorch. Root-cause inspection found the project
test environment at `C:\Python313\python.exe` with PyTorch `2.6.0+cu124`; all
recorded RED/GREEN evidence therefore uses that interpreter.

Initial RED:

```text
C:\Python313\python.exe -m unittest tests.test_general_baselines -v
Ran 24 tests
FAILED (failures=23)
```

The failures explicitly reported the missing profile resolver, general adapter
builder, prompt/cache APIs, and training contract. The pre-existing lightweight
module-import behavior was the one passing assertion.

Additional focused RED cycles were recorded before their implementations:

- Seven training-contract failures before `train_general_baselines.py` existed.
- Exact Time-LLM source normalization failed against the initially unnormalized
  pure prompt helper, then passed after matching `layers/StandardNorm.py:36-55`.
- Frozen GPT-2 final-token encoder contract failed before
  `FrozenGPT2PromptEncoder` was added.

## GREEN evidence

Focused suite:

```text
C:\Python313\python.exe -m unittest tests.test_general_baselines -v
Ran 27 tests in 2.288s
OK (skipped=1)
```

The skipped check is the optional real-local-repository commit integration; the
workspace has no local official clones. Fake official source trees cover all 144
model/dataset/horizon combinations on CPU, including ECL's 321 output channels.

Full suite:

```text
C:\Python313\python.exe -m unittest discover -s tests -v
Ran 150 tests in 33.669s
OK (skipped=2)
```

The second skip is the existing opt-in CUDA ECL smoke test. `compileall` and
`git diff --check` also passed.

## Implemented contracts

- Immutable source identity, architecture, prompt policy, protocol override, and
  training mechanics for all 144 formal combinations.
- Lazy isolated imports of the six exact official classes, with local checkout
  commit validation for formal builds and fake-tree bypass only when explicitly
  requested by tests.
- Uniform `[B,36,C] -> [B,H,C]` M2M wrapper with no output-channel slicing.
- No decoder history for APIs that do not consume it. TimesNet receives only a
  zero constructor compatibility value; no model receives `label_len=18`.
- Exact TimeCMA scaled-value/timestamp/trend prompt, frozen local GPT-2
  final-token encoder, provenance-digested disk cache, absolute sample/variable
  keys, and fail-closed future-boundary checks.
- Exact Time-LLM RevIN normalization, per-variable min/max/median/trend/top-five
  FFT-autocorrelation-lag prompt, source prompt-bank descriptions, local-only
  official backbone loading, frozen backbone parameters, and recorded bf16
  profile.
- Task 3 train/validation/test dataset construction with one train-fitted scaler,
  stable scaler checksum, source-native optimizer/scheduler helpers,
  validation-MSE-only checkpoint decisions, and standardized-space MSE/MAE
  result records.

## Files

- Added `bstalignment/general_baseline_profiles.py`
- Added `bstalignment/train_general_baselines.py`
- Modified `bstalignment/baseline_adapters.py` additively; existing battery setup
  functions and source lists are unchanged
- Added `tests/test_general_baselines.py`
- Corrected `docs/general_forecasting_source_audit.md`
- Added the approved design and implementation plan under `docs/superpowers`

## Self-review and concerns

- No official/model/Transformers dependency is imported at ordinary
  `baseline_adapters` module import. Transformers is imported only by the
  explicit local GPT-2 constructor or the official Time-LLM source build.
- Neither prompt implementation imports or calls GraphReportTS prompting.
- Cache records contain only observed indices `start..forecast_origin-1`; a
  record at or beyond the origin, partial entry, changed provenance, changed
  scaler checksum, or changed boundary is rejected.
- Existing battery tests remain green and no battery trainer/model/profile file
  was modified.
- Real official construction was not executed because local clones and text
  weights are absent. The fake-tree contracts and direct server audit cover the
  interfaces, but deployment readiness still requires a CPU/local-clone smoke
  with locally provisioned weights.
- The pinned Time-LLM repository identifies `huggyllama/llama-7b` but does not
  pin a Hugging Face model or tokenizer revision. Formal construction therefore
  requires caller-supplied local paths and explicit revision strings and refuses
  network fallback; this provenance must be supplied before a real run.
- The source TimeCMA loop can use test MSE in checkpoint updates after epoch 10.
  That behavior is deliberately not retained: the formal shared contract uses
  validation MSE only, as required by the project protocol.
