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

## 2026-07-13 review-fix pass

Design amendment: `e99a82b` (`Document Task 6 review fixes`). Fix implementation:
`55c2687` (`Fix Task 6 source adapter contracts`). No training, CUDA, weights,
real embedding generation, or network model access occurred during the fix pass.

### Full source identities and checkout state

The pinned server repositories were re-read with full `git rev-parse HEAD`:

| Baseline | Full SHA |
| --- | --- |
| PatchTST | `204c21efe0b39603ad6e2ca640ef5896646ab1a9` |
| iTransformer | `c2426e68ca13f74aaec08045c5c724d8ad328124` |
| TimeCMA | `223e4ae9364bec3e3a2d8bb39ab6eed2cf510296` |
| TimesNet | `4e938a1767106324dd753b2a44832bf870a0252e` |
| DLinear | `0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6` |
| Time-LLM | `b13e881f86cd0475ce1b72c17110430663334955` |

The server DLinear tree reports two tracked deleted Pyraformer bytecode files.
It remained read-only and was used only for inspection. The corrected formal
validator rejects any such tracked-dirty checkout, resolves the pinned manifest
revision to a full SHA, and compares that identity exactly to full `HEAD`.

### Review RED evidence

The complete review tests were written before production changes:

```text
C:\Python313\python.exe -m unittest tests.test_general_baselines -v
Ran 34 tests in 4.219s
FAILED (failures=6, errors=49, skipped=1)
```

The 49 subtest errors were the 48 iTransformer/TimesNet dataset-horizon marker
contracts plus the missing Time-LLM runtime-result argument. Focused failures
also proved permanent fake Transformers descriptor mutation, absent full checkout
validation, pre-gate stale reset, absent clipping/optimizer helper, and absent
Task 3 marker conversion/collation.

An additional source-frequency RED caught profile architecture overwriting the
dataset cadence after marker forwarding was implemented:

```text
C:\Python313\python.exe -m unittest \
  tests.test_general_baselines.GeneralBaselineAdapterTests.test_fake_official_classes_receive_source_configs_and_return_m2m_outputs -v
Ran 1 test
FAILED (failures=24)
```

Those 24 subtest failures were every minute-frequency iTransformer/TimesNet
combination still receiving config `freq='h'` instead of `freq='t'`.

Self-review then strengthened the scoped-loader fake to inherit
`from_pretrained`, matching real Hugging Face class structure. The focused RED
failed with `ValueError: LlamaModel has no direct from_pretrained descriptor`.
The context now records whether each class originally owned the descriptor and
uses `delattr` on exit for inherited methods, restoring exact lookup semantics.
The same focused test then passed, followed by the complete focused suite.

### Review GREEN evidence

```text
C:\Python313\python.exe -m unittest tests.test_general_baselines -v
Ran 34 tests in 5.484s
OK (skipped=1)
```

The optional local-real-repository integration remains skipped because no local
official clones are installed.

```text
C:\Python313\python.exe -m unittest discover -s tests -v
Ran 157 tests in 36.289s
OK (skipped=2)
```

The second skip remains the existing opt-in CUDA ECL smoke test.

After the inherited-descriptor self-review fix, final re-verification was:

```text
C:\Python313\python.exe -m unittest tests.test_general_baselines -q
Ran 34 tests in 5.539s
OK (skipped=1)

C:\Python313\python.exe -m unittest discover -s tests -q
Ran 157 tests in 35.175s
OK (skipped=2)
```

### Findings closed

1. Task 3 timestamps now become exact THUML `timeF` markers: four hourly
   dimensions for ETTh1/ETTh2/ECL and five minute dimensions for
   ETTm1/ETTm2/Weather. Baseline collation preserves encoder and target marks;
   the shared batch-forward path passes both to iTransformer and TimesNet, and
   adapters reject missing/malformed encoder marks.
2. Time-LLM loader redirection is scoped to one constructor. Each exact original
   class descriptor is restored in `finally`; sequential fake builds prove
   distinct paths/revisions and no global mutation or cross-contamination.
3. TimeCMA stale failures now accumulate before the delayed gate. The gate
   suppresses only `should_stop`; improvements still reset stale state.
4. Checkout validation compares full resolved identities, rejects tracked dirt,
   ignores only untracked files as specified, and attaches full SHA provenance.
5. Time-LLM adapter provenance now records resolved model/tokenizer paths,
   revisions, requested execution precision, and observed backbone dtype. The
   formal result schema rejects placeholders and requires/emits this provenance.
6. Import safety is now stated precisely: `train_general_baselines` eagerly
   imports NumPy and PyTorch for executable optimizer/scheduler helpers, but a
   subprocess test proves no Transformers, official `models`/`model`/`layers`,
   weights, or initialized CUDA context appears at import.

The optimizer mechanics are executable: the helper applies profile gradient
clipping, performs the optimizer step, and then performs source batch scheduling;
the one-based epoch scheduler helper remains covered. Existing battery behavior
remains unchanged and the full battery/general suite is green.
