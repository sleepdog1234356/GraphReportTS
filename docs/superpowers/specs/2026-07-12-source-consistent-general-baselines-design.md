# Source-Consistent General Baseline Adapters Design

## Goal

Add six general-forecasting baseline adapters that use the official pinned
implementations, the shared Task 3 data protocol, multivariate-to-multivariate
outputs, and source-native training mechanics without changing battery behavior.

## Binding source-audit rule

Every external module/class path and every optimizer, scheduler, learning-rate,
epoch, patience, and architecture value must be verified directly against the
pinned read-only server clone before it is encoded. The implementation must
follow the source when it differs from an earlier audit or design assumption.
The Task 6 report records the supporting source file and line or script lines.

## Components

`bstalignment/general_baseline_profiles.py` owns immutable source identities,
prompt policies, dataset/horizon architecture settings, protocol overrides, and
complete source-native training profiles. It provides
`resolve_general_profile(name, dataset, pred_len)`.

`bstalignment/baseline_adapters.py` keeps existing battery setup behavior and
adds lazy, import-isolated official model construction through
`build_general_baseline(name, dataset_meta, args)`. Every wrapper accepts
`[B, 36, C]` and returns every channel as `[B, H, C]`.

`bstalignment/train_general_baselines.py` provides external-model-import-safe
orchestration over the Task 3 dataset, split, and train-only scaler. It eagerly
imports NumPy and PyTorch because its optimizer/scheduler contracts are
executable, but imports no external source tree, Transformers package, weights,
or CUDA context. It applies the resolved
optimizer/scheduler/early-stop mechanics, selects checkpoints by validation MSE,
and emits one result/provenance schema. This task does not run training.

## Source loading

External repositories remain unvendored. Imports happen only during adapter
construction from a caller-supplied external root. A temporary import scope
isolates conflicting top-level packages such as `models`, `model`, and `layers`,
then restores the process module state. Formal construction resolves both the
pinned revision and `HEAD` to full SHAs, requires equality, and rejects tracked
dirty changes. Ordinary module import must not require an external repository,
Transformers, model weights, CUDA initialization, or network access.

The official source paths are not assumed by this design document: the direct
pinned-source audit determines and records the final paths before tests encode
their contracts.

## Model API policy

The formal protocol fixes `seq_len=36`, the requested prediction horizon,
`features=M`, and `enc_in=dec_in=c_out=C` where those configuration fields
exist. Dataset/horizon source architecture values remain unchanged unless they
are invalid with a 36-step history. Any required patch or stride adjustment is
explicit profile metadata. `label_len=18` is present only for an official API
that consumes decoder context; an unused four-argument signature alone does not
justify decoder-history leakage.

Task 3 timestamps are converted to the source `timeF` format before baseline
forwarding: ETTh1/ETTh2/ECL use four hourly columns and ETTm1/ETTm2/Weather use
five minute-frequency columns. iTransformer and TimesNet require encoder marks;
the shared batch path also passes target marks through the decoder-marker API.

## TimeCMA prompt cache

TimeCMA uses the exact official dataset cadence template, scaled observed values,
timestamp formatting, integer value rendering, summed-first-difference trend,
and frozen GPT-2 final-token embedding behavior. Cache provenance contains the
dataset, split, input length, source commit, scaler checksum, model/tokenizer
revision, and precision. Entries are keyed by absolute sample index and variable
index and record their observed interval. Construction and loading reject any
entry whose observed interval reaches the forecast origin or whose provenance
does not match. The implementation supports an injected fake encoder for CPU
tests and does not generate real embeddings in this task.

## Time-LLM prompt behavior

Time-LLM retains its official per-variable minimum, maximum, median, trend, and
top-five FFT-autocorrelation-lag prompt and the dataset-specific official prompt
bank description. The official selected backbone is frozen, and provenance
records absolute model/tokenizer paths, revisions, requested execution
precision, and observed backbone dtype. Loader redirection is scoped to one
constructor and restores exact original Transformers descriptors. No
GraphReportTS prompt function is imported or reused.

## Training profiles and evaluation

`resolve_general_profile` returns the source-derived architecture and training
contract for a model/dataset/horizon. It separately records source settings and
formal-protocol overrides so the shared 36-step input, data loader, scaler,
seeds, paths, and validation-only checkpoint rule cannot be mistaken for source
defaults. The shared evaluator reports standardized-space MSE and MAE over all
channels. TimeCMA stale failures accumulate before its delayed stop gate; only
the stop action is suppressed before the gate.

## Testing

Tests are written and run RED before implementation. Fake official source trees
exercise all 144 model/dataset/horizon combinations on CPU, lazy import behavior,
cross-repository isolation, all-channel shapes, source metadata, prompt policy,
training mechanics, and label-length policy. Additional tests cover exact
TimeCMA formatting/cache provenance/future-leakage rejection, exact Time-LLM
statistics/lags/descriptions/frozen backbone, and the shared result schema.
Integration against real repositories is optional and skipped when local clones
are absent. Existing battery and full-project tests remain green.

## Scope

This task implements contracts, adapters, profiles, orchestration, tests, and
audit documentation. It does not train, use CUDA, generate real embeddings,
download models, copy server repositories, or modify the read-only server.
