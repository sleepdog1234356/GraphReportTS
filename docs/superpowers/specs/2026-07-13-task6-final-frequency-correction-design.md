# Task 6 Final Frequency Correction Design

## Goal and pinned-source rule

Correct the iTransformer and TimesNet formal marker contract to reproduce the
behavior of the pinned scripts, even where that behavior differs from each
dataset's factual sampling cadence. At iTransformer `c2426e6`, `run.py:29-30`
defaults `freq='h'`; at TimesNet `4e938a1`, `run.py:32-33` does the same. Neither
repository's formal ETT/ECL/Weather scripts supplies `--freq`. Their data
loaders pass the retained default to `time_features` at iTransformer
`data_provider/data_loader.py:73-75` and TimesNet
`data_provider/data_loader.py:89-91`.

## Runtime behavior

iTransformer and TimesNet profiles therefore retain source-native `freq='h'`
for all six datasets. Their Task 3 timestamp collation produces the four hourly
`timeF` columns for every formal dataset and passes those encoder and decoder
markers through the existing official forward arguments. The adapter must not
replace this value with a dataset-derived `t` value.

This source-native quirk is not a project protocol override. Dataset schemas
continue to validate factual cadence, and TimeCMA/Time-LLM prompts continue to
describe ETTm1/ETTm2 and Weather with their factual minute cadence.

## Tests and documentation

RED tests cover ETTm1, ETTm2, and Weather profiles and executable fake-source
build/forward paths for both models. They require `freq='h'`, four marker
columns, and marker delivery to the official arguments. Existing factual-cadence
prompt tests remain unchanged. The source audit, original review-fix design,
profile evidence, and Task 6 report will explicitly record the pinned behavior.

The executable trainer intentionally imports core NumPy and PyTorch. Official
external repositories, Transformers, model weights, and CUDA initialization
remain lazy. No training, CUDA use, weight access, or server mutation is in
scope.
