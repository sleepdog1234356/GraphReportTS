# Task 6 Review Fixes Design

## Goal

Close all Task 6 review findings without changing the six pinned source
identities, battery behavior, or the no-training/no-CUDA/no-weights scope.

## Timestamp flow

Task 3 remains the source of split windows, scaling, and raw timestamps. A
general-baseline collation helper converts those timestamps to the exact THUML
`timeF` representation audited in both iTransformer and TimesNet
`utils/timefeatures.py:34-148`: hourly datasets use hour/day-of-week/day-of-month/
day-of-year (four columns), while ETTm1, ETTm2, and Weather add minute-of-hour
(five columns). Encoder and target markers are retained in the batch and passed
through one general-baseline forward helper. iTransformer and TimesNet reject a
missing encoder marker instead of silently receiving `None`; available target
markers are passed as the official decoder-marker argument.

## Scoped Time-LLM loading

Time-LLM's official constructor keeps using its source `from_pretrained` calls,
but path redirection exists only inside a context manager. Before patching, the
context captures each class's exact `from_pretrained` descriptor from
`vars(cls)`. Its `finally` block restores those same descriptor objects even if
construction fails. Sequential builds may therefore use different local paths
and revisions without cross-contamination or permanent Transformers mutation.

## Checkout and runtime provenance

Formal source validation resolves `HEAD` and the manifest's pinned revision to
full commit SHAs and requires exact full-SHA equality. It also runs
`git status --porcelain --untracked-files=no` and rejects any tracked change;
untracked files are deliberately outside this source-integrity check. The full
SHA is attached to adapter runtime provenance.

Time-LLM runtime provenance contains resolved absolute model and tokenizer
paths, explicit model and tokenizer revisions, explicit execution precision,
and the constructed backbone parameter dtype. Formal construction rejects
missing values, placeholders, nonexistent local paths, and unsupported
precision. The shared result record requires and emits this runtime provenance
for Time-LLM.

## Training mechanics

Validation failures increment stale state before and after TimeCMA's delayed
gate. The gate suppresses only the stop action; an improvement still resets the
counter. A general optimizer-step helper applies profile gradient clipping,
calls `optimizer.step()`, and then applies batch-stepped source scheduling.
Epoch scheduler behavior remains in the existing one-based helper.

## Import-safety boundary

`train_general_baselines` eagerly imports NumPy and PyTorch because it is an
executable training-contract module. Import safety means it imports no official
external source package, Transformers module, model weights, or CUDA context.
Tests assert that precise boundary instead of claiming the trainer has no heavy
core numerical imports.

## Tests and scope

Each finding receives a focused failing test before its implementation. Fake
official modules reject missing markers and expose fake Llama loader classes for
two sequential path/revision builds. Temporary Git repositories exercise full
SHA, wrong revision, tracked-dirty, and untracked-file behavior. Chained epoch,
runtime provenance, result schema, gradient clipping, scheduler, marker
collation, and import-safety tests complete the contracts. No training, CUDA,
weights, network access, or real embedding generation is performed.
