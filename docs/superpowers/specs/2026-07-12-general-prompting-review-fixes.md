# General Prompting Review Fixes

## Goal

Correct tied-trend variable selection and make general prompt audit metadata match the 192-token text-encoder limit without changing battery prompt behavior.

## Selection semantics

For more than 12 variables, rank columns by `(absolute_trend, canonical_index)` to select the six smallest. Select the six largest from the remaining columns, ranking by `(-absolute_trend, canonical_index)`. Render selected summaries in canonical index order. This makes the groups disjoint even when all trends tie, so exactly 12 distinct variables are emitted.

## Prompt audit

The data prompt's deterministic fallback uses a 192 whitespace-token budget and only the fields `pretoken_word_count`, `pretoken_word_budget`, and `pretoken_word_truncated`. It is not a model tokenizer result.

`HFTextEncoder` uses its existing initialized tokenizer to calculate untruncated tokenizer counts and truncation against `max_length`. `SimpleTextEncoder` reports its corresponding split-token behavior. General models expose optional per-prompt `prompt_audit`; evaluation aggregates `encoder_token_count_mean`, `encoder_truncated_count`, `encoder_truncated_rate`, and `encoder_token_limit` into final JSON metrics. Battery outputs are unchanged because audit is requested only for general models.

## Tests

Tests first demonstrate failed tied-trend distinctness, stale 256 metadata, fallback truncation, actual simple-encoder truncation, tokenizer-helper truncation, general-model audit output, and evaluation metric aggregation. Verification is CPU-only.
