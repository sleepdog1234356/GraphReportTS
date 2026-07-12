# Leakage-Free General Prompting Design

## Goal

Replace the shared report helper on the formal general-forecasting data path with a deterministic, leakage-free prompt built only from each sample's standardized observed history.

## Scope

The change applies only to `GeneralForecastGraphDataset`. Battery datasets and the shared `raw_signal.build_report_from_array` helper remain unchanged.

## Prompt construction

`build_general_prompt(history, columns, frequency, pred_len)` receives a `[36, C]` standardized history, canonical column names, a sampling interval, and one formal horizon. It emits the fixed generic template:

```text
Task: multivariate time-series forecasting.
Observation: 36 past steps sampled every {frequency}; {num_variables} variables are observed.
Window summary: aggregate mean={...}, standard deviation={...}, mean absolute change={...}, trend balance={up_count} increasing/{down_count} decreasing/{flat_count} approximately flat.
Variable summaries: {bounded deterministic summaries of variable name, last value, trend, volatility}.
Instruction: predict all {num_variables} variables for the next {pred_len} steps.
Use only the observed window.
```

All statistics use the provided history only. Float formatting is fixed. Aggregate metrics include every variable. For at most 12 columns, summaries retain canonical order; otherwise the builder selects six highest and six lowest absolute trends, using canonical index as the tie-breaker. The prompt never includes a dataset name, split, future value, battery-specific term, or dataset-specific forecast hint.

## Audit metadata and integration

The prompting module also returns deterministic audit metadata for its prompt: token count, configured token budget, and whether the bounded template required truncation. `data_general.py` attaches that metadata to every general sample and passes it through batch collation. The text-builder interface continues to return a string; metadata is obtained from its paired result API. Frequency comes from the formal dataset schema.

## Tests

Unit tests snapshot stable 7-, 21-, and 321-variable prompts; assert exact template fields, all-variable aggregates, bounded/deterministic selection, formal horizons, future-value invariance, and absence of prohibited vocabulary. Integration coverage verifies `GeneralForecastGraphDataset` chooses the new module, uses schema frequency, and includes audit metadata. Existing battery prompt tests remain untouched.
