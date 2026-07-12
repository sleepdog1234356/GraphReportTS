# Leakage-Free General Prompting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate auditable, deterministic, leakage-free prompts for the six formal general-forecasting datasets.

**Architecture:** A standalone prompting module owns template rendering, observed-history statistics, deterministic variable selection, and audit metadata. The general dataset calls it with train-standardized history and schema frequency; shared and battery prompt paths are not modified.

**Tech Stack:** Python 3, NumPy, PyTorch dataset adapter, `unittest`.

## Global Constraints

- Formal datasets are `ETTm1`, `ETTm2`, `ETTh1`, `ETTh2`, `ECL`, and `Weather`; Traffic is excluded.
- The history length is exactly 36, feature mode is `M`, and horizons are exactly 96, 192, 336, and 720.
- Prompts use standardized observed history only, exact canonical names, aggregates over every variable, and no more than 12 deterministic summaries.
- Do not alter battery prompting or shared prompt helpers; do not run training or CUDA.

---

### Task 1: Test and build the isolated prompt renderer

**Files:**
- Create: `tests/test_general_prompting.py`
- Create: `bstalignment/general_prompting.py`

**Interfaces:**
- Produces: `build_general_prompt(history, columns, frequency, pred_len) -> str`
- Produces: `build_general_prompt_result(history, columns, frequency, pred_len) -> GeneralPromptResult`

- [ ] **Step 1: Write the failing tests**

```python
def test_large_variable_selection_uses_trend_then_canonical_index():
    result = build_general_prompt(np.zeros((36, 321)), tuple(f"x{i}" for i in range(321)), "1 hour", 96)
    self.assertIn("x0", result)
    self.assertLessEqual(result.count("last="), 12)
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_general_prompting -v`

Expected: import failure because `bstalignment.general_prompting` does not exist.

- [ ] **Step 3: Add minimal renderer code**

```python
def build_general_prompt(history, columns, frequency, pred_len):
    return build_general_prompt_result(history, columns, frequency, pred_len).prompt
```

The result builder validates formal inputs, computes history-only metrics, renders the canonical template, and supplies deterministic audit metadata.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m unittest tests.test_general_prompting -v`

Expected: PASS.

### Task 2: Integrate the renderer only into general data

**Files:**
- Modify: `bstalignment/data_general.py`
- Modify: `tests/test_general_prompting.py`

**Interfaces:**
- Consumes: `build_general_prompt_result(history, columns, frequency, pred_len) -> GeneralPromptResult`
- Produces: General dataset samples with `prompt` and `prompt_metadata`.

- [ ] **Step 1: Write the failing integration test**

```python
sample = GeneralForecastGraphDataset("ECL", data_root=str(root), split="val", pred_len=96)[0]
self.assertEqual(sample["prompt_metadata"]["frequency"], "1 hour")
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m unittest tests.test_general_prompting -v`

Expected: FAIL because general dataset samples do not yet expose prompt metadata.

- [ ] **Step 3: Add minimal integration**

```python
result = build_general_prompt_result(x, self.columns, str(dataset_schema(self.dataset_name).frequency), self.pred_len)
```

Store `result.prompt` and `result.metadata` in the general sample and preserve it through `collate_general_graph_batch`.

- [ ] **Step 4: Run focused and full CPU tests**

Run: `python -m unittest tests.test_general_prompting -v; python -m unittest discover -s tests -v`

Expected: PASS, with CUDA smoke remaining skipped unless explicitly enabled.

- [ ] **Step 5: Commit**

```bash
git add bstalignment/general_prompting.py bstalignment/data_general.py tests/test_general_prompting.py
git commit -m "Add general leakage-free forecasting prompts"
```
