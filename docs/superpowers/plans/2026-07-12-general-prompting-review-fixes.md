# General Prompting Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct Task 5 tied-trend selection and expose truthful 192-token encoder audit metrics.

**Architecture:** General prompting owns only deterministic pre-tokenization fallback metadata. Text encoders own actual token counts and truncation. GraphReportTS exposes general-only per-prompt audits and evaluation aggregates them into JSON metrics.

**Tech Stack:** Python 3.13, NumPy, PyTorch, `unittest`.

## Global Constraints

- General datasets remain the six formal datasets, input history is 36, feature mode is M, and horizons are 96, 192, 336, and 720.
- Do not change shared or battery prompt helpers, battery output, training execution, or CUDA execution.
- General fallback and encoder limits are 192; actual tokenizer metrics are separately named from pre-tokenization metrics.

---

### Task 1: Selection and fallback audit

**Files:**
- Modify: `tests/test_general_prompting.py`
- Modify: `bstalignment/general_prompting.py`

**Interfaces:**
- Produces: twelve unique tied-trend summary columns.
- Produces: `pretoken_word_count`, `pretoken_word_budget`, and `pretoken_word_truncated`.

- [ ] Write failing tied-trend and fallback-audit tests, run `py -3.13 -m unittest tests.test_general_prompting.GeneralPromptingTests -v`, and verify failure caused by overlapping tie groups and the old 256 budget.
- [ ] Implement disjoint selection and truthful fallback field names, then rerun the focused prompt tests to GREEN.

### Task 2: Encoder audit and JSON metrics

**Files:**
- Modify: `bstalignment/models.py`
- Modify: `bstalignment/graph_report_model.py`
- Modify: `bstalignment/train_graph_report.py`
- Modify: `tests/test_general_prompting.py`

**Interfaces:**
- Produces: optional `prompt_audit` from general GraphReportTS output.
- Produces: `encoder_token_count_mean`, `encoder_truncated_count`, `encoder_truncated_rate`, and `encoder_token_limit` in evaluation metrics.

- [ ] Write failing actual-encoder, model-output, and metric-aggregation tests; run `py -3.13 -m unittest tests.test_general_prompting -v` and verify the absent audit failure.
- [ ] Use the existing HF tokenizer and the simple encoder's existing split behavior to compute audit without constructing tokenizers per dataset sample; expose audit only for general models.
- [ ] Run `py -3.13 -m unittest tests.test_general_prompting -v` and `py -3.13 -m unittest discover -s tests -v`; then self-review and commit.
