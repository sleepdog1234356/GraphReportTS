# Core Battery Ablation Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the formal 16-item battery ablation matrix with four controlled one-factor ablations on MIT, CALCE, and XJTU, while reusing the completed full-model results and handing the new suite into the running remote pipeline without interrupting its active jobs.

**Architecture:** Preserve the default Hankel-graph model path byte-for-byte at the state-dict interface, add a mutually exclusive direct raw-sequence Transformer path for `no_hankel_graph`, and version a new `core-v1` runner that schedules only four variants. Reuse cycle-level graph caches where possible, add a cycle-level six-channel sequence cache, validate prompt identity and full-result provenance before training, and deploy through a separate remote Git worktree plus an atomic future-stage launcher swap.

**Tech Stack:** Python 3.10, PyTorch, NumPy memmaps, pandas, `unittest`, Bash, Git worktrees, NVIDIA CUDA/RTX 4090.

## Global Constraints

- Formal datasets are exactly `mit`, `calce`, and `xjtu`.
- Formal inputs are exactly 32 observed cycles and 20 future-only targets with no historical SOH.
- GraphReportTS main and ablation batch size remains exactly 64.
- Seed remains exactly 42 and existing train/validation/test cell splits and cycle scaling remain unchanged.
- Execution remains on the current FP32 path: do not add AMP, GradScaler, autocast, or TF32 setting changes.
- Reuse `MAIN_TRAINING_PROFILE` unchanged: 80-epoch ceiling, five-epoch warmup, alignment ramp in epochs 6-15, validation-MSE plateau scheduling, early stopping from epoch 20 with patience 20.
- Reuse the completed formal main-model row only after strict metadata and artifact validation.
- Every text-enabled variant receives a prompt string identical to the full graph-cache prompt for the same sample.
- `no_hankel_graph` must never construct or execute Hankel maps, derivative maps, patches, `GraphMapEncoder`, graph attention, structural bias, or domain edges.
- The current remote active worktree must not receive updated Python sources while main-model or baseline subprocesses may still start.
- Legacy general ablations and legacy developer switches remain available but are excluded from the formal `core-v1` schedule.

## File Structure

**Create:**

- `bstalignment/precompute_battery_sequence_cache.py` — build and validate cycle-level `[128, 6]` sequence caches.
- `bstalignment/run_core_ablation_suite.py` — define `core-v1`, validate full reuse, build commands, resume jobs, and write five-row summaries.
- `tests/test_core_ablation.py` — focused unit and integration coverage for sequence inputs, model isolation, caches, provenance, and the new runner.

**Modify:**

- `bstalignment/raw_signal.py` — shared resampled-channel builder, fixed six-channel battery sequence, and canonical full prompt channel names.
- `bstalignment/data_battery_raw.py` — mutually exclusive graph/sequence input mode, sequence-cache loading, canonical prompts, and dual-mode collation.
- `bstalignment/graph_report_model.py` — `RawSequenceEncoder`, conditional graph/sequence modules, removable gate, and removable prompt path.
- `bstalignment/training_strategy.py` — recognize the raw sequence encoder as a core parameter group while retaining exact optimizer coverage.
- `bstalignment/train_graph_report.py` — input-mode CLI, sequence-cache CLI, protocol-stage metadata, timing/provenance, and representation-aware forwarding.
- `scripts/run_battery_ablations_full_hf.sh` — invoke only `core-v1`, resolve its real path through a symlink, separate code and asset roots, and use independent force control.
- `scripts/run_battery_v3_training_strategy_pipeline.sh` — export `ABLATION_FORCE_RETRAIN` without changing stage order.
- `README.md` — replace the formal ablation description and document the four variants and full reuse.
- `docs/work_report.md` — record the approved core suite, run count, cache behavior, and safe remote handoff.
- `tests/test_training_strategy.py` — adjust formal shell/document assertions while retaining legacy runner tests.

---

### Task 1: Shared Six-Channel Resampled Battery Input

**Files:**
- Modify: `bstalignment/raw_signal.py`
- Create: `tests/test_core_ablation.py`

**Interfaces:**
- Produces: `BATTERY_BASE_CHANNELS`, an ordered tuple of strings
- Produces: `BATTERY_SEQUENCE_CHANNELS`, an ordered tuple of strings
- Produces: `FULL_BATTERY_PROMPT_MAP_NAMES`, an ordered tuple of strings
- Produces: `build_resampled_channels(channels: Mapping[str, np.ndarray], resample_len: int, include_ic_dv: bool, required_channels: Sequence[str] = ()) -> dict[str, np.ndarray]`
- Produces: `build_battery_sequence(channels: Mapping[str, np.ndarray], resample_len: int = 128, include_ic_dv: bool = True) -> tuple[np.ndarray, list[str]]`
- Preserves: the existing `build_multiview_maps` arguments, return type, ordering, and numerical behavior

- [ ] **Step 1: Write failing numerical-contract tests**

Add `ResampledBatterySequenceTests` to `tests/test_core_ablation.py`:

```python
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from bstalignment.raw_signal import (
    BATTERY_SEQUENCE_CHANNELS,
    FULL_BATTERY_PROMPT_MAP_NAMES,
    build_battery_sequence,
    build_multiview_maps,
)


class ResampledBatterySequenceTests(unittest.TestCase):
    def channels(self):
        return {
            "current": np.linspace(0.0, 1.0, 17, dtype=np.float32),
            "voltage": np.linspace(3.0, 4.2, 17, dtype=np.float32),
            "temperature": np.linspace(25.0, 35.0, 17, dtype=np.float32),
            "capacity": np.linspace(0.0, 1.1, 17, dtype=np.float32),
        }

    def test_sequence_has_fixed_six_channel_contract(self):
        values, names = build_battery_sequence(self.channels(), resample_len=16)
        self.assertEqual(tuple(names), BATTERY_SEQUENCE_CHANNELS)
        self.assertEqual(values.shape, (16, 6))
        self.assertEqual(values.dtype, np.float32)
        self.assertTrue(np.isfinite(values).all())

    def test_full_prompt_names_match_current_full_map_order(self):
        _, names = build_multiview_maps(
            self.channels(),
            resample_len=16,
            delay_dim=2,
            delay_lag=1,
            include_derivatives=True,
            include_hankel=True,
            include_ic_dv=True,
        )
        self.assertEqual(tuple(names[:10]), FULL_BATTERY_PROMPT_MAP_NAMES)

    def test_missing_formal_channel_is_rejected(self):
        channels = self.channels()
        channels.pop("temperature")
        with self.assertRaisesRegex(ValueError, "temperature"):
            build_battery_sequence(channels, resample_len=16)
```

- [ ] **Step 2: Run the new tests and verify the missing interfaces fail**

Run:

```powershell
python -m unittest tests.test_core_ablation.ResampledBatterySequenceTests -v
```

Expected: import failure naming `BATTERY_SEQUENCE_CHANNELS` or `build_battery_sequence`.

- [ ] **Step 3: Implement the shared resampled-channel functions**

In `bstalignment/raw_signal.py`, add the constants and functions, then make `build_multiview_maps` consume `build_resampled_channels` instead of rebuilding `base` locally:

```python
from typing import Dict, List, Mapping, Sequence, Tuple

BATTERY_BASE_CHANNELS = ("current", "voltage", "temperature", "capacity")
BATTERY_SEQUENCE_CHANNELS = BATTERY_BASE_CHANNELS + ("ic_dqdv", "dv_dq")
FULL_BATTERY_PROMPT_MAP_NAMES = tuple(
    f"{channel}:{view}"
    for channel in BATTERY_SEQUENCE_CHANNELS
    for view in ("hankel", "d1", "d2")
)[:10]


def build_resampled_channels(
    channels: Mapping[str, np.ndarray],
    resample_len: int,
    include_ic_dv: bool,
    required_channels: Sequence[str] = (),
) -> Dict[str, np.ndarray]:
    missing = [name for name in required_channels if name not in channels or np.asarray(channels[name]).size == 0]
    if missing:
        raise ValueError(f"Missing required raw battery channels: {missing}")
    base = {
        name: robust_scale(resample_1d(values, resample_len))
        for name, values in channels.items()
        if values is not None and np.asarray(values).size > 0
    }
    if include_ic_dv:
        if "capacity" not in base or "voltage" not in base:
            raise ValueError("IC/DV requires non-empty capacity and voltage channels")
        base["ic_dqdv"] = robust_scale(smooth_gradient(base["capacity"], base["voltage"]))
        base["dv_dq"] = robust_scale(smooth_gradient(base["voltage"], base["capacity"]))
    return base


def build_battery_sequence(
    channels: Mapping[str, np.ndarray],
    resample_len: int = 128,
    include_ic_dv: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    base = build_resampled_channels(
        channels,
        resample_len=resample_len,
        include_ic_dv=include_ic_dv,
        required_channels=BATTERY_BASE_CHANNELS,
    )
    names = list(BATTERY_SEQUENCE_CHANNELS if include_ic_dv else BATTERY_BASE_CHANNELS)
    return np.stack([base[name] for name in names], axis=-1).astype(np.float32), names
```

Keep the old `ic_dqdv` numerical formula unchanged. Update internal references from the old `dv_dvdq` label to `dv_dq` only where they describe the same computed `dV/dQ` channel; cache schema versions added later prevent old cache reuse.

- [ ] **Step 4: Run raw-signal and existing protocol tests**

Run:

```powershell
python -m unittest tests.test_core_ablation.ResampledBatterySequenceTests tests.test_training_strategy.BatteryWindowProtocolTests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit the shared input contract**

```powershell
git add bstalignment/raw_signal.py tests/test_core_ablation.py
git commit -m "feat: add canonical battery sequence inputs"
```

---

### Task 2: Representation-Aware Battery Dataset and Collation

**Files:**
- Modify: `bstalignment/data_battery_raw.py`
- Modify: `tests/test_core_ablation.py`

**Interfaces:**
- Consumes: `build_battery_sequence`, `FULL_BATTERY_PROMPT_MAP_NAMES`
- Produces: `BATTERY_SEQUENCE_CACHE_VERSION = "battery-sequence-cycle-history-v1"`
- Produces: `battery_sequence_cache_config(dataset_name: str, split: str, max_horizon: int, resample_len: int, allow_summary_fallback: bool, seed: int, max_cycles: int | None, history_len: int) -> dict[str, Any]`
- Produces: `battery_sequence_cache_path(cache_root: str | Path, config: dict[str, Any]) -> Path`
- Extends: `BatteryRawGraphDataset.__init__` with keyword `input_representation: str = "graph"`
- Produces batch key: graph mode uses `maps`; sequence mode uses `raw_sequences`
- Preserves: default graph samples and `collate_graph_report_batch` output

- [ ] **Step 1: Write failing direct-dataset and collation tests**

Extend `tests/test_core_ablation.py` with the existing processed-data fixture pattern and these assertions:

```python
from pathlib import Path
from tempfile import TemporaryDirectory

from bstalignment.data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch


class SequenceDatasetTests(unittest.TestCase):
    def test_sequence_mode_returns_no_maps(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            ds = BatteryRawGraphDataset(
                dataset_name="calce",
                data_root=root,
                split="train",
                history_len=32,
                max_horizon=20,
                resample_len=16,
                input_representation="sequence",
                include_ic_dv=True,
            )
            item = ds[0]
            self.assertNotIn("maps", item)
            self.assertEqual(item["raw_sequences"].shape, (32, 16, 6))

    def test_sequence_collation_preserves_formal_shapes(self):
        item = {
            "raw_sequences": torch.ones(32, 16, 6),
            "history_features": torch.ones(32, 8),
            "history_cycles": torch.arange(32),
            "y": torch.ones(20),
            "mask": torch.ones(20, dtype=torch.bool),
            "horizon": torch.tensor(20),
            "prompt": "prompt",
            "cell_id": "cell",
            "cycle": 32,
            "target_steps": torch.arange(1, 21),
        }
        batch = collate_graph_report_batch([item, item])
        self.assertNotIn("maps", batch)
        self.assertEqual(batch["raw_sequences"].shape, (2, 32, 16, 6))
        self.assertEqual(batch["y"].shape, (2, 20))
```

- [ ] **Step 2: Run the tests and verify `input_representation` is rejected**

Run:

```powershell
python -m unittest tests.test_core_ablation.SequenceDatasetTests -v
```

Expected: failure stating that `BatteryRawGraphDataset.__init__` does not accept `input_representation`.

- [ ] **Step 3: Add the representation switch without changing graph defaults**

In `BatteryRawGraphDataset.__init__`, validate and store:

```python
if input_representation not in {"graph", "sequence"}:
    raise ValueError(f"Unknown battery input_representation: {input_representation}")
self.input_representation = input_representation
if self.input_representation == "sequence" and not self.include_ic_dv:
    raise ValueError("Formal sequence representation requires IC/DV")
```

Add one cycle-input dispatcher used by both MIT and processed item paths:

```python
def _build_cycle_input_from_channels(self, channels):
    if self.input_representation == "sequence":
        return build_battery_sequence(channels, self.resample_len, include_ic_dv=True)
    return build_multiview_maps(
        channels,
        resample_len=self.resample_len,
        delay_dim=self.delay_dim,
        delay_lag=self.delay_lag,
        include_derivatives=self.include_derivatives,
        include_hankel=self.include_hankel,
        include_ic_dv=self.include_ic_dv,
    )
```

Use `FULL_BATTERY_PROMPT_MAP_NAMES` in `_prompt_from_history` for both representations so the default full prompt remains exactly the current string. In `_getitem_mit` and `_getitem_processed`, assign the stacked input to `maps` or `raw_sequences` according to `input_representation`.

Introduce `_mit_cycle_input` and `_processed_cycle_input` as the representation-aware methods. Keep `_mit_cycle_maps` and `_processed_cycle_maps` as backward-compatible graph-only wrappers for `precompute_battery_graph_cache.py`; each wrapper raises if `input_representation != "graph"`. The new sequence cache calls only the representation-aware methods.

Update `collate_graph_report_batch` to reject mixed representations and collate the present key:

```python
has_maps = ["maps" in item for item in batch]
if any(has_maps) and not all(has_maps):
    raise ValueError("Cannot collate mixed graph and sequence battery samples")
input_key = "maps" if all(has_maps) else "raw_sequences"
```

- [ ] **Step 4: Add prompt-identity and graph-regression tests**

Add a test that builds the same processed sample once in graph mode and once in sequence mode and asserts:

```python
self.assertEqual(graph_item["prompt"], sequence_item["prompt"])
self.assertEqual(graph_item["cell_id"], sequence_item["cell_id"])
self.assertEqual(graph_item["cycle"], sequence_item["cycle"])
torch.testing.assert_close(graph_item["y"], sequence_item["y"], rtol=0.0, atol=0.0)
```

Run:

```powershell
python -m unittest tests.test_core_ablation.SequenceDatasetTests tests.test_training_strategy.BatteryWindowProtocolTests -v
```

Expected: all tests pass and existing graph shapes remain unchanged.

- [ ] **Step 5: Commit representation-aware data loading**

```powershell
git add bstalignment/data_battery_raw.py tests/test_core_ablation.py
git commit -m "feat: add direct sequence battery dataset mode"
```

---

### Task 3: Cycle-Level Sequence Cache

**Files:**
- Create: `bstalignment/precompute_battery_sequence_cache.py`
- Modify: `bstalignment/data_battery_raw.py`
- Modify: `tests/test_core_ablation.py`

**Interfaces:**
- Consumes: representation-aware `BatteryRawGraphDataset`
- Produces: `precompute_sequence_split(args: argparse.Namespace, split: str) -> Path`
- Extends dataset arguments: `precomputed_sequence_cache_dir: str | None`, `require_precomputed_sequence_cache: bool`
- Sequence cache layout: `cycle_sequence_history`
- Sequence cache files: `cycle_sequences.npy`, `history_indices.npy`, `y.npy`, `mask.npy`, `horizon.npy`, `target_steps.npy`, `history_features.npy`, `history_cycles.npy`, `meta.jsonl`, `manifest.json`

- [ ] **Step 1: Write failing cache parity and no-graph tests**

Add `SequenceCacheTests` using the processed fixture:

```python
from argparse import Namespace
from bstalignment.precompute_battery_sequence_cache import precompute_sequence_split


class SequenceCacheTests(unittest.TestCase):
    def test_sequence_cache_matches_direct_dataset_without_map_calls(self):
        from tests.test_training_strategy import BatteryDataFixtureMixin

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            BatteryDataFixtureMixin.write_processed_data(root, [1, 1, 1, 1])
            args = Namespace(
                dataset="calce", data_root=str(root), cache_dir=str(root / "sequence_cache"),
                pred_len=20, history_len=32, resample_len=16, seed=42,
                max_cycles=None, batch_size=4, num_workers=0, force=True,
            )
            with patch("bstalignment.data_battery_raw.build_multiview_maps", side_effect=AssertionError("graph path called")):
                cache_path = precompute_sequence_split(args, "train")
            direct = BatteryRawGraphDataset(
                dataset_name="calce", data_root=root, split="train", history_len=32,
                max_horizon=20, resample_len=16, input_representation="sequence",
            )
            cached = BatteryRawGraphDataset(
                dataset_name="calce", data_root=root, split="train", history_len=32,
                max_horizon=20, resample_len=16, input_representation="sequence",
                precomputed_sequence_cache_dir=str(root / "sequence_cache"),
                require_precomputed_sequence_cache=True,
            )
            self.assertTrue(cache_path.exists())
            self.assertEqual(len(cached), len(direct))
            for key in ("raw_sequences", "y", "mask", "target_steps", "history_features", "history_cycles"):
                torch.testing.assert_close(cached[0][key], direct[0][key], rtol=0.0, atol=0.0)
            self.assertEqual(cached[0]["prompt"], direct[0]["prompt"])
```

- [ ] **Step 2: Run the cache test and verify the module is missing**

Run:

```powershell
python -m unittest tests.test_core_ablation.SequenceCacheTests -v
```

Expected: import failure for `bstalignment.precompute_battery_sequence_cache`.

- [ ] **Step 3: Implement sequence config, path, writer, and loader**

Keep the proven graph-cache writer unchanged. Import only its cache-neutral metadata helpers into the new sequence writer:

```python
from .precompute_battery_graph_cache import (
    _collect_cycle_entries,
    _flush_and_close_memmap,
    _sample_payload,
    _write_meta,
)
```

Do not import or call `_entry_maps`, `_cycle_map_results`, or `_parallel_cycle_map_results`. This keeps graph computation unreachable from the sequence writer while reusing the existing sample-target, history-feature, prompt, atomic-memmap, and metadata behavior.

Implement the sequence-only entry function and bounded thread pool in the new module:

```python
def _entry_sequence(ds: BatteryRawGraphDataset, entry: tuple[str, int, int]) -> tuple[np.ndarray, list[str]]:
    kind, owner_idx, row_or_cycle = entry
    if kind == "processed":
        values, names = ds._processed_cycle_input(ds.processed_cells[owner_idx], row_or_cycle)
    else:
        values, names = ds._mit_cycle_input(ds.records[owner_idx], row_or_cycle)
    if ds.input_representation != "sequence":
        raise RuntimeError("Sequence cache requires input_representation=sequence")
    return values, names


def _sequence_results(ds, cycle_items, num_workers):
    def compute(item):
        cycle_idx, key, entry = item
        values, names = _entry_sequence(ds, entry)
        return cycle_idx, key, values, names

    if num_workers == 0:
        for item in cycle_items:
            yield compute(item)
        return
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        yield from pool.map(compute, cycle_items)
```

Instantiate the dataset with `cycle_cache_size=0`, so the worker threads only read immutable records/cells and write results in the parent thread.

Add the sequence cache config/path functions in `data_battery_raw.py`. In the writer, use the same atomic temporary-directory pattern as the graph cache, but store:

```python
cycle_sequences = np.lib.format.open_memmap(
    tmp_path / "cycle_sequences.npy",
    mode="w+",
    dtype=np.float32,
    shape=(len(cycle_items), int(args.resample_len), 6),
)
```

Build every cycle through dataset sequence mode and call `_sample_payload(ds, sample, list(FULL_BATTERY_PROMPT_MAP_NAMES), mit_dfs)` for targets, history data, and prompts. The manifest must contain:

```python
manifest = {
    "layout": "cycle_sequence_history",
    "config": config,
    "cycle_scale": ds.cycle_scale,
    "sample_count": sample_count,
    "cycle_count": len(cycle_items),
    "cycle_sequence_shape": [int(args.resample_len), 6],
    "files": {
        "cycle_sequences": "cycle_sequences.npy",
        "history_indices": "history_indices.npy",
        "y": "y.npy",
        "mask": "mask.npy",
        "horizon": "horizon.npy",
        "target_steps": "target_steps.npy",
        "history_features": "history_features.npy",
        "history_cycles": "history_cycles.npy",
        "meta": "meta.jsonl",
    },
}
```

Dataset loading in sequence mode must require this layout and return `raw_sequences`, never `maps`.

- [ ] **Step 4: Run graph and sequence cache regressions**

Run:

```powershell
python -m unittest tests.test_core_ablation.SequenceCacheTests tests.test_training_strategy.BatteryWindowProtocolTests.test_precomputed_graph_cache_matches_uncached_full_horizon_samples tests.test_training_strategy.BatteryWindowProtocolTests.test_parallel_precompute_uses_parallel_path_and_matches_serial_cache -v
```

Expected: all tests pass; graph-cache files remain byte/array equivalent and the sequence test proves no map call occurs.

- [ ] **Step 5: Commit the sequence cache**

```powershell
git add bstalignment/data_battery_raw.py bstalignment/precompute_battery_sequence_cache.py tests/test_core_ablation.py
git commit -m "feat: add cycle-level battery sequence cache"
```

---

### Task 4: Mutually Exclusive Graph and Raw-Sequence Model Paths

**Files:**
- Modify: `bstalignment/graph_report_model.py`
- Modify: `bstalignment/training_strategy.py`
- Modify: `tests/test_core_ablation.py`

**Interfaces:**
- Produces: `RawSequenceEncoder(input_dim: int = 6, d_model: int = 128, max_length: int = 128, layers: int = 2, n_heads: int = 4, dropout: float = 0.1)`
- Extends config: `battery_input_mode: str = "hankel_graph"`, `raw_sequence_len: int = 128`, `raw_sequence_dim: int = 6`
- Extends forward: optional `raw_sequences: torch.Tensor | None`
- Preserves: default full state-dict keys and tensor shapes

- [ ] **Step 1: Write failing model-isolation tests**

Add `CoreAblationModelTests`:

```python
class CoreAblationModelTests(unittest.TestCase):
    def config(self, **updates):
        values = dict(
            variant="battery", d_model=8, output_dim=1, graph_layers=1,
            patch_size=2, patch_stride=1, topk_edges=1,
            use_hf_text_encoder=False, temporal_heads=2,
            raw_sequence_len=16, raw_sequence_dim=6,
        )
        values.update(updates)
        return GraphReportTSConfig(**values)

    def test_raw_sequence_model_has_no_graph_encoder(self):
        model = GraphReportTS(self.config(battery_input_mode="raw_sequence"))
        self.assertIsNone(model.graph_encoder)
        self.assertIsNotNone(model.raw_sequence_encoder)
        out = model(
            None, ["battery prompt", "battery prompt"], torch.tensor([20, 20]),
            history_features=torch.randn(2, 32, 8),
            raw_sequences=torch.randn(2, 32, 16, 6),
        )
        self.assertEqual(out["pred"].shape, (2, 20))

    def test_no_gate_has_constant_one_and_no_gate_parameters(self):
        model = GraphReportTS(self.config(use_text_gate=False))
        self.assertIsNone(model.semantic_fusion.gate)
        out = model(torch.randn(2, 32, 3, 2, 3), ["p", "p"], torch.tensor([20, 20]), history_features=torch.randn(2, 32, 8))
        torch.testing.assert_close(out["gate"], torch.ones_like(out["gate"]))

    def test_no_prompt_constructs_no_semantic_modules(self):
        model = GraphReportTS(self.config(use_report_prompt=False))
        self.assertIsNone(model.text_encoder)
        self.assertIsNone(model.semantic_fusion)
        self.assertFalse(any(name.startswith(("text_encoder", "semantic_fusion", "fusion")) for name, _ in model.named_parameters()))
```

- [ ] **Step 2: Run tests and verify missing config/encoder failures**

Run:

```powershell
python -m unittest tests.test_core_ablation.CoreAblationModelTests -v
```

Expected: failures for unknown `battery_input_mode` and `raw_sequence_len` fields.

- [ ] **Step 3: Implement `RawSequenceEncoder`**

Add the class next to the existing cycle encoders:

```python
class RawSequenceEncoder(nn.Module):
    def __init__(self, input_dim=6, d_model=128, max_length=128, layers=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embed = nn.Embedding(max_length, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=_valid_n_heads(d_model, n_heads),
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.pool = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        b, t, length, channels = sequences.shape
        flat = sequences.reshape(b * t, length, channels)
        pos = torch.arange(length, device=sequences.device)
        tokens = self.encoder(self.input_proj(flat.float()) + self.pos_embed(pos).unsqueeze(0))
        weights = torch.softmax(self.pool(tokens).squeeze(-1), dim=-1)
        pooled = self.norm(torch.sum(tokens * weights.unsqueeze(-1), dim=1))
        return pooled.reshape(b, t, -1)
```

- [ ] **Step 4: Make graph/sequence and semantic modules conditional**

Validate `battery_input_mode` in `GraphReportTS.__init__`. Construct exactly one of `graph_encoder` and `raw_sequence_encoder`. Preserve the existing full construction and names when the default is `hankel_graph`.

Change `GatedSemanticFusion` so `self.gate` is `None` when `use_gate=False`, and branch to a constant-one tensor. Construct `fusion` and `semantic_fusion` only when report prompt plus cross-modal fusion is enabled.

Add a private encoder dispatcher:

```python
def _encode_battery_history(self, maps, raw_sequences):
    if self.cfg.battery_input_mode == "raw_sequence":
        if maps is not None or raw_sequences is None:
            raise ValueError("raw_sequence mode requires raw_sequences and forbids maps")
        cycle_repr = self.raw_sequence_encoder(raw_sequences)
        context, tokens = self.temporal_encoder(cycle_repr)
        return context, tokens, {"tokens": tokens, "graph_attn": None}
    if maps is None or raw_sequences is not None:
        raise ValueError("hankel_graph mode requires maps and forbids raw_sequences")
    return self._encode_graph_history(maps)
```

- [ ] **Step 5: Verify optimizer coverage and full state-dict preservation**

Add assertions that `build_graph_report_optimizer` places every raw encoder parameter in a `role == "core"` group and that no forbidden parameter appears for no-gate/no-prompt models. Add a golden comparison that constructs a default full model before and after configuration serialization and compares `state_dict().keys()` and shapes.

Run:

```powershell
python -m unittest tests.test_core_ablation.CoreAblationModelTests tests.test_training_strategy.MainStrategyTests -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit conditional model paths**

```powershell
git add bstalignment/graph_report_model.py bstalignment/training_strategy.py tests/test_core_ablation.py
git commit -m "feat: add graph-free battery sequence encoder"
```

---

### Task 5: Trainer Integration, Timing, and Version Metadata

**Files:**
- Modify: `bstalignment/train_graph_report.py`
- Modify: `tests/test_core_ablation.py`
- Modify: `tests/test_training_strategy.py`

**Interfaces:**
- Adds CLI: `--battery_input_mode {hankel_graph,raw_sequence}`
- Adds CLI: `--precomputed_sequence_cache_dir PATH`
- Adds CLI: `--require_precomputed_sequence_cache`
- Adds CLI: `--protocol_stage {main,ablation}` with default `main`
- Adds CLI: `--ablation_suite_version STRING`
- Adds CLI: `--run_dir PATH`; when absent, preserve the existing `out_dir/variant/dataset` layout
- Adds `epoch_seconds`, `total_train_seconds`, `stopped_epoch`, and `trainable_parameter_count` to produced metadata/artifacts

- [ ] **Step 1: Write failing parser, forwarding, and policy tests**

Add tests that patch `sys.argv`, parse raw-sequence flags, and assert formal batch defaults remain 64. Add a forwarding test:

```python
batch = {
    "raw_sequences": torch.randn(2, 32, 16, 6),
    "prompt": ["p", "p"],
    "horizon": torch.tensor([20, 20]),
    "history_features": torch.randn(2, 32, 8),
}
out = graph_report_trainer._model_forward(raw_model, batch)
self.assertEqual(out["pred"].shape, (2, 20))
```

Add source-policy assertions that `train_graph_report.py` contains no `autocast` or `GradScaler`.

Define the CUDA smoke test referenced by the remote handoff. It uses small tensors so it validates module wiring and optimizer behavior without competing materially with the active job:

```python
@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
def test_all_core_variants_complete_one_optimizer_step_on_cuda(self):
    device = torch.device("cuda")
    variants = {
        "no_hankel_graph": dict(battery_input_mode="raw_sequence"),
        "no_report_prompt": dict(use_report_prompt=False),
        "no_ic_dv": {},
        "no_text_gate": dict(use_text_gate=False),
    }
    for name, updates in variants.items():
        with self.subTest(name=name):
            cfg = GraphReportTSConfig(
                variant="battery", d_model=8, output_dim=1, graph_layers=1,
                patch_size=2, patch_stride=1, topk_edges=1,
                use_hf_text_encoder=False, battery_history_len=2,
                temporal_heads=2, raw_sequence_len=16, raw_sequence_dim=6,
                **updates,
            )
            model = GraphReportTS(cfg).to(device)
            optimizer = build_graph_report_optimizer(model, MAIN_TRAINING_PROFILE)
            batch = {
                "prompt": ["battery prompt", "battery prompt"],
                "horizon": torch.tensor([20, 20], device=device),
                "history_features": torch.randn(2, 2, 8, device=device),
            }
            if name == "no_hankel_graph":
                batch["raw_sequences"] = torch.randn(2, 2, 16, 6, device=device)
            else:
                map_channels = 12 if name == "no_ic_dv" else 18
                batch["maps"] = torch.randn(2, 2, map_channels, 2, 3, device=device)
            output = graph_report_trainer._model_forward(model, batch)
            loss = output["pred"].square().mean()
            self.assertTrue(torch.isfinite(loss))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
```

- [ ] **Step 2: Run focused tests and verify parser failures**

Run:

```powershell
python -m unittest tests.test_core_ablation.TrainerIntegrationTests tests.test_training_strategy.VariantBatchDefaultTests -v
```

Expected: unknown CLI arguments or missing representation-aware forwarding.

- [ ] **Step 3: Wire input representation through loaders and model config**

Pass `input_representation="sequence"` only for `battery_input_mode == "raw_sequence"`. Pass sequence-cache arguments only in that mode and graph-cache arguments only in graph mode. Update `_model_forward`:

```python
return model(
    batch.get("maps"),
    batch["prompt"],
    batch["horizon"],
    steps=steps,
    history_features=batch.get("history_features"),
    raw_sequences=batch.get("raw_sequences"),
)
```

Replace every `batch["maps"].size(0)` with:

```python
def _batch_size(batch: Dict[str, Any]) -> int:
    source = batch.get("maps")
    if source is None:
        source = batch["raw_sequences"]
    return int(source.size(0))
```

Use `args.protocol_stage` in `require_formal_battery_protocol` and store it in `run_config.json`.

Resolve the output directory without changing legacy callers:

```python
if args.run_dir is None:
    out_dir = ensure_dir(Path(args.out_dir) / args.variant / args.dataset)
else:
    out_dir = ensure_dir(Path(args.run_dir))
```

Add a parser test proving the default path behavior remains unchanged and a core-runner test proving `--run_dir root/battery/mit/no_hankel_graph` writes directly to that directory.

- [ ] **Step 4: Record timing and provenance without changing scheduling**

Use `time.perf_counter()` around each complete train-plus-validation epoch. Add `epoch_seconds` to each `epoch_history.jsonl` row. After stopping, write `run_summary.json`:

```python
save_json(
    {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "mean_epoch_seconds": float(sum(epoch_seconds) / len(epoch_seconds)),
        "total_train_seconds": float(sum(epoch_seconds)),
        "trainable_parameter_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "training_strategy_version": training_strategy_version,
        "ablation_suite_version": args.ablation_suite_version,
    },
    out_dir / "run_summary.json",
)
```

Persist `ablation_suite_version` and `trainable_parameter_count` in `run_config.json` and checkpoints. Resume must retain the original timer history and append only new epoch durations.

- [ ] **Step 5: Run trainer and strategy tests**

Run:

```powershell
python -m unittest tests.test_core_ablation.TrainerIntegrationTests tests.test_training_strategy.FormalBatteryProtocolTests tests.test_training_strategy.MainStrategyTests tests.test_training_strategy.MainTrainerPolicyTests -v
```

Expected: all tests pass; profile values and batch enforcement remain unchanged.

- [ ] **Step 6: Commit trainer integration**

```powershell
git add bstalignment/train_graph_report.py tests/test_core_ablation.py tests/test_training_strategy.py
git commit -m "feat: train versioned core ablation variants"
```

---

### Task 6: `core-v1` Runner, Full Reuse, and Prompt Validation

**Files:**
- Create: `bstalignment/run_core_ablation_suite.py`
- Modify: `tests/test_core_ablation.py`

**Interfaces:**
- Produces: `CORE_ABLATION_SUITE_VERSION = "core-v1"`
- Produces: `CORE_BATTERY_ABLATIONS`, an ordered mapping from variant name to an immutable tuple of CLI tokens
- Produces: `require_reusable_full_reference(result_dir: Path, dataset: str, training_strategy_version: str) -> dict[str, Any]`
- Produces: `core_run_config_matches(result_dir: Path, dataset: str, ablation: str, training_strategy_version: str) -> bool`
- Produces: `verify_prompt_cache_identity(reference_cache: Path, candidate_cache: Path) -> None`
- Default schedule: three datasets by four trained variants

- [ ] **Step 1: Write failing matrix and provenance tests**

Add `CoreAblationRunnerTests`:

```python
from bstalignment.run_core_ablation_suite import (
    CORE_ABLATION_SUITE_VERSION,
    CORE_BATTERY_ABLATIONS,
    require_reusable_full_reference,
)


class CoreAblationRunnerTests(unittest.TestCase):
    def test_formal_matrix_contains_only_four_single_factor_variants(self):
        self.assertEqual(CORE_ABLATION_SUITE_VERSION, "core-v1")
        self.assertEqual(
            list(CORE_BATTERY_ABLATIONS),
            ["no_hankel_graph", "no_report_prompt", "no_ic_dv", "no_text_gate"],
        )

    def test_full_reference_requires_complete_matching_artifacts(self):
        with TemporaryDirectory() as tmp:
            result = Path(tmp)
            (result / "best.pt").write_bytes(b"checkpoint")
            (result / "test_metrics.json").write_text('{"mse": 0.1, "mae": 0.2, "rmse": 0.316}', encoding="utf-8")
            (result / "run_config.json").write_text(json.dumps({
                "training_strategy_version": TRAINING_STRATEGY_VERSION,
                "args": {
                    "variant": "battery", "dataset": "mit", "history_len": 32,
                    "pred_len": 20, "batch_size": 64, "seed": 42,
                    "no_ic_dv": False, "no_hankel_map": False,
                    "no_derivative_map": False, "no_report_prompt": False,
                    "no_cross_modal": False, "no_text_gate": False,
                    "no_semantic_alignment": False, "no_align_loss": False,
                },
                "model_cfg": {
                    "variant": "battery", "freeze_text": True, "use_hf_text_encoder": True,
                    "use_report_prompt": True, "use_cross_modal_fusion": True,
                    "use_dynamic_graph": True, "use_domain_edges": True,
                    "unified_decoder": True, "battery_history_len": 32,
                    "history_feature_dim": 8, "use_multi_cycle_raw": True,
                    "single_cycle_raw": False, "use_numeric_history": True,
                    "use_text_gate": True, "use_semantic_alignment": True,
                    "use_relative_steps": True,
                },
            }), encoding="utf-8")
            row = require_reusable_full_reference(result, "mit", TRAINING_STRATEGY_VERSION)
            self.assertEqual(row["result_source"], "reused_main")
```

Add mismatch cases for dataset, seed, batch, missing artifact, model flag, malformed JSON, and strategy version. Assert the directory still exists after every failure.

- [ ] **Step 2: Run runner tests and verify the new module is missing**

Run:

```powershell
python -m unittest tests.test_core_ablation.CoreAblationRunnerTests -v
```

Expected: import failure for `run_core_ablation_suite`.

- [ ] **Step 3: Implement strict specs and non-destructive matching**

Define:

```python
CORE_ABLATION_SUITE_VERSION = "core-v1"
CORE_BATTERY_ABLATIONS = {
    "no_hankel_graph": ("--battery_input_mode", "raw_sequence"),
    "no_report_prompt": ("--no_report_prompt",),
    "no_ic_dv": ("--no_ic_dv",),
    "no_text_gate": ("--no_text_gate",),
}
```

`require_reusable_full_reference` must load all three artifacts, compare every controlled field, and raise `RuntimeError` containing dataset, expected value, observed value, and path. It must never delete or rewrite the reference.

The current full runs predate `battery_input_mode`; treat a missing field as the legacy-equivalent value `hankel_graph`, but reject any explicit non-graph value. All existing `no_*` argument fields listed in the test fixture must be present and false.

`core_run_config_matches` requires training strategy, suite version, dataset, batch 64, history 32, prediction 20, seed 42, and exactly the flags for the named variant. A mismatched existing output is a hard failure; only a matching incomplete output may resume.

- [ ] **Step 4: Implement cache selection and prompt identity**

For `no_hankel_graph`, schedule `bstalignment.precompute_battery_sequence_cache` and pass the required sequence cache to the trainer. For `no_ic_dv`, schedule the graph cache writer with `--no_ic_dv`. For `no_report_prompt` and `no_text_gate`, reuse the full graph cache.

Every train command passes `--run_dir <out_root>/battery/<dataset>/<ablation>`. Do not pass a variant-specific path through legacy `--out_dir`, because the trainer would append another `battery/<dataset>` segment.

Compare `meta.jsonl` files in sample order:

```python
def verify_prompt_cache_identity(reference_cache: Path, candidate_cache: Path) -> None:
    reference = [json.loads(line) for line in (reference_cache / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line]
    candidate = [json.loads(line) for line in (candidate_cache / "meta.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if len(reference) != len(candidate):
        raise RuntimeError(f"Prompt sample count mismatch: expected={len(reference)} observed={len(candidate)} path={candidate_cache}")
    for index, (expected, observed) in enumerate(zip(reference, candidate)):
        identity = ("cell_id", "cycle", "prompt")
        if any(expected[key] != observed[key] for key in identity):
            raise RuntimeError(f"Prompt mismatch dataset sample={index} expected={expected} observed={observed} path={candidate_cache}")
```

- [ ] **Step 5: Build commands, resume safely, and write summaries**

CLI defaults to `--datasets mit calce xjtu` and `--ablations` in constant order. Require `--full_result_root`, `--graph_cache_dir`, `--sequence_cache_dir`, `--out_root`, `--text_model`, and `--full_reference_commit` from the shell wrapper.

Resolve the implementation commit once with `git rev-parse HEAD` and store it as `source_git_commit` on trained rows. Store the required CLI value as `full_reference_git_commit` on the reused full row and on every trained row.

Dry-run validates CLI/protocol values and prints exactly 12 `train_graph_report` commands without requiring full results or caches to exist. Normal execution validates artifacts, imports the full row, runs/resumes the four variants, and writes per-dataset CSV plus combined CSV. Populate timing columns from `run_summary.json`; leave them empty only for reused full results without historical timing. Every row also contains `training_strategy_version`, `ablation_suite_version`, `result_source`, `source_git_commit`, and `full_reference_git_commit`.

- [ ] **Step 6: Test matrix, resume, mismatch, and five-row summaries**

Mock `subprocess.run` and assert:

- default dry-run emits 12 training commands;
- each dataset summary has `full` plus four variants;
- complete matching results skip;
- matching `last.pt` resumes without `--no_resume`;
- mismatched metadata raises and preserves files;
- legacy `run_ablation_suite` metadata cannot satisfy `core-v1`.

Run:

```powershell
python -m unittest tests.test_core_ablation.CoreAblationRunnerTests -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit the core runner**

```powershell
git add bstalignment/run_core_ablation_suite.py tests/test_core_ablation.py
git commit -m "feat: add focused core battery ablation suite"
```

---

### Task 7: Formal Shell Entry, Documentation, and Pipeline Policy

**Files:**
- Modify: `scripts/run_battery_ablations_full_hf.sh`
- Modify: `scripts/run_battery_v3_training_strategy_pipeline.sh`
- Modify: `tests/test_training_strategy.py`
- Modify: `README.md`
- Modify: `docs/work_report.md`

**Interfaces:**
- `scripts/run_battery_ablations_full_hf.sh ACTIVE_ASSET_ROOT`
- Environment: `ABLATION_CODE_ROOT`, `ABLATION_FORCE_RETRAIN`, `FULL_REFERENCE_COMMIT`
- Preserves top-level order: `main -> baselines -> ablations`

- [ ] **Step 1: Write failing shell-policy and documentation tests**

Update `PipelineScriptTests` to assert:

```python
ablation = Path("scripts/run_battery_ablations_full_hf.sh").read_text(encoding="utf-8")
self.assertIn("bstalignment.run_core_ablation_suite", ablation)
self.assertIn("ABLATION_FORCE_RETRAIN", ablation)
self.assertIn("readlink -f", ablation)
self.assertNotIn("bstalignment.run_ablation_suite", ablation)
self.assertNotIn("FORCE_RETRAIN", ablation.replace("ABLATION_FORCE_RETRAIN", ""))
```

Assert README and work report name all four variants, say 12 new jobs, and document full reuse.

- [ ] **Step 2: Run tests and verify the old runner is still referenced**

Run:

```powershell
python -m unittest tests.test_training_strategy.PipelineScriptTests -v
```

Expected: failure because the shell still invokes `run_ablation_suite`.

- [ ] **Step 3: Replace the formal ablation shell body**

Resolve a symlink to find the new code worktree while treating the first argument as the active asset root:

```bash
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
CODE_ROOT="${ABLATION_CODE_ROOT:-$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)}"
ASSET_ROOT="${1:-$CODE_ROOT}"

asset_path() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s\n' "$ASSET_ROOT/$1" ;;
  esac
}

OUT_ROOT="$(asset_path "${OUT_ROOT:-runs/full_hf_v3_training_strategy_nosoh}")"
GRAPH_CACHE_DIR="$(asset_path "${BATTERY_GRAPH_CACHE_DIR:-runs/cache/battery_graph}")"
SEQUENCE_CACHE_DIR="$(asset_path "${BATTERY_SEQUENCE_CACHE_DIR:-runs/cache/battery_sequence}")"
TEXT_MODEL="$(asset_path "${TEXT_MODEL:-hf_models/distilbert-base-uncased}")"
FULL_REFERENCE_COMMIT="${FULL_REFERENCE_COMMIT:-$(git -C "$ASSET_ROOT" rev-parse HEAD)}"

FORCE_ARGS=()
case "${ABLATION_FORCE_RETRAIN:-0}" in
  0) ;;
  1) FORCE_ARGS=(--force_retrain) ;;
  *) echo "ABLATION_FORCE_RETRAIN must be 0 or 1" >&2; exit 2 ;;
esac

cd "$CODE_ROOT"
python -u -m bstalignment.run_core_ablation_suite \
  --datasets mit calce xjtu \
  --data_root "$ASSET_ROOT/bstalignment/data" \
  --full_result_root "$OUT_ROOT/graph_report_ts/battery" \
  --out_root "$OUT_ROOT/graph_report_core_ablation" \
  --graph_cache_dir "$GRAPH_CACHE_DIR" \
  --sequence_cache_dir "$SEQUENCE_CACHE_DIR" \
  --text_model "$TEXT_MODEL" \
  --batch_size 64 \
  --cache_task_batch_size 128 \
  --num_workers 16 \
  --device cuda \
  --full_reference_commit "$FULL_REFERENCE_COMMIT" \
  "${FORCE_ARGS[@]}"
```

This explicit branch ensures `ABLATION_FORCE_RETRAIN=0` does not pass `--force_retrain`.

- [ ] **Step 4: Keep pipeline order and separate force controls**

In `run_battery_v3_training_strategy_pipeline.sh`, retain the existing three command order. Export:

```bash
export ABLATION_FORCE_RETRAIN="${ABLATION_FORCE_RETRAIN:-0}"
```

Do not route the top-level `FORCE_RETRAIN=1` into the ablation stage.

- [ ] **Step 5: Update public documentation**

Replace the formal 16-item description with the exact five-row matrix, explain that `full` is reused, state that 12 new jobs are trained, distinguish the new structural `no_hankel_graph` from the legacy `no_hankel_map`, and document the sequence cache and recovery controls.

- [ ] **Step 6: Run shell and documentation tests**

Run:

```powershell
python -m unittest tests.test_training_strategy.PipelineScriptTests tests.test_training_strategy.AblationCompletionPolicyTests -v
```

Expected: all tests pass and legacy completion-policy unit tests remain green for the non-formal runner.

- [ ] **Step 7: Commit shell and documentation changes**

```powershell
git add scripts/run_battery_ablations_full_hf.sh scripts/run_battery_v3_training_strategy_pipeline.sh tests/test_training_strategy.py README.md docs/work_report.md
git commit -m "docs: switch formal pipeline to core ablations"
```

---

### Task 8: Full Local Verification and Remote No-Interruption Handoff

**Files:**
- Verify: all files changed in Tasks 1-7
- Operationally change only: remote future-stage `scripts/run_battery_ablations_full_hf.sh` path, by atomic symlink replacement

**Interfaces:**
- Consumes: a clean implementation commit on branch `codex/core-ablation-redesign`
- Produces: a detached remote worktree at `/root/autodl-tmp/GraphReportTS-core-ablation`
- Preserves: active remote training/baseline PID and active Python source tree

- [ ] **Step 1: Run the complete local test suite**

Run:

```powershell
python -m compileall bstalignment
python -m unittest tests.test_core_ablation tests.test_training_strategy -v
git diff --check
git status --short
```

Expected: compilation succeeds, all tests pass, `git diff --check` is silent, and the worktree is clean.

- [ ] **Step 2: Review the final diff against the approved spec**

Run:

```powershell
$base = git merge-base origin/main HEAD
git diff $base HEAD --stat
git diff $base HEAD -- bstalignment/graph_report_model.py bstalignment/train_graph_report.py bstalignment/run_core_ablation_suite.py scripts/run_battery_ablations_full_hf.sh
```

Confirm from the diff that the full default mode remains `hankel_graph`, no AMP appears, only four formal variants are scheduled, and the active main/baseline scripts are unchanged except for the independent ablation force export.

- [ ] **Step 3: Push the implementation branch for remote worktree creation**

Run:

```powershell
git push -u origin codex/core-ablation-redesign
```

Expected: the remote branch is updated successfully. Record the implementation commit:

```powershell
git rev-parse HEAD
```

- [ ] **Step 4: Create a detached remote worktree without updating the active tree**

Run from the local machine:

```powershell
ssh connect.westc.seetacloud.com 'ACTIVE=/root/autodl-tmp/GraphReportTS; NEW=/root/autodl-tmp/GraphReportTS-core-ablation; git -C "$ACTIVE" fetch origin codex/core-ablation-redesign; COMMIT=$(git -C "$ACTIVE" rev-parse FETCH_HEAD); if [ -e "$NEW" ]; then test "$(git -C "$NEW" rev-parse HEAD)" = "$COMMIT"; else git -C "$ACTIVE" worktree add --detach "$NEW" "$COMMIT"; fi; git -C "$NEW" status --short'
```

Expected: the new worktree is clean and the active worktree HEAD is unchanged.

- [ ] **Step 5: Run remote CPU preflight in the new worktree**

Run:

```powershell
ssh connect.westc.seetacloud.com 'NEW=/root/autodl-tmp/GraphReportTS-core-ablation; PY=/root/miniconda3/envs/graphreport/bin/python; cd "$NEW"; "$PY" -m compileall bstalignment; "$PY" -m unittest tests.test_core_ablation tests.test_training_strategy -v'
```

Expected: all tests pass without changing files under the active worktree.

- [ ] **Step 6: Run formal dry-run and one-batch GPU preflight against active assets**

Run:

```powershell
ssh connect.westc.seetacloud.com 'ACTIVE=/root/autodl-tmp/GraphReportTS; NEW=/root/autodl-tmp/GraphReportTS-core-ablation; PY=/root/miniconda3/envs/graphreport/bin/python; COMMIT=$(git -C "$ACTIVE" rev-parse HEAD); cd "$NEW"; "$PY" -m bstalignment.run_core_ablation_suite --datasets mit calce xjtu --data_root "$ACTIVE/bstalignment/data" --full_result_root "$ACTIVE/runs/full_hf_v3_training_strategy_nosoh/graph_report_ts/battery" --out_root "$ACTIVE/runs/full_hf_v3_training_strategy_nosoh/graph_report_core_ablation" --graph_cache_dir "$ACTIVE/runs/cache/battery_graph" --sequence_cache_dir "$ACTIVE/runs/cache/battery_sequence" --text_model "$ACTIVE/hf_models/distilbert-base-uncased" --batch_size 64 --cache_task_batch_size 128 --num_workers 16 --device cuda --full_reference_commit "$COMMIT" --dry_run | tee /tmp/core_ablation_dry_run.log; test "$(grep -c "bstalignment.train_graph_report" /tmp/core_ablation_dry_run.log)" -eq 12; "$PY" -m unittest tests.test_core_ablation.TrainerIntegrationTests.test_all_core_variants_complete_one_optimizer_step_on_cuda -v'
```

Expected: graph variants and sequence variants all pass; no formal output directory is created by dry-run or one-batch tests.

- [ ] **Step 7: Capture active-job evidence immediately before cutover**

Run:

```powershell
ssh connect.westc.seetacloud.com 'ACTIVE=/root/autodl-tmp/GraphReportTS; PID=$(pgrep -fo "bstalignment\.(train_graph_report|train_battery_official_baselines)"); test -n "$PID"; echo "$PID" > /tmp/graphreport_active_pid; ps -p "$PID" -o pid,ppid,etime,stat,cmd; readlink -f /proc/"$PID"/cwd; find "$ACTIVE/runs/full_hf_v3_training_strategy_nosoh" -name last.pt -printf "%T@ %p\n" 2>/dev/null | sort -n | tail -1; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader'
```

Expected: an active main or baseline Python PID exists and its CWD is the original active worktree.

- [ ] **Step 8: Atomically redirect only the future ablation entry**

Run:

```powershell
ssh connect.westc.seetacloud.com 'ACTIVE=/root/autodl-tmp/GraphReportTS; NEW=/root/autodl-tmp/GraphReportTS-core-ablation; TARGET="$ACTIVE/scripts/run_battery_ablations_full_hf.sh"; NEXT="$ACTIVE/scripts/.run_battery_ablations_full_hf.sh.next"; BACKUP="$ACTIVE/scripts/run_battery_ablations_full_hf.sh.pre-core"; test -f "$NEW/scripts/run_battery_ablations_full_hf.sh"; test ! -e "$NEXT"; cp -a "$TARGET" "$BACKUP"; ln -s "$NEW/scripts/run_battery_ablations_full_hf.sh" "$NEXT"; mv -Tf "$NEXT" "$TARGET"; test "$(readlink -f "$TARGET")" = "$NEW/scripts/run_battery_ablations_full_hf.sh"'
```

Expected: only the future-stage path changes; no active Python file or process is touched.

- [ ] **Step 9: Prove the active job continued uninterrupted**

Run immediately after Step 8:

```powershell
ssh connect.westc.seetacloud.com 'PID_BEFORE=$(cat /tmp/graphreport_active_pid); PID_AFTER=$(pgrep -fo "bstalignment\.(train_graph_report|train_battery_official_baselines)"); test "$PID_BEFORE" = "$PID_AFTER"; ps -p "$PID_AFTER" -o pid,ppid,etime,stat,cmd; nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader; readlink -f /root/autodl-tmp/GraphReportTS/scripts/run_battery_ablations_full_hf.sh'
```

Expected: PID is unchanged, elapsed time has increased, GPU state remains normal for the current stage, and the future ablation entry resolves to the new worktree.

- [ ] **Step 10: Verify eventual handoff behavior**

When the parent pipeline reaches the ablation stage, confirm the process command contains `bstalignment.run_core_ablation_suite`, the output root is `graph_report_core_ablation`, the first dataset imports a `full` row, and no legacy `run_ablation_suite` process starts.

If the delegated stage fails validation, leave completed main/baseline outputs and valid ablation outputs untouched, fix the new worktree on the implementation branch, create a new detached worktree commit, rerun preflight, and atomically repoint the same future-stage symlink. Do not restore the legacy 16-item runner.

---

## Completion Checklist

- [ ] The local branch contains focused commits for all seven implementation tasks.
- [ ] `python -m unittest tests.test_core_ablation tests.test_training_strategy -v` passes locally and remotely.
- [ ] The formal dry-run emits exactly 12 new training commands.
- [ ] The default full model retains its state-dict keys, tensor shapes, prompt strings, and FP32 training path.
- [ ] The remote active PID survives the future-stage launcher swap.
- [ ] The future ablation process uses `core-v1` and never starts the legacy 16-item suite.
- [ ] Per-dataset summaries contain exactly `full`, `no_hankel_graph`, `no_report_prompt`, `no_ic_dv`, and `no_text_gate`.
