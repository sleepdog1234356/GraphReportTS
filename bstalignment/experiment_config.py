from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class DataConfig:
    dataset_group: str = "battery"  # battery or general
    dataset_name: str = "mit"
    data_root: str = "bstalignment/data"
    raw_root: str = "bstalignment/data/raw"
    processed_root: str = "bstalignment/data/processed"
    output_root: str = "runs/graph_report_ts"
    split: str = "test"
    input_len: int = 96
    label_len: int = 0
    pred_len: int = 20
    history_len: int = 32
    resample_len: int = 128
    early_history_ratio: float = 0.5
    target_col: str = "SOH"
    features: str = "M"
    freq: str = "h"


@dataclass
class ModelConfig:
    model_name: str = "battery_graph_report"  # battery_graph_report or general_graph_report
    d_model: int = 128
    n_heads: int = 4
    graph_layers: int = 2
    dropout: float = 0.1
    patch_size: int = 8
    patch_stride: int = 4
    delay_dim: int = 8
    delay_lag: int = 1
    topk_edges: int = 4
    text_model: str = "distilbert-base-uncased"
    use_hf_text_encoder: bool = True
    freeze_text: bool = True
    text_max_length: int = 192
    use_ic_dv: bool = True
    use_derivative_map: bool = True
    use_hankel_map: bool = True
    use_report_prompt: bool = True
    use_dynamic_graph: bool = True
    use_domain_edges: bool = True
    use_cross_modal_fusion: bool = True
    unified_decoder: bool = True
    use_multi_cycle_raw: bool = True
    use_numeric_history: bool = True
    use_text_gate: bool = True
    use_semantic_alignment: bool = True
    use_relative_steps: bool = True
    temporal_layers: int = 1
    temporal_heads: int = 4


@dataclass
class TrainConfig:
    seed: int = 42
    device: str = "cuda"
    epochs: int = 80
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    early_stop_patience: int = 10
    early_stop_min_delta: float = 1e-5
    loss: str = "smooth_l1"  # smooth_l1, mse, mae
    w_align: float = 0.001
    grad_clip: float = 1.0


@dataclass
class BaselineConfig:
    """Reference baseline source locations and expected local placement.

    External baselines are intentionally not vendored in this repository. Put
    cloned source trees under `external/`; the official baseline trainer imports
    model definitions from those source trees.
    """

    root: str = "external"
    enabled: List[str] = field(default_factory=lambda: ["patchtst", "itransformer", "timecma"])
    sources: Dict[str, str] = field(
        default_factory=lambda: {
            "patchtst": "https://github.com/yuqinie98/PatchTST",
            "itransformer": "https://github.com/thuml/iTransformer",
            "timecma": "https://github.com/ChenxiLiu-HNU/TimeCMA",
            "timesnet": "https://github.com/thuml/Time-Series-Library",
            "dlinear": "https://github.com/cure-lab/LTSF-Linear",
            "time_llm": "https://github.com/KimMeen/Time-LLM",
        }
    )


@dataclass
class ExperimentConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


BATTERY_DATASET_NOTES = {
    "mit": {
        "raw_dir": "bstalignment/data/mit",
        "processed_dir": "bstalignment/data/processed/battery/mit",
        "required": [
            "cell_id",
            "cycle_id",
            "time",
            "current",
            "voltage",
            "temperature",
            "capacity or current-integrated capacity",
            "SOH label",
            "metadata: charge_policy, chemistry if available",
        ],
    },
    "calce": {
        "raw_dir": "bstalignment/data/raw/battery/calce",
        "processed_dir": "bstalignment/data/processed/battery/calce",
        "required": [
            "cells 35, 36, 37, 38",
            "per-cycle current/voltage/temperature/time sequences",
            "charge/discharge phase markers or inferred phase split",
            "capacity and SOH labels",
            "precomputed IC dQ/dV and DV dV/dQ after smoothing",
        ],
    },
    "xjtu": {
        "raw_dir": "bstalignment/data/raw/battery/xjtu",
        "processed_dir": "bstalignment/data/processed/battery/xjtu",
        "required": [
            "cell_id and cycle_id index",
            "per-cycle current/voltage/temperature/time sequences",
            "resampled I/V/T/Q arrays",
            "SOH labels",
            "working-condition metadata if available",
        ],
    },
}


GENERAL_DATASET_NOTES = {
    name: {
        "raw_dir": f"bstalignment/data/raw/general/{name}",
        "processed_dir": f"bstalignment/data/processed/general/{name}",
        "required": [
            "CSV with timestamp column",
            "numeric covariate columns",
            "target column or all-variable multivariate target",
            "train/val/test split compatible with TimeCMA if possible",
        ],
    }
    for name in ["ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "FRED", "ILI", "Weather"]
}


def ensure_research_dirs(cfg: ExperimentConfig) -> None:
    for path in [
        cfg.data.raw_root,
        cfg.data.processed_root,
        cfg.data.output_root,
        cfg.baselines.root,
    ]:
        Path(path).mkdir(parents=True, exist_ok=True)
