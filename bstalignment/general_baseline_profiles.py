"""Pinned, source-derived profiles for formal general forecasting baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any, Mapping


FORMAL_DATASETS = ("ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather")
FORMAL_HORIZONS = (96, 192, 336, 720)


@dataclass(frozen=True)
class SourceIdentity:
    name: str
    url: str
    commit: str
    repo_dir: str
    source_subdir: str
    module: str
    class_name: str
    prompt_policy: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainingMechanics:
    optimizer: str
    loss: str
    lr: float
    weight_decay: float
    scheduler: str
    scheduler_step: str
    max_epochs: int
    early_stop_patience: int
    batch_size: int
    early_stop_start_epoch: int = 1
    pct_start: float | None = None
    cosine_t_max: int | None = None
    eta_min: float = 0.0
    gradient_clip: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneralBaselineProfile:
    name: str
    dataset: str
    pred_len: int
    source: SourceIdentity
    training: TrainingMechanics
    architecture_items: tuple[tuple[str, Any], ...]
    source_evidence: tuple[str, ...]
    seq_len: int = 36
    features: str = "M"
    label_len: int | None = None
    patch_adjustments: tuple[str, ...] = ()
    precision: str = "float32"
    protocol_override_items: tuple[tuple[str, Any], ...] = (
        ("seq_len", 36),
        ("features", "M"),
        ("split_scaler", "Task3 shared train-only scaler"),
        ("checkpoint_selection", "validation_mse"),
    )

    @property
    def architecture(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.architecture_items))

    @property
    def protocol_overrides(self) -> Mapping[str, Any]:
        return MappingProxyType(dict(self.protocol_override_items))


SOURCES = MappingProxyType(
    {
        "PatchTST": SourceIdentity(
            "PatchTST", "https://github.com/yuqinie98/PatchTST", "204c21e", "patchtst",
            "PatchTST_supervised", "models.PatchTST", "Model", "none",
        ),
        "iTransformer": SourceIdentity(
            "iTransformer", "https://github.com/thuml/iTransformer", "c2426e6", "itransformer",
            "", "model.iTransformer", "Model", "none",
        ),
        "TimeCMA": SourceIdentity(
            "TimeCMA", "https://github.com/ChenxiLiu-HNU/TimeCMA", "223e4ae", "timecma",
            "", "models.TimeCMA", "Dual", "timecma",
        ),
        "TimesNet": SourceIdentity(
            "TimesNet", "https://github.com/thuml/Time-Series-Library", "4e938a1", "timesnet",
            "", "models.TimesNet", "Model", "none",
        ),
        "DLinear": SourceIdentity(
            "DLinear", "https://github.com/cure-lab/LTSF-Linear", "0c11366", "dlinear",
            "", "models.DLinear", "Model", "none",
        ),
        "Time-LLM": SourceIdentity(
            "Time-LLM", "https://github.com/KimMeen/Time-LLM", "b13e881", "time_llm",
            "", "models.TimeLLM", "Model", "time_llm",
        ),
    }
)

_ALIASES = {
    "patchtst": "PatchTST",
    "itransformer": "iTransformer",
    "timecma": "TimeCMA",
    "timesnet": "TimesNet",
    "dlinear": "DLinear",
    "time-llm": "Time-LLM",
    "time_llm": "Time-LLM",
    "timellm": "Time-LLM",
}


def canonical_general_baseline_name(name: str) -> str:
    if name in SOURCES:
        return name
    try:
        return _ALIASES[name.lower()]
    except (AttributeError, KeyError) as exc:
        raise ValueError(f"unknown formal general baseline: {name}") from exc


def _items(**values: Any) -> tuple[tuple[str, Any], ...]:
    return tuple(values.items())


def _patchtst_architecture(dataset: str) -> tuple[tuple[str, Any], ...]:
    if dataset in {"ETTh1", "ETTh2"}:
        return _items(e_layers=3, n_heads=4, d_model=16, d_ff=128, dropout=0.3,
                      fc_dropout=0.3, head_dropout=0.0, patch_len=16, stride=8,
                      padding_patch="end", individual=False, revin=True, affine=False,
                      subtract_last=False, decomposition=False, kernel_size=25)
    return _items(e_layers=3, n_heads=16, d_model=128, d_ff=256, dropout=0.2,
                  fc_dropout=0.2, head_dropout=0.0, patch_len=16, stride=8,
                  padding_patch="end", individual=False, revin=True, affine=False,
                  subtract_last=False, decomposition=False, kernel_size=25)


def _patchtst_training(dataset: str) -> TrainingMechanics:
    if dataset in {"ETTm1", "ETTm2"}:
        return TrainingMechanics("adam", "mse", 1e-4, 0.0, "one_cycle", "batch", 100, 20, 128, pct_start=0.4)
    if dataset == "ECL":
        return TrainingMechanics("adam", "mse", 1e-4, 0.0, "one_cycle", "batch", 100, 10, 32, pct_start=0.2)
    patience = 20 if dataset == "Weather" else 100
    return TrainingMechanics("adam", "mse", 1e-4, 0.0, "type3", "epoch", 100, patience, 128)


def _itransformer_architecture(dataset: str, pred_len: int) -> tuple[tuple[str, Any], ...]:
    if dataset == "ETTh1":
        d_model = 256 if pred_len in {96, 192} else 512
        e_layers = 2
    elif dataset in {"ETTh2", "ETTm1", "ETTm2"}:
        d_model, e_layers = 128, 2
    else:
        d_model, e_layers = 512, 3
    return _items(e_layers=e_layers, n_heads=8, d_model=d_model, d_ff=d_model,
                  dropout=0.1, factor=1, embed="timeF", freq="h", activation="gelu",
                  output_attention=False, use_norm=True, class_strategy="projection")


def _itransformer_training(dataset: str) -> TrainingMechanics:
    lr, batch = (5e-4, 16) if dataset == "ECL" else (1e-4, 32)
    return TrainingMechanics("adam", "mse", lr, 0.0, "type1", "epoch", 10, 3, batch)


_TIMESNET_DIMS = {
    "ETTh1": {horizon: (16, 32) for horizon in FORMAL_HORIZONS},
    "ETTh2": {horizon: (32, 32) for horizon in FORMAL_HORIZONS},
    "ETTm1": {96: (64, 64), 192: (64, 64), 336: (16, 32), 720: (16, 32)},
    "ETTm2": {96: (32, 32), 192: (32, 32), 336: (32, 32), 720: (16, 32)},
    "ECL": {horizon: (256, 512) for horizon in FORMAL_HORIZONS},
    "Weather": {horizon: (32, 32) for horizon in FORMAL_HORIZONS},
}
_TIMESNET_EPOCHS = {
    ("ETTm1", 336): 3,
    ("ETTm2", 192): 1,
    ("ETTm2", 720): 1,
    ("Weather", 192): 1,
    ("Weather", 720): 1,
}


def _timesnet_architecture(dataset: str, pred_len: int) -> tuple[tuple[str, Any], ...]:
    d_model, d_ff = _TIMESNET_DIMS[dataset][pred_len]
    return _items(e_layers=2, d_layers=1, d_model=d_model, d_ff=d_ff, top_k=5,
                  num_kernels=6, n_heads=8, dropout=0.1, embed="timeF", freq="h",
                  task_name="long_term_forecast")


def _timesnet_training(dataset: str, pred_len: int) -> TrainingMechanics:
    return TrainingMechanics(
        "adam", "mse", 1e-4, 0.0, "type1", "epoch",
        _TIMESNET_EPOCHS.get((dataset, pred_len), 10), 3, 32,
    )


_DLINEAR_LR = {
    "ETTh1": {horizon: 0.005 for horizon in FORMAL_HORIZONS},
    "ETTh2": {horizon: 0.05 for horizon in FORMAL_HORIZONS},
    "ETTm1": {horizon: 0.0001 for horizon in FORMAL_HORIZONS},
    "ETTm2": {96: 0.001, 192: 0.001, 336: 0.01, 720: 0.01},
    "ECL": {horizon: 0.001 for horizon in FORMAL_HORIZONS},
    "Weather": {horizon: 0.0001 for horizon in FORMAL_HORIZONS},
}
_DLINEAR_BATCH = {"ETTh1": 32, "ETTh2": 32, "ETTm1": 8, "ETTm2": 32, "ECL": 16, "Weather": 16}


def _dlinear_training(dataset: str, pred_len: int) -> TrainingMechanics:
    return TrainingMechanics("adam", "mse", _DLINEAR_LR[dataset][pred_len], 0.0,
                             "type1", "epoch", 10, 3, _DLINEAR_BATCH[dataset])


_TIMECMA_ARCH = {
    "ETTm1": {96: (64, 2, 2, 0.5), 192: (64, 2, 2, 0.5), 336: (64, 2, 2, 0.5), 720: (64, 2, 2, 0.7)},
    "ETTm2": {horizon: (64, 2, 2, 0.3) for horizon in FORMAL_HORIZONS},
    "ETTh1": {96: (64, 1, 2, 0.7), 192: (64, 1, 2, 0.7), 336: (64, 1, 2, 0.7), 720: (32, 2, 2, 0.8)},
    "ETTh2": {horizon: (64, 2, 2, 0.3) for horizon in FORMAL_HORIZONS},
    "ECL": {96: (128, 3, 6, 0.3), 192: (128, 3, 6, 0.3), 336: (128, 3, 6, 0.1), 720: (128, 3, 6, 0.1)},
    "Weather": {96: (64, 6, 2, 0.1), 192: (32, 1, 2, 0.1), 336: (32, 1, 2, 0.1), 720: (32, 1, 1, 0.1)},
}


def _timecma_architecture(dataset: str, pred_len: int) -> tuple[tuple[str, Any], ...]:
    channel, e_layers, d_layers, dropout = _TIMECMA_ARCH[dataset][pred_len]
    return _items(channel=channel, e_layers=e_layers, d_layers=d_layers, dropout=dropout,
                  d_llm=768, d_ff=32, n_heads=8)


def _timecma_training(dataset: str, pred_len: int) -> TrainingMechanics:
    if dataset == "Weather":
        epochs = 20 if pred_len == 96 else 100
        lr, batch = (1e-3, 32) if pred_len == 96 else (1e-4, 32)
    else:
        epochs = 999
        lr = 1e-3 if dataset == "ECL" else 1e-4
        batch = 8 if dataset == "ECL" else 16
    return TrainingMechanics(
        "adamw", "mse", lr, 1e-3, "cosine", "epoch", epochs, 50, batch,
        early_stop_start_epoch=epochs // 2, cosine_t_max=min(epochs, 50),
        eta_min=1e-6, gradient_clip=5.0,
    )


_TIMELLM_SETTINGS = {
    "ETTm1": {h: (32, 128, 0.001, 100, "one_cycle", 24) for h in FORMAL_HORIZONS},
    "ETTm2": {
        96: (32, 128, 0.01, 10, "type1", 16),
        192: (32, 128, 0.002, 10, "one_cycle", 24),
        336: (32, 128, 0.002, 10, "one_cycle", 24),
        720: (32, 128, 0.002, 10, "one_cycle", 24),
    },
    "ETTh1": {
        96: (32, 128, 0.01, 100, "type1", 24),
        192: (32, 128, 0.02, 100, "type1", 24),
        336: (32, 128, 0.001, 100, "cosine", 24),
        720: (32, 128, 0.01, 100, "type1", 24),
    },
    "ETTh2": {
        96: (32, 128, 0.01, 10, "type1", 24),
        192: (32, 128, 0.002, 10, "one_cycle", 24),
        336: (32, 128, 0.005, 10, "one_cycle", 24),
        720: (16, 128, 0.005, 20, "one_cycle", 24),
    },
    "ECL": {h: (16, 32, 0.01, 10, "type1", 24) for h in FORMAL_HORIZONS},
    "Weather": {
        96: (32, 32, 0.01, 10, "type1", 24),
        192: (32, 32, 0.01, 10, "type1", 24),
        336: (32, 128, 0.01, 10, "type1", 24),
        720: (32, 128, 0.01, 15, "type1", 24),
    },
}


def _timellm_architecture(dataset: str, pred_len: int) -> tuple[tuple[str, Any], ...]:
    d_model, d_ff, _, _, _, _ = _TIMELLM_SETTINGS[dataset][pred_len]
    return _items(d_model=d_model, d_ff=d_ff, n_heads=8, e_layers=2, d_layers=1,
                  factor=1, dropout=0.1, patch_len=16, stride=8, top_k=5,
                  llm_model="LLAMA", llm_model_id="huggyllama/llama-7b",
                  llm_dim=4096, llm_layers=32, prompt_domain=1,
                  model_revision="required-at-runtime", tokenizer_revision="required-at-runtime")


def _timellm_training(dataset: str, pred_len: int) -> TrainingMechanics:
    _, _, lr, epochs, scheduler, batch = _TIMELLM_SETTINGS[dataset][pred_len]
    return TrainingMechanics(
        "adam", "mse", lr, 0.0, scheduler, "batch" if scheduler == "one_cycle" else "epoch",
        epochs, 10, batch, pct_start=0.2 if scheduler == "one_cycle" else None,
        cosine_t_max=20 if scheduler == "cosine" else None,
        eta_min=1e-8 if scheduler == "cosine" else 0.0,
    )


_DESCRIPTIONS = {
    "ETT": (
        "The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment. "
        "This dataset consists of 2 years data from two separated counties in China. To explore the granularity on the Long "
        "sequence time-series forecasting (LSTF) problem, different subsets are created, {ETTh1, ETTh2} for 1-hour-level and "
        "ETTm1 for 15-minutes-level. Each data point consists of the target value ”oil temperature” and 6 power load features. "
        "The train/val/test is 12/4/4 months.\n"
    ),
    "ECL": (
        "Measurements of electric power consumption in one household with a one-minute sampling rate over a period of almost "
        "4 years. Different electrical quantities and some sub-metering values are available.This archive contains 2075259 "
        "measurements gathered in a house located in Sceaux (7km of Paris, France) between December 2006 and November 2010 "
        "(47 months)."
    ),
    "Weather": "Weather is recorded every 10 minutes for the 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc.",
}


def time_llm_description(dataset: str) -> str:
    if dataset in {"ETTm1", "ETTm2", "ETTh1", "ETTh2"}:
        return _DESCRIPTIONS["ETT"]
    try:
        return _DESCRIPTIONS[dataset]
    except KeyError as exc:
        raise ValueError(f"unknown formal general dataset: {dataset}") from exc


def resolve_general_profile(name: str, dataset: str, pred_len: int) -> GeneralBaselineProfile:
    canonical = canonical_general_baseline_name(name)
    if dataset not in FORMAL_DATASETS:
        raise ValueError(f"unknown formal general dataset: {dataset}")
    if pred_len not in FORMAL_HORIZONS:
        raise ValueError(f"unsupported formal prediction length: {pred_len}")

    if canonical == "PatchTST":
        architecture, training = _patchtst_architecture(dataset), _patchtst_training(dataset)
        evidence = (f"PatchTST_supervised/scripts/PatchTST/{'electricity' if dataset == 'ECL' else dataset.lower()}.sh",
                    "PatchTST_supervised/exp/exp_main.py:47-52,112-124,193-212",
                    "PatchTST_supervised/run_longExp.py:78-85")
    elif canonical == "iTransformer":
        architecture, training = _itransformer_architecture(dataset, pred_len), _itransformer_training(dataset)
        evidence = ("scripts/multivariate_forecasting: audited dataset script", "experiments/exp_long_term_forecasting.py:32-37,94-177", "run.py:62-68")
    elif canonical == "TimesNet":
        architecture, training = _timesnet_architecture(dataset, pred_len), _timesnet_training(dataset, pred_len)
        evidence = ("scripts/long_term_forecast: audited TimesNet dataset script", "exp/exp_long_term_forecasting.py:34-39,88-161", "run.py:57,90-96")
    elif canonical == "DLinear":
        architecture, training = _items(individual=False, moving_avg=25), _dlinear_training(dataset, pred_len)
        evidence = (f"scripts/EXP-LongForecasting/Linear/{'electricity' if dataset == 'ECL' else dataset.lower()}.sh",
                    "models/DLinear.py:38-87", "exp/exp_main.py:46-51,112-205", "run_longExp.py:66-72")
    elif canonical == "TimeCMA":
        architecture, training = _timecma_architecture(dataset, pred_len), _timecma_training(dataset, pred_len)
        evidence = (f"scripts/{dataset}.sh: audited horizon block", "models/TimeCMA.py:6-102", "train.py:17-49,69-90,233-299")
    else:
        architecture, training = _timellm_architecture(dataset, pred_len), _timellm_training(dataset, pred_len)
        evidence = (f"scripts/TimeLLM_{'ECL' if dataset == 'ECL' else dataset}.sh: audited horizon block",
                    "models/TimeLLM.py:30-197,200-264", "run_main.py:55-99,145-164,236-264")

    return GeneralBaselineProfile(
        name=canonical,
        dataset=dataset,
        pred_len=pred_len,
        source=SOURCES[canonical],
        training=training,
        architecture_items=architecture,
        source_evidence=evidence,
        precision="bf16" if canonical == "Time-LLM" else "float32",
    )
