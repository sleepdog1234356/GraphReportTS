from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..general_protocol import StandardScalerNP, window_spec
from .contracts import GraphReportTSv2Config
from .general_data import GeneralForecastV2Dataset, SyntheticGeneralV2Dataset, collate_general_v2
from .losses import general_v2_loss
from .model import GraphReportTSv2
from .prompt_hidden_cache import (
    ensure_prompt_hidden_split_cache,
    install_prompt_hidden_caches,
    prompt_hidden_base_identity,
)
from .results import make_result_record, stable_digest, write_result
from .semantic import FrozenDistilBERTEncoder
from .training import (
    V2TrainingConfig,
    alignment_weight,
    apply_warmup,
    build_optimizer,
    build_plateau_scheduler,
    optimizer_learning_rates,
    save_checkpoint,
    should_stop_v2,
    step_plateau_scheduler,
)


GENERAL_TRAINING_STRATEGY_VERSION = "v2-general-adaptive-sparse-graph-compact-prompt-v3"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphReportTS-v2 on general multivariate datasets")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_root", default="data/general")
    parser.add_argument("--provenance_manifest", default=None)
    parser.add_argument("--input_len", type=int, choices=(36, 96), default=36)
    parser.add_argument("--prompt_len", type=int, choices=(36, 96), default=36)
    parser.add_argument("--horizon", type=int, choices=(24, 36, 48, 60, 96, 192, 336, 720), default=24)
    parser.add_argument("--mode", choices=("quick", "preflight", "formal"), default="quick")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--pin_memory", action="store_true")
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--init_model_checkpoint",
        default=None,
        help="load model weights and validation baseline only, while resetting all training state",
    )
    parser.add_argument("--collect_branch_gradient_norms", action="store_true")
    parser.add_argument("--gradient_diagnostic_interval", type=int, default=128)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--text_model", default="hf_models/distilbert-base-uncased")
    parser.add_argument("--text_backend", choices=("distilbert", "simple"), default="distilbert")
    parser.add_argument("--text_hidden_cache_size", type=int, default=4_096)
    parser.add_argument("--text_hidden_cache_max_bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--text_hidden_precompute_cache",
        "--text-hidden-precompute-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--text_hidden_cache_root",
        default="data/general/cache/distilbert_hidden",
    )
    parser.add_argument("--text_hidden_precompute_batch_size", type=int, default=64)
    parser.add_argument("--prompt_build_workers", type=int, default=0)
    parser.add_argument("--no_text", action="store_true")
    parser.add_argument("--linear_anchor_init", choices=("random", "ridge"), default="random")
    parser.add_argument("--linear_anchor_ridge", type=float, default=1e-4)
    parser.add_argument("--freeze_linear_anchor", action="store_true")
    parser.add_argument(
        "--correction_gate_mode",
        choices=("trainable", "fixed_one"),
        default="trainable",
    )
    parser.add_argument("--model_name", default="GraphReportTS-v2")
    parser.add_argument("--training_protocol", default=GENERAL_TRAINING_STRATEGY_VERSION)
    parser.add_argument("--core_lr", type=float)
    parser.add_argument("--semantic_lr", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--align_start_epoch", type=int)
    parser.add_argument("--align_full_epoch", type=int)
    parser.add_argument("--align_weight", type=float)
    parser.add_argument("--gradient_clip", type=float)
    parser.add_argument("--plateau_factor", type=float)
    parser.add_argument("--plateau_patience", type=int)
    parser.add_argument("--plateau_threshold", type=float)
    parser.add_argument("--plateau_cooldown", type=int)
    parser.add_argument("--core_min_lr", type=float)
    parser.add_argument("--semantic_min_lr", type=float)
    parser.add_argument("--min_lr_reductions_before_stop", type=int)
    parser.add_argument(
        "--graph_embedding_variant",
        choices=(
            "patch",
            "series_context",
            "series_context_diff",
            "series_context_decomp",
            "global_node",
            "global_node_diff",
            "global_node_raw",
        ),
        default="patch",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--synthetic_size", type=int, default=24)
    parser.add_argument("--max_train_batches", type=int, default=None)
    parser.add_argument("--max_eval_batches", type=int, default=None)
    parser.add_argument(
        "--skip_final_test",
        action="store_true",
        help="stop after training/validation and checkpoint persistence; intended for validation-only pilots",
    )
    args = parser.parse_args(argv)
    try:
        spec = window_spec(args.input_len, args.prompt_len)
    except ValueError as exc:
        parser.error(str(exc))
    if args.horizon not in spec.horizons:
        parser.error(
            f"horizon {args.horizon} is incompatible with input_len={args.input_len}, "
            f"prompt_len={args.prompt_len}; expected one of {spec.horizons}"
        )
    if args.mode == "quick" and (args.input_len, args.prompt_len, args.horizon) != (36, 36, 24):
        parser.error("quick mode supports only the legacy L36/P36/H24 smoke protocol")
    if args.mode == "formal" and (args.no_text or args.text_backend != "distilbert"):
        parser.error("formal mode requires the local DistilBERT text branch")
    if args.persistent_workers and args.num_workers == 0:
        parser.error("--persistent_workers requires --num_workers > 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch_factor must be positive")
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch_size must be positive and num_workers non-negative")
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be positive")
    if args.resume and args.init_model_checkpoint:
        parser.error("--resume and --init_model_checkpoint are mutually exclusive")
    if args.gradient_diagnostic_interval < 1:
        parser.error("--gradient_diagnostic_interval must be positive")
    if args.linear_anchor_ridge <= 0:
        parser.error("--linear_anchor_ridge must be positive")
    if not args.model_name.strip() or not args.training_protocol.strip():
        parser.error("--model_name and --training_protocol must not be empty")
    positive_optional = (
        "core_lr",
        "semantic_lr",
        "weight_decay",
        "gradient_clip",
        "plateau_factor",
        "plateau_threshold",
        "core_min_lr",
        "semantic_min_lr",
    )
    for name in positive_optional:
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name} must be positive")
    nonnegative_optional = (
        "warmup_epochs",
        "align_start_epoch",
        "align_full_epoch",
        "align_weight",
        "plateau_patience",
        "plateau_cooldown",
        "min_lr_reductions_before_stop",
    )
    for name in nonnegative_optional:
        value = getattr(args, name)
        if value is not None and value < 0:
            parser.error(f"--{name} must be non-negative")
    if args.text_hidden_cache_size < 0 or args.text_hidden_cache_max_bytes < 0:
        parser.error("text hidden-cache limits must be non-negative")
    if args.text_hidden_precompute_batch_size < 1:
        parser.error("--text_hidden_precompute_batch_size must be positive")
    if args.prompt_build_workers < 0:
        parser.error("--prompt_build_workers must be non-negative")
    if args.text_hidden_precompute_cache and not args.text_hidden_cache_root:
        parser.error("--text_hidden_cache_root must not be empty")
    if args.freeze_linear_anchor and args.linear_anchor_init != "ridge":
        parser.error("--freeze_linear_anchor requires --linear_anchor_init ridge")
    for name in ("max_train_batches", "max_eval_batches"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be positive")
    if args.mode == "formal" and (args.max_train_batches is not None or args.max_eval_batches is not None):
        parser.error("batch limits are preflight-only and cannot be used in formal mode")
    if args.mode == "formal" and not args.provenance_manifest:
        parser.error("formal mode requires --provenance_manifest")
    return args


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict[str, object]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict[str, object] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


MappingLike = dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


LINEAR_ANCHOR_CONFIG_SCHEMA = "graph-report-ts-v2-linear-anchor-v1"


def linear_anchor_settings(initialization: str, ridge: float, frozen: bool) -> dict[str, object]:
    if initialization not in ("random", "ridge"):
        raise ValueError(f"unsupported linear anchor initialization: {initialization}")
    if float(ridge) <= 0:
        raise ValueError("linear anchor ridge regularization must be positive")
    if frozen and initialization != "ridge":
        raise ValueError("a frozen linear anchor requires ridge initialization")
    return {
        "schema": LINEAR_ANCHOR_CONFIG_SCHEMA,
        "initialization": initialization,
        "ridge": float(ridge) if initialization == "ridge" else None,
        "frozen": bool(frozen),
    }


def validate_resume_linear_anchor(saved: object, current: MappingLike) -> None:
    if saved is None:
        legacy_default = linear_anchor_settings("random", 1e-4, False)
        if current != legacy_default:
            raise ValueError(
                "resume checkpoint predates linear-anchor settings and is only compatible with "
                "--linear_anchor_init random without --freeze_linear_anchor"
            )
        return
    if not isinstance(saved, dict) or saved != current:
        raise ValueError("resume checkpoint linear-anchor settings do not match this run")


def fit_shared_ridge_anchor(
    values: np.ndarray,
    samples: Sequence[int],
    *,
    input_len: int,
    horizon: int,
    ridge: float = 1e-4,
    chunk_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one train-only linear map shared by every variable.

    Only standardized ``values`` and the training split's numeric-window
    ``samples`` are consumed. Sufficient statistics keep wide datasets such as
    ECL from materializing every variable window at once.
    """

    series = np.asarray(values)
    starts = np.asarray(list(samples), dtype=np.int64)
    if series.ndim != 2:
        raise ValueError("linear anchor values must be [time, variables]")
    if starts.ndim != 1 or starts.size == 0:
        raise ValueError("linear anchor requires at least one training sample")
    if input_len < 1 or horizon < 1:
        raise ValueError("linear anchor input_len and horizon must be positive")
    if float(ridge) <= 0:
        raise ValueError("linear anchor ridge regularization must be positive")
    if chunk_size < 1:
        raise ValueError("linear anchor chunk_size must be positive")
    if np.any(starts < 0) or np.any(starts + input_len + horizon > series.shape[0]):
        raise ValueError("linear anchor training sample exceeds the provided series")

    augmented_size = input_len + 1
    gram = np.zeros((augmented_size, augmented_size), dtype=np.float64)
    rhs = np.zeros((augmented_size, horizon), dtype=np.float64)
    fitted_rows = 0
    for offset in range(0, starts.size, chunk_size):
        chunk = starts[offset : offset + chunk_size]
        history = np.stack(
            [series[start : start + input_len].T for start in chunk], axis=0
        ).reshape(-1, input_len)
        target = np.stack(
            [series[start + input_len : start + input_len + horizon].T for start in chunk], axis=0
        ).reshape(-1, horizon)
        history = np.asarray(history, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if not np.isfinite(history).all() or not np.isfinite(target).all():
            raise ValueError("linear anchor training windows must be finite")

        history_sum = history.sum(axis=0)
        gram[:input_len, :input_len] += history.T @ history
        gram[:input_len, input_len] += history_sum
        gram[input_len, :input_len] += history_sum
        gram[input_len, input_len] += history.shape[0]
        rhs[:input_len] += history.T @ target
        rhs[input_len] += target.sum(axis=0)
        fitted_rows += history.shape[0]

    if fitted_rows == 0:
        raise ValueError("linear anchor did not receive any variable windows")
    gram[np.arange(input_len), np.arange(input_len)] += float(ridge)
    solution = np.linalg.solve(gram, rhs)
    weight = solution[:input_len].T.astype(np.float32, copy=False)
    bias = solution[input_len].astype(np.float32, copy=False)
    return weight, bias


def write_linear_anchor(
    anchor: torch.nn.Linear,
    weight: np.ndarray,
    bias: np.ndarray,
    *,
    horizon: int,
) -> None:
    if anchor.bias is None:
        raise ValueError("linear anchor requires an intercept parameter")
    if horizon < 1 or horizon > anchor.out_features:
        raise ValueError("linear anchor horizon exceeds the output width")
    expected_weight = (horizon, anchor.in_features)
    if tuple(weight.shape) != expected_weight or tuple(bias.shape) != (horizon,):
        raise ValueError(
            f"linear anchor coefficients must be {expected_weight} with bias {(horizon,)}"
        )
    with torch.no_grad():
        anchor.weight[:horizon].copy_(
            torch.as_tensor(weight, device=anchor.weight.device, dtype=anchor.weight.dtype)
        )
        anchor.bias[:horizon].copy_(
            torch.as_tensor(bias, device=anchor.bias.device, dtype=anchor.bias.dtype)
        )


def freeze_linear_anchor(anchor: torch.nn.Linear, *, frozen: bool) -> None:
    anchor.weight.requires_grad_(not frozen)
    if anchor.bias is not None:
        anchor.bias.requires_grad_(not frozen)


def _load_project_identity(path: str | None, *, required: bool) -> dict[str, Any]:
    if path is None:
        if required:
            raise ValueError("formal mode requires a provenance manifest with project identity")
        return {
            "name": "GraphReportTS",
            "identity_kind": "unrecorded_quick_run",
            "git_commit": None,
            "tree_sha256": "unrecorded",
        }
    manifest_path = Path(path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"provenance manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("provenance manifest does not contain project identity")
    missing = [name for name in ("name", "identity_kind", "tree_sha256") if not project.get(name)]
    if missing:
        raise ValueError(f"project identity is missing required fields: {', '.join(missing)}")
    identity = dict(project)
    identity["manifest_path"] = str(manifest_path)
    identity["manifest_sha256"] = stable_digest(payload)
    return identity


def _identity_matches(saved: object, current: MappingLike) -> bool:
    if not isinstance(saved, dict):
        return False
    ignored = {"manifest_path", "manifest_sha256"}
    saved_identity = {key: value for key, value in saved.items() if key not in ignored}
    current_identity = {key: value for key, value in current.items() if key not in ignored}
    return stable_digest(saved_identity) == stable_digest(current_identity)


def _portable_dataset_identity(identity: MappingLike) -> dict[str, Any]:
    """Ignore only the relocated CSV path while retaining every content identity field."""

    portable = dict(identity)
    source = portable.get("source_csv")
    if isinstance(source, dict):
        source = dict(source)
        source.pop("path", None)
        portable["source_csv"] = source
    return portable


def _dataset_identity_matches(saved: object, current: MappingLike) -> bool:
    if not isinstance(saved, dict):
        return False
    return stable_digest(_portable_dataset_identity(saved)) == stable_digest(
        _portable_dataset_identity(current)
    )


def _validate_model_checkpoint(
    checkpoint: MappingLike,
    *,
    args: argparse.Namespace,
    config: GraphReportTSv2Config,
    anchor_settings: MappingLike,
    project_identity: MappingLike,
    dataset_identity: MappingLike,
) -> None:
    validate_resume_linear_anchor(checkpoint.get("linear_anchor"), anchor_settings)
    saved_config = checkpoint.get("config")
    if not isinstance(saved_config, dict):
        raise ValueError("model-only checkpoint is missing model config")
    expected = {
        "domain": "general",
        "input_len": args.input_len,
        "pred_len": args.horizon,
        "max_nodes": config.max_nodes,
        "graph_embedding_variant": args.graph_embedding_variant,
    }
    for key, value in expected.items():
        if saved_config.get(key) != value:
            raise ValueError(
                f"model-only checkpoint {key} does not match this run: "
                f"{saved_config.get(key)!r} != {value!r}"
            )
    saved_project = checkpoint.get("project_identity")
    if not _identity_matches(saved_project, project_identity):
        raise ValueError("model-only checkpoint project identity does not match this run")
    saved_dataset = checkpoint.get("dataset_identity")
    if not _dataset_identity_matches(saved_dataset, dataset_identity):
        raise ValueError("model-only checkpoint dataset identity does not match this run")


def _parameter_branch(name: str) -> str | None:
    if name.startswith(("patchifier.", "edge_builder.", "graph_mixer.")):
        return "graph"
    if name.startswith(("text_encoder.proj.", "semantic_fusion.")):
        return "semantic"
    if name.startswith("head.") and not name.startswith("head.linear_anchor."):
        return "head"
    return None


def _sample_branch_gradient_norms(model: GraphReportTSv2) -> dict[str, float]:
    squared = {"graph": 0.0, "semantic": 0.0, "head": 0.0}
    for name, parameter in model.named_parameters():
        branch = _parameter_branch(name)
        if branch is None or parameter.grad is None:
            continue
        squared[branch] += float(parameter.grad.detach().float().square().sum().cpu())
    return {f"{branch}_gradient_norm": value ** 0.5 for branch, value in squared.items()}


def _gate_statistics(metrics: MappingLike) -> dict[str, float]:
    return {
        "semantic_gate_mean": float(metrics["gate"]),
        "correction_gate_mean": float(metrics["correction_gate"]),
    }


def _runtime_metadata(device: torch.device, amp: str) -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "device": str(device),
        "amp": amp,
        "cuda_device_name": None,
        "peak_memory_allocated_bytes": 0,
        "peak_memory_reserved_bytes": 0,
    }
    if device.type == "cuda":
        runtime.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return runtime


def build_loaders(
    args: argparse.Namespace,
    dataset_transform=None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    if args.dataset.lower() == "synthetic":
        train = SyntheticGeneralV2Dataset(
            args.synthetic_size,
            7,
            args.horizon,
            args.seed,
            input_len=args.input_len,
            prompt_len=args.prompt_len,
        )
        val = SyntheticGeneralV2Dataset(
            max(4, args.synthetic_size // 4),
            7,
            args.horizon,
            args.seed + 1,
            input_len=args.input_len,
            prompt_len=args.prompt_len,
        )
        test = SyntheticGeneralV2Dataset(
            max(4, args.synthetic_size // 4),
            7,
            args.horizon,
            args.seed + 2,
            input_len=args.input_len,
            prompt_len=args.prompt_len,
        )
    else:
        scaler = StandardScalerNP()
        train = GeneralForecastV2Dataset(
            args.dataset,
            args.data_root,
            "train",
            args.horizon,
            scaler,
            fit_scaler=True,
            fit_prompt_thresholds=True,
            input_len=args.input_len,
            prompt_len=args.prompt_len,
        )
        shared = {
            "fill_values": train.fill_values,
            "prompt_thresholds": train.prompt_thresholds,
            "input_len": args.input_len,
            "prompt_len": args.prompt_len,
        }
        val = GeneralForecastV2Dataset(
            args.dataset, args.data_root, "val", args.horizon, scaler, **shared
        )
        test = GeneralForecastV2Dataset(
            args.dataset, args.data_root, "test", args.horizon, scaler, **shared
        )
    if dataset_transform is not None:
        train = dataset_transform(train, "train")
        val = dataset_transform(val, "val")
        test = dataset_transform(test, "test")
    common = dict(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_general_v2,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
    )
    if args.num_workers > 0:
        common["prefetch_factor"] = args.prefetch_factor
    return (
        DataLoader(train, shuffle=True, **common),
        DataLoader(val, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )


def _dataset_prompts(dataset: object, workers: int = 0) -> list[str]:
    """Read only already-defined causal prompts from one dataset split."""

    prompt_at = getattr(dataset, "prompt_at", None)
    if callable(prompt_at):
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="general-prompt") as pool:
                return [str(prompt) for prompt in pool.map(prompt_at, range(len(dataset)))]
        return [str(prompt_at(index)) for index in range(len(dataset))]
    return [str(dataset[index]["prompt"]) for index in range(len(dataset))]


def prepare_general_prompt_hidden_cache(
    model: GraphReportTSv2,
    loaders: Sequence[DataLoader],
    *,
    args: argparse.Namespace,
    preprocessing: MappingLike,
) -> dict[str, object]:
    """Precompute frozen DistilBERT states for the constructed train/val/test prompts."""

    if not args.text_hidden_precompute_cache:
        return {"enabled": False, "reason": "disabled_by_cli"}
    if args.no_text:
        return {"enabled": False, "reason": "text_branch_disabled"}
    if args.text_backend != "distilbert":
        return {"enabled": False, "reason": "non_distilbert_backend"}
    encoder = model.text_encoder
    if not isinstance(encoder, FrozenDistilBERTEncoder):
        raise TypeError("persistent prompt cache requires FrozenDistilBERTEncoder")
    prompt_schema = preprocessing.get("prompt_schema")
    dataset_identity = preprocessing.get("dataset_identity")
    if not isinstance(prompt_schema, dict) or not isinstance(dataset_identity, dict):
        raise RuntimeError("prompt hidden-cache provenance requires prompt and dataset identities")
    base_identity = prompt_hidden_base_identity(
        model_path=encoder.model_path,
        text_max_length=encoder.max_length,
        prompt_schema=prompt_schema,
        dataset_identity=dataset_identity,
    )
    manifests: list[dict[str, object]] = []
    split_summaries: dict[str, object] = {}
    for split, loader in zip(("train", "val", "test"), loaders):
        print(
            json.dumps(
                {
                    "event": "prompt_hidden_precompute_start",
                    "split": split,
                    "samples": len(loader.dataset),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        prompts = _dataset_prompts(loader.dataset, workers=args.prompt_build_workers)
        manifest = ensure_prompt_hidden_split_cache(
            encoder,
            prompts,
            split=split,
            cache_root=args.text_hidden_cache_root,
            dataset_name=args.dataset,
            horizon=args.horizon,
            base_identity=base_identity,
            precompute_batch_size=args.text_hidden_precompute_batch_size,
        )
        manifests.append(manifest)
        identity = manifest["identity"]
        split_summaries[split] = {
            "manifest_path": manifest["manifest_path"],
            "identity_sha256": manifest["identity_sha256"],
            "cache_status": manifest["cache_status"],
            "prompt_count": identity["prompt_count"],
            "unique_prompt_count": identity["unique_prompt_count"],
            "prompt_hashes_sha256": identity["prompt_hashes_sha256"],
            "chunks": manifest["chunks"],
        }
        print(
            json.dumps(
                {
                    "event": "prompt_hidden_precompute_complete",
                    "split": split,
                    "cache_status": manifest["cache_status"],
                    "prompt_count": identity["prompt_count"],
                    "unique_prompt_count": identity["unique_prompt_count"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    installed = install_prompt_hidden_caches(encoder, manifests)
    print(
        json.dumps({"event": "prompt_hidden_memory_cache_ready", **installed}, ensure_ascii=False),
        flush=True,
    )
    return {
        "enabled": True,
        "schema": base_identity["schema"],
        "cache_root": str(Path(args.text_hidden_cache_root).expanduser().resolve()),
        "cache_boundary": base_identity["cache_boundary"],
        "storage_dtype": base_identity["storage_dtype"],
        "model": base_identity["model"],
        "text_max_length": base_identity["text_max_length"],
        "prompt_schema_sha256": stable_digest(prompt_schema),
        "dataset_identity_sha256": stable_digest(dataset_identity),
        "splits": split_summaries,
        "memory_cache": installed,
    }


def run_epoch(
    model: GraphReportTSv2,
    loader: DataLoader,
    device: torch.device,
    align: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
    amp_dtype: torch.dtype | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    non_blocking: bool = False,
    max_batches: int | None = None,
    collect_branch_gradient_norms: bool = False,
    gradient_diagnostic_interval: int = 128,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {
        "loss": 0.0,
        "mse": 0.0,
        "mae": 0.0,
        "rmse": 0.0,
        "gate": 0.0,
        "correction_gate": 0.0,
        "residual_abs_mean": 0.0,
    }
    count = 0
    squared_error = 0.0
    absolute_error = 0.0
    observed_targets = 0
    residual_absolute_error = 0.0
    residual_observed_targets = 0
    gradient_sums = {"graph_gradient_norm": 0.0, "semantic_gradient_norm": 0.0, "head_gradient_norm": 0.0}
    gradient_samples = 0
    for batch_index, batch in enumerate(loader, start=1):
        batch = batch.to(device, non_blocking=non_blocking)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                output = model(batch)
                losses = general_v2_loss(output, batch.target, batch.target_mask, align)
            if training:
                if scaler is not None:
                    scaler.scale(losses["total"]).backward()
                    scaler.unscale_(optimizer)
                    if collect_branch_gradient_norms and batch_index % gradient_diagnostic_interval == 0:
                        for key, value in _sample_branch_gradient_norms(model).items():
                            gradient_sums[key] += value
                        gradient_samples += 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["total"].backward()
                    if collect_branch_gradient_norms and batch_index % gradient_diagnostic_interval == 0:
                        for key, value in _sample_branch_gradient_norms(model).items():
                            gradient_sums[key] += value
                        gradient_samples += 1
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()
        prediction = output["pred"]
        if not torch.is_tensor(prediction):
            raise TypeError("model prediction must be a tensor")
        error = (prediction.detach() - batch.target)[batch.target_mask]
        anchor_prediction = model.head.linear_anchor(batch.values.transpose(1, 2))[
            :, :, : model.config.pred_len
        ].transpose(1, 2)
        residual = (prediction.detach() - anchor_prediction.detach())[batch.target_mask]
        squared_error += float(error.square().sum().cpu())
        absolute_error += float(error.abs().sum().cpu())
        observed_targets += int(error.numel())
        residual_absolute_error += float(residual.abs().sum().cpu())
        residual_observed_targets += int(residual.numel())
        totals["loss"] += float(losses["total"].detach().cpu())
        totals["gate"] += float(output["gate"].detach().mean().cpu())
        totals["correction_gate"] += float(output["correction_gate"].detach().mean().cpu())
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    if not count:
        raise RuntimeError("empty data loader")
    if not observed_targets:
        raise RuntimeError("data loader contains no observed targets")
    totals["mse"] = squared_error / observed_targets
    totals["mae"] = absolute_error / observed_targets
    totals["rmse"] = totals["mse"] ** 0.5
    totals["loss"] /= count
    totals["gate"] /= count
    totals["correction_gate"] /= count
    totals["residual_abs_mean"] = residual_absolute_error / max(residual_observed_targets, 1)
    if collect_branch_gradient_norms:
        if gradient_samples == 0:
            # Short preflight runs still need a diagnostic sample.
            for key, value in _sample_branch_gradient_norms(model).items():
                gradient_sums[key] += value
            gradient_samples = 1
        for key, value in gradient_sums.items():
            totals[key] = value / gradient_samples
    return totals


def main(
    argv: list[str] | None = None,
    *,
    model_factory: Any | None = None,
) -> None:
    args = parse_args(argv)
    _seed(args.seed)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if args.amp != "none" and device.type != "cuda":
        raise ValueError("bf16/fp16 AMP requires a CUDA device")
    if args.amp == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    project_identity = _load_project_identity(args.provenance_manifest, required=args.mode == "formal")
    amp_dtype = {"none": None, "bf16": torch.bfloat16, "fp16": torch.float16}[args.amp]
    train_loader, val_loader, test_loader = build_loaders(args)
    dataset = train_loader.dataset
    preprocessing = dataset.preprocessing_state() if hasattr(dataset, "preprocessing_state") else None
    if not isinstance(preprocessing, dict) or not isinstance(preprocessing.get("dataset_identity"), dict):
        raise RuntimeError("general v2 training dataset did not provide dataset identity")
    dataset_identity = dict(preprocessing["dataset_identity"])
    anchor_settings = linear_anchor_settings(
        args.linear_anchor_init,
        args.linear_anchor_ridge,
        args.freeze_linear_anchor,
    )
    anchor_fit: dict[str, object] | None = None
    config = GraphReportTSv2Config(
        domain="general",
        input_len=args.input_len,
        pred_len=args.horizon,
        max_pred_len=max(60, args.horizon),
        max_nodes=14_000 if args.input_len == 96 else 6_000,
        text_model=args.text_model,
        text_backend=args.text_backend,
        text_hidden_cache_size=args.text_hidden_cache_size,
        text_hidden_cache_max_bytes=args.text_hidden_cache_max_bytes,
        use_text=not args.no_text,
        graph_embedding_variant=args.graph_embedding_variant,
        correction_gate_mode=args.correction_gate_mode,
    )
    factory = GraphReportTSv2 if model_factory is None else model_factory
    model = factory(config).to(device)
    prompt_hidden_cache = prepare_general_prompt_hidden_cache(
        model,
        (train_loader, val_loader, test_loader),
        args=args,
        preprocessing=preprocessing,
    )
    linear_anchor = model.head.linear_anchor
    if args.linear_anchor_init == "ridge" and not args.resume and not args.init_model_checkpoint:
        dataset_values = getattr(dataset, "values", None)
        dataset_samples = getattr(dataset, "samples", None)
        if dataset_values is None or dataset_samples is None:
            raise ValueError(
                "ridge linear-anchor initialization requires train_loader.dataset.values and .samples"
            )
        ridge_weight, ridge_bias = fit_shared_ridge_anchor(
            dataset_values,
            dataset_samples,
            input_len=config.input_len,
            horizon=args.horizon,
            ridge=args.linear_anchor_ridge,
        )
        write_linear_anchor(linear_anchor, ridge_weight, ridge_bias, horizon=args.horizon)
        anchor_fit = {
            "training_samples": len(dataset_samples),
            "shared_variable_rows": int(len(dataset_samples) * np.asarray(dataset_values).shape[1]),
            "input_len": config.input_len,
            "horizon": args.horizon,
        }
    freeze_linear_anchor(linear_anchor, frozen=args.freeze_linear_anchor)
    training = V2TrainingConfig(
        epochs=args.epochs or (8 if args.mode == "quick" else (1 if args.mode == "preflight" else 80)),
        patience=args.patience if args.patience is not None else 20,
    )
    training_overrides = {
        name: getattr(args, name)
        for name in (
            "core_lr",
            "semantic_lr",
            "weight_decay",
            "warmup_epochs",
            "align_start_epoch",
            "align_full_epoch",
            "align_weight",
            "gradient_clip",
            "plateau_factor",
            "plateau_patience",
            "plateau_threshold",
            "plateau_cooldown",
            "core_min_lr",
            "semantic_min_lr",
            "min_lr_reductions_before_stop",
        )
        if getattr(args, name) is not None
    }
    if training_overrides:
        training = replace(training, **training_overrides)
    optimizer = build_optimizer(model, training)
    scheduler = build_plateau_scheduler(optimizer, training)
    scaler = torch.cuda.amp.GradScaler(enabled=True) if args.amp == "fp16" else None
    best_mse = float("inf")
    stale = 0
    lr_reductions = 0
    history: list[dict[str, object]] = []
    start_epoch = 1
    best_epoch = 0
    best_gate_statistics: dict[str, float] | None = None
    initialization_audit: dict[str, object] | None = None
    if args.init_model_checkpoint:
        source_path = Path(args.init_model_checkpoint).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"model-only checkpoint not found: {source_path}")
        source = torch.load(source_path, map_location=device, weights_only=False)
        _validate_model_checkpoint(
            source,
            args=args,
            config=config,
            anchor_settings=anchor_settings,
            project_identity=project_identity,
            dataset_identity=dataset_identity,
        )
        model.load_state_dict(source["model"])
        source_best_mse = float(source.get("best_val_mse", source["val_mse"]))
        source_best_epoch = int(source.get("best_epoch", source["epoch"]))
        best_mse = source_best_mse
        saved_anchor_fit = source.get("linear_anchor_fit")
        if isinstance(saved_anchor_fit, dict):
            anchor_fit = dict(saved_anchor_fit)
        initialization_audit = {
            "schema": "graph-report-ts-v2-model-only-initialization-v1",
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": _file_sha256(source_path),
            "source_best_epoch": source_best_epoch,
            "source_best_validation_mse": source_best_mse,
            "optimizer_reset": True,
            "scheduler_reset": True,
            "history_reset": True,
            "rng_reset_to_requested_seed": args.seed,
        }
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        saved_strategy = resume.get("training_strategy_version")
        if saved_strategy != args.training_protocol:
            raise ValueError(
                "resume checkpoint training strategy does not match "
                f"{args.training_protocol}"
            )
        validate_resume_linear_anchor(resume.get("linear_anchor"), anchor_settings)
        saved_config = resume.get("config", {})
        if saved_config:
            expected_resume_config = {
                "domain": "general",
                "input_len": args.input_len,
                "pred_len": args.horizon,
                "max_nodes": config.max_nodes,
                "graph_embedding_variant": args.graph_embedding_variant,
            }
            for key, expected in expected_resume_config.items():
                if saved_config.get(key) != expected:
                    raise ValueError(
                        f"resume checkpoint {key} does not match this run: "
                        f"{saved_config.get(key)!r} != {expected!r}"
                    )
        saved_project = resume.get("project_identity")
        if saved_project is not None and not _identity_matches(saved_project, project_identity):
            raise ValueError("resume checkpoint project identity does not match this run")
        if args.mode == "formal" and saved_project is None:
            raise ValueError("formal resume checkpoint is missing project identity")
        saved_dataset = resume.get("dataset_identity")
        if saved_dataset is not None and not _dataset_identity_matches(saved_dataset, dataset_identity):
            raise ValueError("resume checkpoint dataset identity does not match this run")
        if args.mode == "formal" and saved_dataset is None:
            raise ValueError("formal resume checkpoint is missing dataset identity")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scheduler") is None:
            raise ValueError("resume checkpoint is missing plateau scheduler state")
        scheduler.load_state_dict(resume["scheduler"])
        if scaler is not None and resume.get("amp_scaler") is not None:
            scaler.load_state_dict(resume["amp_scaler"])
        start_epoch = int(resume["epoch"]) + 1
        best_mse = float(resume.get("best_val_mse", resume["val_mse"]))
        best_epoch = int(resume.get("best_epoch", resume["epoch"]))
        saved_gate_statistics = resume.get("best_gate_statistics")
        if isinstance(saved_gate_statistics, dict):
            best_gate_statistics = {str(key): float(value) for key, value in saved_gate_statistics.items()}
        stale = int(resume.get("stale", 0))
        lr_reductions = int(resume.get("lr_reductions", 0))
        history = list(resume.get("history", []))
        saved_anchor_fit = resume.get("linear_anchor_fit")
        if isinstance(saved_anchor_fit, dict):
            anchor_fit = dict(saved_anchor_fit)
        _restore_rng_state(resume.get("rng_state"))
    run_config: dict[str, Any] = {
        "schema": "graph-report-ts-v2-general-run-config-v1",
        "mode": args.mode,
        "arguments": vars(args),
        "model_config": model.export_config(),
        "training_config": asdict(training),
        "training_strategy_version": args.training_protocol,
        "linear_anchor": anchor_settings,
        "linear_anchor_fit": anchor_fit,
        "project_identity": project_identity,
        "dataset_identity": dataset_identity,
        "preprocessing": preprocessing,
        "prompt_hidden_cache": prompt_hidden_cache,
        "best_epoch": best_epoch or None,
        "best_validation_mse": best_mse if best_epoch else None,
        "best_gate_statistics": best_gate_statistics,
        "model_only_initialization": initialization_audit,
    }
    _write_json(output_dir / "run_config.json", run_config)
    if initialization_audit is not None:
        save_checkpoint(
            output_dir / "best.pt",
            model,
            optimizer,
            0,
            best_mse,
            {
                "training_strategy_version": args.training_protocol,
                "linear_anchor": anchor_settings,
                "linear_anchor_fit": anchor_fit,
                "config": model.export_config(),
                "preprocessing": preprocessing,
                "project_identity": project_identity,
                "dataset_identity": dataset_identity,
                "prompt_hidden_cache": prompt_hidden_cache,
                "best_val_mse": best_mse,
                "best_epoch": 0,
                "source_best_epoch": initialization_audit["source_best_epoch"],
                "model_only_initialization": initialization_audit,
                "history": [],
                "stale": 0,
                "lr_reductions": 0,
                "scheduler": scheduler.state_dict(),
                "amp_scaler": None,
                "rng_state": _rng_state(),
            },
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, training.epochs + 1):
        apply_warmup(optimizer, epoch, training)
        align = alignment_weight(epoch, training)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            align,
            optimizer,
            training.gradient_clip,
            amp_dtype,
            scaler,
            args.pin_memory,
            args.max_train_batches,
            args.collect_branch_gradient_norms,
            args.gradient_diagnostic_interval,
        )
        with torch.no_grad():
            val_metrics = run_epoch(
                model,
                val_loader,
                device,
                align,
                amp_dtype=amp_dtype,
                non_blocking=args.pin_memory,
                max_batches=args.max_eval_batches,
            )
        improved = val_metrics["mse"] < best_mse
        if improved:
            best_mse = val_metrics["mse"]
            best_epoch = epoch
            best_gate_statistics = _gate_statistics(val_metrics)
            stale = 0
        else:
            stale += 1
        lr_reduced = False
        if epoch >= training.warmup_epochs:
            lr_reduced = step_plateau_scheduler(
                scheduler,
                optimizer,
                val_metrics["mse"],
            )
            if lr_reduced:
                lr_reductions += 1
        early_stop_eligible = should_stop_v2(stale, lr_reductions, training)
        record = {
            "epoch": epoch,
            "align_weight": align,
            "train": train_metrics,
            "val": val_metrics,
            "learning_rates": optimizer_learning_rates(optimizer),
            "lr_reduced": lr_reduced,
            "lr_reductions": lr_reductions,
            "stale": stale,
            "early_stop_eligible": early_stop_eligible,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        checkpoint_extra = {
            "training_strategy_version": args.training_protocol,
            "linear_anchor": anchor_settings,
            "linear_anchor_fit": anchor_fit,
            "config": model.export_config(),
            "preprocessing": preprocessing,
            "project_identity": project_identity,
            "dataset_identity": dataset_identity,
            "prompt_hidden_cache": prompt_hidden_cache,
            "best_val_mse": best_mse,
            "best_epoch": best_epoch,
            "best_gate_statistics": best_gate_statistics,
            "gate_statistics": {
                "train": _gate_statistics(train_metrics),
                "validation": _gate_statistics(val_metrics),
            },
            "stale": stale,
            "scheduler": scheduler.state_dict(),
            "lr_reductions": lr_reductions,
            "history": history,
            "amp_scaler": scaler.state_dict() if scaler is not None else None,
            "rng_state": _rng_state(),
            "model_only_initialization": initialization_audit,
        }
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, val_metrics["mse"], checkpoint_extra)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_mse, checkpoint_extra)
        if early_stop_eligible:
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "epoch": epoch,
                        "stale": stale,
                        "lr_reductions": lr_reductions,
                        "best_epoch": best_epoch,
                        "best_validation_mse": best_mse,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            break
    if args.skip_final_test:
        pilot_summary = {
            "schema": "graph-report-ts-v2-validation-only-pilot-v1",
            "best_val_mse": best_mse,
            "best_epoch": best_epoch,
            "epochs": len(history),
            "test_evaluated": False,
            "history": history,
        }
        _write_json(output_dir / "pilot_summary.json", pilot_summary)
        _write_json(output_dir / "history.json", {"history": history})
        run_config.update(
            {
                "best_epoch": best_epoch,
                "best_validation_mse": best_mse,
                "best_gate_statistics": best_gate_statistics,
                "validation_only_pilot": True,
            }
        )
        _write_json(output_dir / "run_config.json", run_config)
        print(json.dumps({"event": "validation_only_pilot_complete", **pilot_summary}, ensure_ascii=False), flush=True)
        return
    if not (output_dir / "best.pt").exists():
        if not args.resume:
            raise RuntimeError("training completed without a best checkpoint")
        save_checkpoint(
            output_dir / "best.pt",
            model,
            optimizer,
            start_epoch - 1,
            best_mse,
            {
                "training_strategy_version": args.training_protocol,
                "linear_anchor": anchor_settings,
                "linear_anchor_fit": anchor_fit,
                "config": model.export_config(),
                "preprocessing": preprocessing,
                "project_identity": project_identity,
                "dataset_identity": dataset_identity,
                "prompt_hidden_cache": prompt_hidden_cache,
                "best_val_mse": best_mse,
                "best_epoch": best_epoch or (start_epoch - 1),
                "best_gate_statistics": best_gate_statistics,
                "history": history,
                "scheduler": scheduler.state_dict(),
                "lr_reductions": lr_reductions,
            },
        )
    checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    checkpoint_anchor_fit = checkpoint.get("linear_anchor_fit")
    if isinstance(checkpoint_anchor_fit, dict):
        anchor_fit = dict(checkpoint_anchor_fit)
    best_epoch = int(checkpoint.get("best_epoch", checkpoint["epoch"]))
    best_mse = float(checkpoint.get("best_val_mse", checkpoint["val_mse"]))
    checkpoint_gate_statistics = checkpoint.get("best_gate_statistics")
    if isinstance(checkpoint_gate_statistics, dict):
        best_gate_statistics = {str(key): float(value) for key, value in checkpoint_gate_statistics.items()}
    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            test_loader,
            device,
            0.0,
            amp_dtype=amp_dtype,
            non_blocking=args.pin_memory,
            max_batches=args.max_eval_batches,
        )
    runtime = _runtime_metadata(device, args.amp)
    runtime["test_gate_statistics"] = _gate_statistics(test_metrics)
    runtime["project_identity"] = project_identity
    runtime["dataset_identity"] = dataset_identity
    runtime["linear_anchor"] = anchor_settings
    runtime["linear_anchor_fit"] = anchor_fit
    runtime["prompt_hidden_cache"] = prompt_hidden_cache
    runtime["window_protocol"] = {
        "prompt_len": args.prompt_len,
        "input_len": args.input_len,
        "target_len": args.horizon,
        "disjoint_prompt_numeric": True,
    }
    summary = {
        "best_val_mse": best_mse,
        "best_epoch": best_epoch,
        "test": test_metrics,
        "epochs": len(history),
        "amp": args.amp,
        "config": model.export_config(),
        "project_identity": project_identity,
        "dataset_identity": dataset_identity,
        "linear_anchor": anchor_settings,
        "linear_anchor_fit": anchor_fit,
        "prompt_hidden_cache": prompt_hidden_cache,
        "best_gate_statistics": best_gate_statistics,
        "runtime": runtime,
    }
    _write_json(output_dir / "metrics.json", summary)
    _write_json(output_dir / "history.json", {"history": history})
    run_config.update(
        {
            "best_epoch": best_epoch,
            "best_validation_mse": best_mse,
            "best_gate_statistics": best_gate_statistics,
            "test_gate_statistics": _gate_statistics(test_metrics),
            "runtime": runtime,
        }
    )
    _write_json(output_dir / "run_config.json", run_config)
    source_commit = project_identity.get("git_commit") or f"tree:{project_identity['tree_sha256']}"
    result = make_result_record(
        model_name=args.model_name,
        domain="general",
        dataset=args.dataset,
        horizon=args.horizon,
        seed=args.seed,
        best_epoch=best_epoch,
        validation_mse=best_mse,
        test_metrics=test_metrics,
        parameter_count=sum(parameter.numel() for parameter in model.parameters()),
        prompt_policy=f"compressed_previous_{args.prompt_len}",
        source_commit=str(source_commit),
        adapter_schema_hash=stable_digest(
            {"scaler": preprocessing["scaler"], "prompt_schema": preprocessing["prompt_schema"]}
        ),
        optimizer_profile=asdict(training),
        runtime=runtime,
    )
    write_result(result, output_dir)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
