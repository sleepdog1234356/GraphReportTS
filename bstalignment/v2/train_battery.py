from __future__ import annotations

import argparse
import json
import random
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .battery_cache import BatteryFeatureCache
from .battery_data import BatteryForecastV2Dataset, collate_battery_v2
from .contracts import GraphReportTSv2Config, require_local_text_model
from .losses import battery_v2_loss
from .model import BatteryGraphReportTSv2
from .results import make_result_record, masked_regression_metrics, stable_digest, write_result
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


BATTERY_V2_TRAINING_PROTOCOL = "battery-v2-plateau-guarded-v1"
BATTERY_V2_CORRECTED_MODEL_NAME = "GraphReportTS-v2-CorrectedTrain"


def _checkpoint_training_state(
    *,
    run_identity: dict[str, Any],
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    best_mse: float,
    best_epoch: int,
    stale: int,
    lr_reductions: int,
    training_protocol: str = BATTERY_V2_TRAINING_PROTOCOL,
) -> dict[str, Any]:
    return {
        "training_protocol": training_protocol,
        "run_identity": run_identity,
        "scheduler": scheduler.state_dict(),
        "best_mse": float(best_mse),
        "best_epoch": int(best_epoch),
        "stale": int(stale),
        "lr_reductions": int(lr_reductions),
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _project_source_identity(provenance_manifest: str | None) -> tuple[str, dict[str, Any] | None]:
    if provenance_manifest is None:
        return _git_commit(), None
    path = Path(provenance_manifest)
    if not path.exists():
        raise FileNotFoundError(f"provenance manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    project = payload.get("project")
    if not isinstance(project, dict):
        raise ValueError("provenance manifest does not contain a project identity")
    commit = project.get("git_commit")
    tree_hash = project.get("tree_sha256")
    if commit:
        return str(commit), project
    if not isinstance(tree_hash, str) or len(tree_hash) != 64:
        raise ValueError("project provenance requires a git commit or tree SHA-256")
    return f"tree:{tree_hash}", project


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {
        "torch": torch.__version__,
        "device": str(device),
        "cuda": torch.version.cuda,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        result.update(
            {
                "gpu": props.name,
                "gpu_memory_bytes": int(props.total_memory),
                "compute_capability": f"{props.major}.{props.minor}",
                "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
        )
    return result


def _merge_metric_sums(total: dict[str, float], values: dict[str, torch.Tensor], batch_size: int) -> None:
    for key, value in values.items():
        total[key] = total.get(key, 0.0) + float(value.detach().cpu()) * batch_size


def run_epoch(
    model: BatteryGraphReportTSv2,
    loader: DataLoader,
    device: torch.device,
    *,
    align_weight_value: float,
    optimizer: torch.optim.Optimizer | None = None,
    gradient_clip: float = 1.0,
    amp: str = "none",
    grad_scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    loss_sums: dict[str, float] = {}
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    count = 0
    amp_enabled = device.type == "cuda" and amp in {"bf16", "fp16"}
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    with torch.set_grad_enabled(training):
        for raw_batch in loader:
            raw_batch = raw_batch.to(device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
                output = model(raw_batch)
                losses = battery_v2_loss(output, raw_batch.target, raw_batch.target_mask, align_weight_value)
            if training:
                optimizer.zero_grad(set_to_none=True)
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.scale(losses["total"]).backward()
                    grad_scaler.unscale_(optimizer)
                else:
                    losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad], gradient_clip
                )
                if grad_scaler is not None and grad_scaler.is_enabled():
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                else:
                    optimizer.step()
            batch_size = int(raw_batch.target.shape[0])
            _merge_metric_sums(loss_sums, losses, batch_size)
            count += batch_size
            predictions.append(output["pred"].detach().cpu())
            targets.append(raw_batch.target.detach().cpu())
            masks.append(raw_batch.target_mask.detach().cpu())
    if count == 0:
        raise RuntimeError("battery v2 data loader produced no batches")
    metrics = masked_regression_metrics(torch.cat(predictions), torch.cat(targets), torch.cat(masks))
    metrics.update({f"loss_{key}": value / count for key, value in loss_sums.items()})
    return metrics


def build_datasets(args: argparse.Namespace) -> tuple[
    BatteryFeatureCache,
    BatteryForecastV2Dataset,
    BatteryForecastV2Dataset,
    BatteryForecastV2Dataset,
]:
    cache = BatteryFeatureCache.open(args.cache_dir)
    max_samples = args.max_samples if args.smoke else None
    train = BatteryForecastV2Dataset(
        cache,
        split="train",
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        training=True,
        soh_context_dropout=args.soh_context_dropout,
        max_samples=max_samples,
    )
    validation = BatteryForecastV2Dataset(
        cache,
        split="val",
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        scaler=train.scaler,
        prompt_thresholds=train.prompt_thresholds,
        training=False,
        max_samples=max_samples,
    )
    test = BatteryForecastV2Dataset(
        cache,
        split="test",
        seed=args.seed,
        prompt_mode=args.prompt_mode,
        scaler=train.scaler,
        prompt_thresholds=train.prompt_thresholds,
        training=False,
        max_samples=max_samples,
    )
    if not train.split_definition == validation.split_definition == test.split_definition:
        raise RuntimeError("battery v2 train, validation, and test datasets disagree on cell split")
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("battery v2 requires non-empty train, validation, and test splits")
    return cache, train, validation, test


def _loader(dataset: BatteryForecastV2Dataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
        "persistent_workers": args.num_workers > 0,
        "collate_fn": collate_battery_v2,
    }
    if args.num_workers > 0:
        options["prefetch_factor"] = args.prefetch_factor
    return DataLoader(**options)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GraphReportTS-v2 on cached battery sensor features")
    parser.add_argument("--dataset", choices=("mit", "xjtu", "synthetic"), default="mit")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", default="artifacts/battery/graphreportts_v2/runs")
    parser.add_argument("--output_model_subdir", default=None)
    parser.add_argument("--text_model", default="hf_models/distilbert-base-uncased")
    parser.add_argument("--provenance_manifest", default=None)
    parser.add_argument("--prompt_mode", choices=("sensor_only", "soh_assisted"), default="sensor_only")
    parser.add_argument("--soh_context_dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--core_lr", type=float, default=1e-3)
    parser.add_argument("--semantic_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--amp", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max_samples", type=int, default=16)
    parser.add_argument("--disable_text", action="store_true")
    parser.add_argument("--precompute_text", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--text_cache_root", default=None)
    parser.add_argument("--text_precompute_batch_size", type=int, default=256)
    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
    *,
    model_factory: Callable[[GraphReportTSv2Config], BatteryGraphReportTSv2] = BatteryGraphReportTSv2,
    model_config_overrides: Mapping[str, Any] | None = None,
    model_name: str = BATTERY_V2_CORRECTED_MODEL_NAME,
    training_protocol: str = BATTERY_V2_TRAINING_PROTOCOL,
    model_prepare_hook: Callable[..., dict[str, Any]] | None = None,
) -> None:
    args = parse_args(argv)
    if args.dataset == "xjtu" and not Path(args.cache_dir).exists():
        raise ValueError("XJTU is deferred; only an existing compatible v2 cache may be used")
    if not args.disable_text:
        require_local_text_model(args.text_model)
    _seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
            args.device = str(device)
        torch.cuda.set_device(device.index)
        torch.cuda.reset_peak_memory_stats(device)
    cache, train_dataset, validation_dataset, test_dataset = build_datasets(args)
    source_identity, project_provenance = _project_source_identity(args.provenance_manifest)
    train_loader = _loader(train_dataset, args, shuffle=True)
    validation_loader = _loader(validation_dataset, args, shuffle=False)
    test_loader = _loader(test_dataset, args, shuffle=False)
    model_config_values: dict[str, Any] = {
        "domain": "battery",
        "input_len": 32,
        "pred_len": 20,
        "text_model": args.text_model,
        "use_text": not args.disable_text,
    }
    model_config_values.update(dict(model_config_overrides or {}))
    model_config = GraphReportTSv2Config(**model_config_values)
    model = model_factory(model_config).to(device)
    preparation = (
        model_prepare_hook(
            model=model,
            datasets=(train_dataset, validation_dataset, test_dataset),
            args=args,
            feature_cache_hash=cache.manifest_hash,
        )
        if model_prepare_hook is not None
        else {"enabled": False}
    )
    training_config = V2TrainingConfig(
        epochs=1 if args.smoke else args.epochs,
        core_lr=args.core_lr,
        semantic_lr=args.semantic_lr,
        weight_decay=args.weight_decay,
        gradient_clip=args.gradient_clip,
        patience=args.patience,
    )
    optimizer = build_optimizer(model, training_config)
    scheduler = build_plateau_scheduler(optimizer, training_config)
    grad_scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp == "fp16")
    output = Path(args.output) / args.dataset
    if args.output_model_subdir:
        output = output / args.output_model_subdir
    output = output / f"seed-{args.seed}"
    output.mkdir(parents=True, exist_ok=True)
    split_definition = {
        "schema": (
            "xjtu_protocol_stratified_v1"
            if str(cache.manifest.get("provenance", {}).get("dataset", "")).lower() == "xjtu"
            else "cell_random_v1"
        ),
        "train": list(train_dataset.split_definition.train),
        "validation": list(train_dataset.split_definition.val),
        "test": list(train_dataset.split_definition.test),
    }
    run_identity = {
        "training_protocol": training_protocol,
        "result_model_name": model_name,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "cache_hash": cache.manifest_hash,
        "scaler_hash": train_dataset.scaler.schema_hash,
        "prompt_thresholds": {
            name: list(values) for name, values in train_dataset.prompt_thresholds.items()
        },
        "prompt_mode": args.prompt_mode,
        "soh_context_dropout": float(args.soh_context_dropout),
        "split_definition": split_definition,
        "model": asdict(model_config),
        "optimizer": {
            "core_lr": float(args.core_lr),
            "semantic_lr": float(args.semantic_lr),
            "weight_decay": float(args.weight_decay),
            "gradient_clip": float(args.gradient_clip),
            "amp": args.amp,
        },
        "project_source": source_identity,
    }
    (output / "run_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "training_protocol": training_protocol,
                "result_model_name": model_name,
                "model": asdict(model_config),
                "training": asdict(training_config),
                "cache_hash": cache.manifest_hash,
                "scaler": train_dataset.scaler.state_dict(),
                "prompt_thresholds": {
                    name: list(values) for name, values in train_dataset.prompt_thresholds.items()
                },
                "split_definition": split_definition,
                "samples": {
                    "train": len(train_dataset),
                    "validation": len(validation_dataset),
                    "test": len(test_dataset),
                },
                "runtime": _runtime_metadata(device),
                "project_provenance": project_provenance,
                "run_identity": run_identity,
                "preparation": preparation,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    best_mse = float("inf")
    best_epoch = 0
    stale = 0
    lr_reductions = 0
    start_epoch = 1
    last_path = output / "last.pt"
    if args.resume and last_path.exists() and not args.smoke:
        checkpoint = torch.load(last_path, map_location=device, weights_only=False)
        if stable_digest(checkpoint.get("run_identity", {})) != stable_digest(run_identity):
            raise RuntimeError("battery-v2 resume checkpoint identity differs from the requested run")
        if checkpoint.get("scheduler") is None:
            raise RuntimeError("battery-v2 resume checkpoint is missing scheduler state")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        best_mse = float(checkpoint["best_mse"])
        best_epoch = int(checkpoint["best_epoch"])
        stale = int(checkpoint["stale"])
        lr_reductions = int(checkpoint.get("lr_reductions", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
    history_path = output / "history.jsonl"
    for epoch in range(start_epoch, training_config.epochs + 1):
        train_dataset.set_epoch(epoch)
        apply_warmup(optimizer, epoch, training_config)
        current_alignment = alignment_weight(epoch, training_config)
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            align_weight_value=current_alignment,
            optimizer=optimizer,
            gradient_clip=training_config.gradient_clip,
            amp=args.amp,
            grad_scaler=grad_scaler,
        )
        validation_metrics = run_epoch(
            model,
            validation_loader,
            device,
            align_weight_value=current_alignment,
            amp=args.amp,
        )
        improved = validation_metrics["mse"] < best_mse
        if improved:
            best_mse = validation_metrics["mse"]
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        lr_reduced = step_plateau_scheduler(scheduler, optimizer, validation_metrics["mse"])
        if lr_reduced:
            lr_reductions += 1
        checkpoint_state = _checkpoint_training_state(
            run_identity=run_identity,
            scheduler=scheduler,
            best_mse=best_mse,
            best_epoch=best_epoch,
            stale=stale,
            lr_reductions=lr_reductions,
            training_protocol=training_protocol,
        )
        if improved:
            save_checkpoint(
                output / "best.pt",
                model,
                optimizer,
                epoch,
                best_mse,
                {
                    "cache_hash": cache.manifest_hash,
                    "scaler": train_dataset.scaler.state_dict(),
                    "prompt_thresholds": {
                        name: list(values) for name, values in train_dataset.prompt_thresholds.items()
                    },
                    **checkpoint_state,
                },
            )
        save_checkpoint(
            last_path,
            model,
            optimizer,
            epoch,
            validation_metrics["mse"],
            checkpoint_state,
        )
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "align_weight": current_alignment,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "best_epoch": best_epoch,
                        "best_validation_mse": best_mse,
                        "stale": stale,
                        "learning_rates": optimizer_learning_rates(optimizer),
                        "lr_reduced": lr_reduced,
                        "lr_reductions": lr_reductions,
                        "training_protocol": training_protocol,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        print(
            f"model={model_name} epoch={epoch} "
            f"train_loss={train_metrics['loss_total']:.8g} "
            f"val_mse={validation_metrics['mse']:.8g} best={best_mse:.8g} stale={stale} "
            f"lr={optimizer_learning_rates(optimizer)} reductions={lr_reductions}",
            flush=True,
        )
        if should_stop_v2(stale, lr_reductions, training_config):
            print(
                f"model={model_name} early_stop_epoch={epoch} "
                f"best_epoch={best_epoch} lr_reductions={lr_reductions}",
                flush=True,
            )
            break
    best = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    test_metrics = run_epoch(model, test_loader, device, align_weight_value=0.0, amp=args.amp)
    record = make_result_record(
        model_name=model_name,
        domain="battery",
        dataset=args.dataset,
        seed=args.seed,
        best_epoch=best_epoch,
        validation_mse=best_mse,
        test_metrics=test_metrics,
        parameter_count=_parameter_count(model),
        prompt_policy=args.prompt_mode,
        source_commit=source_identity,
        horizon=20,
        adapter_schema_hash=train_dataset.scaler.schema_hash,
        cache_hash=cache.manifest_hash,
        optimizer_profile={
            **asdict(training_config),
            "training_protocol": training_protocol,
        },
        runtime={
            **_runtime_metadata(device),
            "training_protocol": training_protocol,
            "result_model_name": model_name,
            "lr_reductions": lr_reductions,
            "final_learning_rates": optimizer_learning_rates(optimizer),
            "project_provenance": project_provenance,
            "prompt_thresholds": {
                name: list(values) for name, values in train_dataset.prompt_thresholds.items()
            },
        },
    )
    write_result(record, output)
    print(json.dumps({"best_validation_mse": best_mse, "test": test_metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
