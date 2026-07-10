from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .battery_protocol import require_formal_battery_protocol
    from .data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
    from .data_general import GeneralForecastGraphDataset, StandardScalerNP, collate_general_graph_batch
    from .experiment_config import ExperimentConfig, ensure_research_dirs
    from .graph_report_losses import graph_report_loss, regression_metrics
    from .graph_report_model import GraphReportTS, GraphReportTSConfig
    from .training_strategy import (
        MAIN_TRAINING_PROFILE,
        MainTrainingProfile,
        TRAINING_STRATEGY_VERSION,
        GraphReportScheduler,
        build_graph_report_optimizer,
        graph_report_align_weight,
        graph_report_group_lrs,
        require_checkpoint_strategy_version,
        require_nonempty_splits,
        should_stop_graph_report,
        update_graph_report_stale,
    )
    from .utils import AverageMeter, ensure_dir, save_json, seed_everything, to_device
except ImportError:
    from battery_protocol import require_formal_battery_protocol
    from data_battery_raw import BatteryRawGraphDataset, collate_graph_report_batch
    from data_general import GeneralForecastGraphDataset, StandardScalerNP, collate_general_graph_batch
    from experiment_config import ExperimentConfig, ensure_research_dirs
    from graph_report_losses import graph_report_loss, regression_metrics
    from graph_report_model import GraphReportTS, GraphReportTSConfig
    from training_strategy import (
        MAIN_TRAINING_PROFILE,
        MainTrainingProfile,
        TRAINING_STRATEGY_VERSION,
        GraphReportScheduler,
        build_graph_report_optimizer,
        graph_report_align_weight,
        graph_report_group_lrs,
        require_checkpoint_strategy_version,
        require_nonempty_splits,
        should_stop_graph_report,
        update_graph_report_stale,
    )
    from utils import AverageMeter, ensure_dir, save_json, seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser(description="Train Battery/General GraphReportTS")
    p.add_argument("--variant", choices=["battery", "general"], default="battery")
    p.add_argument("--dataset", type=str, default="mit")
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_dir", type=str, default="runs/graph_report_ts")
    p.add_argument("--run_dir", type=str, default=None)
    p.add_argument("--input_len", type=int, default=96)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--history_len", type=int, default=32)
    p.add_argument("--resample_len", type=int, default=128)
    p.add_argument("--delay_dim", type=int, default=8)
    p.add_argument("--delay_lag", type=int, default=1)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--graph_layers", type=int, default=2)
    p.add_argument("--patch_size", type=int, default=8)
    p.add_argument("--patch_stride", type=int, default=4)
    p.add_argument("--topk_edges", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--text_model", type=str, default="distilbert-base-uncased")
    p.add_argument("--no_hf_text", action="store_true")
    p.add_argument("--unfreeze_text", action="store_true")
    p.add_argument("--no_report_prompt", action="store_true")
    p.add_argument("--no_cross_modal", action="store_true")
    p.add_argument("--no_ic_dv", action="store_true")
    p.add_argument("--no_hankel_map", action="store_true")
    p.add_argument("--no_derivative_map", action="store_true")
    p.add_argument("--no_dynamic_graph", action="store_true")
    p.add_argument("--no_domain_edges", action="store_true")
    p.add_argument("--separate_heads", action="store_true")
    p.add_argument("--no_numeric_history", action="store_true")
    p.add_argument("--no_multi_cycle_raw", action="store_true")
    p.add_argument("--single_cycle_raw", action="store_true")
    p.add_argument("--no_text_gate", action="store_true")
    p.add_argument("--no_semantic_alignment", action="store_true")
    p.add_argument("--no_align_loss", action="store_true")
    p.add_argument("--absolute_step_decoder", action="store_true")
    p.add_argument("--temporal_layers", type=int, default=1)
    p.add_argument("--temporal_heads", type=int, default=4)
    p.add_argument("--allow_summary_fallback", action="store_true", help="Smoke-test only: synthesize raw curves from summary if MIT raw arrays are missing")
    p.add_argument("--cache_items", action="store_true", help="Enable per-worker sample caching for raw map construction. This can grow RAM quickly with many workers.")
    p.add_argument("--no_cache_items", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--precomputed_cache_dir", type=str, default=None, help="Read deterministic battery graph samples from this cache root when available")
    p.add_argument("--require_precomputed_cache", action="store_true", help="Fail instead of falling back to online battery graph map construction")
    p.add_argument("--battery_input_mode", choices=["hankel_graph", "raw_sequence"], default="hankel_graph")
    p.add_argument("--precomputed_sequence_cache_dir", type=str, default=None, help="Read deterministic battery raw-sequence samples from this cache root when available")
    p.add_argument("--require_precomputed_sequence_cache", action="store_true", help="Fail instead of falling back to online battery sequence construction")
    p.add_argument("--protocol_stage", choices=["main", "ablation"], default="main")
    p.add_argument("--ablation_suite_version", type=str, default=None)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--loss", choices=["smooth_l1", "mse", "mae"], default="smooth_l1")
    p.add_argument("--w_align", type=float, default=0.001)
    p.add_argument("--align_warmup_epochs", type=int, default=0)
    p.add_argument("--early_stop_patience", type=int, default=10)
    p.add_argument("--early_stop_min_delta", type=float, default=1e-5)
    p.add_argument("--no_resume", action="store_true", help="Disable automatic resume from last.pt/best.pt in the output directory")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    if args.batch_size is None:
        args.batch_size = 64 if args.variant == "battery" else 32
    return args


def build_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    if args.variant == "battery":
        ds_kwargs = dict(
            dataset_name=args.dataset,
            data_root=args.data_root,
            max_horizon=args.pred_len,
            history_len=args.history_len,
            resample_len=args.resample_len,
            delay_dim=args.delay_dim,
            delay_lag=args.delay_lag,
            include_derivatives=not args.no_derivative_map,
            include_hankel=not args.no_hankel_map,
            include_ic_dv=not args.no_ic_dv,
            allow_summary_fallback=args.allow_summary_fallback,
            cache_items=args.cache_items and not args.no_cache_items,
            seed=args.seed,
            max_cycles=args.max_cycles,
        )
        if args.battery_input_mode == "raw_sequence":
            ds_kwargs.update(
                input_representation="sequence",
                precomputed_sequence_cache_dir=args.precomputed_sequence_cache_dir,
                require_precomputed_sequence_cache=args.require_precomputed_sequence_cache,
            )
        else:
            ds_kwargs.update(
                precomputed_cache_dir=args.precomputed_cache_dir,
                require_precomputed_cache=args.require_precomputed_cache,
            )
        train_ds = BatteryRawGraphDataset(split="train", **ds_kwargs)
        val_ds = BatteryRawGraphDataset(split="val", **ds_kwargs)
        test_ds = BatteryRawGraphDataset(split="test", **ds_kwargs)
        collate = collate_graph_report_batch
        output_dim = 1
    else:
        scaler = StandardScalerNP()
        ds_kwargs = dict(
            dataset_name=args.dataset,
            data_root=args.data_root,
            input_len=args.input_len,
            pred_len=args.pred_len,
            resample_len=args.resample_len,
            delay_dim=args.delay_dim,
            delay_lag=args.delay_lag,
            include_derivatives=not args.no_derivative_map,
            include_hankel=not args.no_hankel_map,
        )
        train_ds = GeneralForecastGraphDataset(split="train", scaler=scaler, fit_scaler=True, **ds_kwargs)
        val_ds = GeneralForecastGraphDataset(split="val", scaler=scaler, fit_scaler=False, **ds_kwargs)
        test_ds = GeneralForecastGraphDataset(split="test", scaler=scaler, fit_scaler=False, **ds_kwargs)
        collate = collate_general_graph_batch
        output_dim = len(train_ds.columns)
    require_nonempty_splits(train_ds, val_ds, test_ds, "GraphReportTS trainer")
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collate,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs.update({"prefetch_factor": 2})

    def make_loader(ds, name: str) -> DataLoader:
        kwargs = dict(loader_kwargs)
        if args.num_workers > 0:
            kwargs["persistent_workers"] = name == "train"
        return DataLoader(ds, shuffle=(name == "train"), **kwargs)

    loaders = [make_loader(ds, name) for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]]
    return loaders[0], loaders[1], loaders[2], output_dim


def _empty_gate_stats() -> Dict[str, float]:
    return {"sum": 0.0, "sum_sq": 0.0, "count": 0.0, "min": float("inf"), "max": float("-inf")}


def _update_gate_stats(stats: Dict[str, float], gate: Optional[torch.Tensor]) -> None:
    if gate is None:
        return
    vals = gate.detach().float().reshape(-1).cpu()
    if vals.numel() == 0:
        return
    stats["sum"] += float(vals.sum())
    stats["sum_sq"] += float((vals * vals).sum())
    stats["count"] += float(vals.numel())
    stats["min"] = min(stats["min"], float(vals.min()))
    stats["max"] = max(stats["max"], float(vals.max()))


def _finalize_gate_stats(stats: Dict[str, float]) -> Dict[str, float]:
    count = max(stats["count"], 1.0)
    mean = stats["sum"] / count
    var = max(stats["sum_sq"] / count - mean * mean, 0.0)
    if stats["count"] <= 0:
        return {"gate_mean": 0.0, "gate_std": 0.0, "gate_min": 0.0, "gate_max": 0.0}
    return {"gate_mean": mean, "gate_std": var**0.5, "gate_min": stats["min"], "gate_max": stats["max"]}


def _model_forward(model: GraphReportTS, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    steps = None
    if model.cfg.variant == "battery" and not model.cfg.use_relative_steps:
        steps = batch.get("target_steps")
    return model(
        batch.get("maps"),
        batch["prompt"],
        batch["horizon"],
        steps=steps,
        history_features=batch.get("history_features"),
        raw_sequences=batch.get("raw_sequences"),
    )


def _batch_size(batch: Dict[str, Any]) -> int:
    source = batch.get("maps")
    if source is None:
        source = batch["raw_sequences"]
    return int(source.size(0))


def _resolve_out_dir(args) -> Path:
    if args.run_dir is None:
        return ensure_dir(Path(args.out_dir) / args.variant / args.dataset)
    return ensure_dir(Path(args.run_dir))


def _resume_checkpoint_path(out_dir: Path) -> Optional[Path]:
    last_path = out_dir / "last.pt"
    if last_path.exists():
        return last_path
    best_path = out_dir / "best.pt"
    return best_path if best_path.exists() else None


def _epoch_duration(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(f"{context} must be a finite non-negative number")
    duration = float(value)
    if not math.isfinite(duration) or duration < 0.0:
        raise RuntimeError(f"{context} must be a finite non-negative number")
    return duration


def _reconcile_epoch_history(
    history_path: Path,
    checkpoint_epoch: int,
    checkpoint_epoch_seconds: Optional[Any],
) -> List[float]:
    if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int) or checkpoint_epoch < 0:
        raise RuntimeError("checkpoint epoch must be a non-negative integer")
    checkpoint_durations: Optional[List[float]] = None
    if checkpoint_epoch_seconds is not None:
        if not isinstance(checkpoint_epoch_seconds, (list, tuple)):
            raise RuntimeError("checkpoint epoch_seconds must be a sequence")
        checkpoint_durations = [
            _epoch_duration(value, f"checkpoint epoch_seconds[{index}]")
            for index, value in enumerate(checkpoint_epoch_seconds)
        ]
        if len(checkpoint_durations) > checkpoint_epoch:
            raise RuntimeError("checkpoint epoch_seconds has more entries than the checkpoint epoch")
    if not history_path.exists():
        return checkpoint_durations or []

    original_lines = history_path.read_text(encoding="utf-8").splitlines()
    retained_lines: List[str] = []
    history_durations: List[float] = []
    timing_started = False
    expected_epoch: Optional[int] = None
    for line_number, line in enumerate(original_lines, start=1):
        if expected_epoch is not None and expected_epoch > checkpoint_epoch:
            break
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"epoch history {history_path} line {line_number} is invalid before checkpoint epoch {checkpoint_epoch}"
            ) from exc
        if not isinstance(row, dict):
            raise RuntimeError(f"epoch history {history_path} line {line_number} must be a JSON object")
        row_epoch = row.get("epoch")
        if isinstance(row_epoch, bool) or not isinstance(row_epoch, int):
            raise RuntimeError(f"epoch history {history_path} line {line_number} has an invalid epoch")
        if row_epoch > checkpoint_epoch:
            break
        if expected_epoch is None:
            expected_epoch = row_epoch
        if row_epoch != expected_epoch:
            raise RuntimeError(
                f"epoch history {history_path} expected epoch {expected_epoch}, found {row_epoch}"
            )
        retained_lines.append(line)
        if "epoch_seconds" in row:
            timing_started = True
            history_durations.append(
                _epoch_duration(row["epoch_seconds"], f"epoch history epoch {row_epoch} epoch_seconds")
            )
        elif timing_started:
            raise RuntimeError(
                f"epoch history {history_path} has a timing gap at epoch {row_epoch}"
            )
        expected_epoch += 1

    if retained_lines and expected_epoch is not None and expected_epoch - 1 != checkpoint_epoch:
        raise RuntimeError(
            f"epoch history {history_path} ends at epoch {expected_epoch - 1}, "
            f"before checkpoint epoch {checkpoint_epoch}"
        )
    if checkpoint_durations is not None and retained_lines:
        if not history_durations and checkpoint_durations:
            raise RuntimeError("checkpoint epoch_seconds does not match retained epoch history timing")
        if history_durations and (
            len(history_durations) > len(checkpoint_durations)
            or checkpoint_durations[-len(history_durations):] != history_durations
        ):
            raise RuntimeError("checkpoint epoch_seconds does not match retained epoch history timing")

    if len(retained_lines) != len(original_lines):
        temporary_path = history_path.with_name(f".{history_path.name}.resume.tmp")
        try:
            with temporary_path.open("w", encoding="utf-8", newline="") as handle:
                for line in retained_lines:
                    handle.write(line + "\n")
            temporary_path.replace(history_path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
    return checkpoint_durations if checkpoint_durations is not None else history_durations


def _run_summary_payload(
    *,
    best_epoch: int,
    stopped_epoch: int,
    epoch_seconds: List[float],
    trainable_parameter_count: int,
    training_strategy_version: str,
    ablation_suite_version: Optional[str],
) -> Dict[str, Any]:
    total_train_seconds = float(sum(epoch_seconds))
    return {
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "mean_epoch_seconds": float(total_train_seconds / len(epoch_seconds)) if epoch_seconds else 0.0,
        "total_train_seconds": total_train_seconds,
        "trainable_parameter_count": int(trainable_parameter_count),
        "training_strategy_version": training_strategy_version,
        "ablation_suite_version": ablation_suite_version,
    }


def _loss_weights(args, profile, epoch: int) -> Dict[str, float]:
    if args.no_report_prompt or args.no_cross_modal or args.no_align_loss or args.no_semantic_alignment:
        align = 0.0
    else:
        align = graph_report_align_weight(epoch, profile)
    return {"align": align}


def _write_gate_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["cell_id", "cycle", "gate"])
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    weights: Dict[str, float],
    loss_type: str,
    gate_path: Optional[Path] = None,
) -> Dict[str, float]:
    model.eval()
    meters = {k: AverageMeter() for k in ["total", "reg", "align"]}
    gate_stats = _empty_gate_stats()
    gate_rows: List[Dict[str, Any]] = []
    mse_sum = mae_sum = count = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        out = _model_forward(model, batch)
        loss = graph_report_loss(out, batch["y"], batch["mask"], weights, loss_type=loss_type)
        n = _batch_size(batch)
        for k, v in loss.items():
            meters[k].update(float(v.detach()), n)
        _update_gate_stats(gate_stats, out.get("gate"))
        if gate_path is not None and "gate" in out:
            gate_vals = out["gate"].detach().reshape(n, -1).mean(dim=1).cpu().tolist()
            cycles = batch["cycle"].detach().cpu().tolist() if torch.is_tensor(batch.get("cycle")) else [0] * n
            for i, gate in enumerate(gate_vals):
                gate_rows.append({"cell_id": batch["cell_id"][i], "cycle": int(cycles[i]), "gate": float(gate)})
        metrics = regression_metrics(out["pred"], batch["y"], batch["mask"])
        elems = float(batch["mask"].sum().detach().cpu())
        mse_sum += metrics["mse"] * elems
        mae_sum += metrics["mae"] * elems
        count += elems
    mse = mse_sum / max(count, 1.0)
    mae = mae_sum / max(count, 1.0)
    out = {k: m.avg for k, m in meters.items()}
    out.update({"mse": mse, "mae": mae, "rmse": mse**0.5})
    out.update(_finalize_gate_stats(gate_stats))
    if gate_path is not None:
        _write_gate_rows(gate_path, gate_rows)
    return out


def main():
    args = parse_args()
    if args.variant == "battery":
        require_formal_battery_protocol(
            observed_cycles=args.history_len,
            prediction_cycles=args.pred_len,
            batch_size=args.batch_size,
            stage=args.protocol_stage,
            context="GraphReportTS battery trainer",
        )
    seed_everything(args.seed)
    cfg = ExperimentConfig()
    ensure_research_dirs(cfg)
    out_dir = _resolve_out_dir(args)
    device = torch.device(args.device)

    train_loader, val_loader, test_loader, output_dim = build_loaders(args)
    model_cfg = GraphReportTSConfig(
        variant=args.variant,
        output_dim=output_dim,
        d_model=args.d_model,
        patch_size=args.patch_size,
        patch_stride=args.patch_stride,
        graph_layers=args.graph_layers,
        topk_edges=args.topk_edges,
        dropout=args.dropout,
        text_model=args.text_model,
        use_hf_text_encoder=not args.no_hf_text,
        freeze_text=not args.unfreeze_text,
        use_report_prompt=not args.no_report_prompt,
        use_cross_modal_fusion=not args.no_cross_modal,
        use_domain_edges=not args.no_domain_edges,
        use_dynamic_graph=not args.no_dynamic_graph,
        unified_decoder=not args.separate_heads,
        battery_history_len=args.history_len,
        history_feature_dim=8,
        use_multi_cycle_raw=not args.no_multi_cycle_raw,
        single_cycle_raw=args.single_cycle_raw,
        use_numeric_history=not args.no_numeric_history,
        use_text_gate=not args.no_text_gate,
        use_semantic_alignment=not args.no_semantic_alignment,
        use_relative_steps=not args.absolute_step_decoder,
        temporal_layers=args.temporal_layers,
        temporal_heads=args.temporal_heads,
        battery_input_mode=args.battery_input_mode,
        raw_sequence_len=args.resample_len,
    )
    model = GraphReportTS(model_cfg).to(device)
    with torch.no_grad():
        init_batch = next(iter(train_loader))
        init_batch = to_device(init_batch, device)
        _ = _model_forward(model, init_batch)
    text_backbone = getattr(getattr(model, "text_encoder", None), "backbone", None)
    if model_cfg.freeze_text and text_backbone is not None:
        assert all(not parameter.requires_grad for parameter in text_backbone.parameters())
        text_backbone.eval()

    profile = MAIN_TRAINING_PROFILE
    training_strategy_version = TRAINING_STRATEGY_VERSION
    trainable_parameter_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    resume_path = None
    resume_checkpoint = None
    if not args.no_resume:
        resume_path = _resume_checkpoint_path(out_dir)
        if resume_path is not None:
            resume_checkpoint = torch.load(resume_path, map_location=device)
            require_checkpoint_strategy_version(resume_checkpoint, "GraphReportTS trainer")
            if "training_profile" in resume_checkpoint:
                profile = MainTrainingProfile(**resume_checkpoint["training_profile"])

    opt = build_graph_report_optimizer(model, profile)
    scheduler = GraphReportScheduler(opt, profile)
    eval_weights = _loss_weights(args, profile, profile.max_epochs)

    save_json(
        {
            "args": vars(args),
            "model_cfg": model_cfg.__dict__,
            "output_dim": output_dim,
            "protocol_stage": args.protocol_stage,
            "training_strategy_version": training_strategy_version,
            "training_profile": profile.__dict__,
            "ablation_suite_version": args.ablation_suite_version,
            "trainable_parameter_count": trainable_parameter_count,
        },
        out_dir / "run_config.json",
    )
    best = float("inf")
    stale = 0
    start_epoch = 1
    best_epoch = 0
    stopped_epoch = 0
    epoch_seconds: List[float] = []
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        if "optimizer" in resume_checkpoint:
            opt.load_state_dict(resume_checkpoint["optimizer"])
        if "scheduler" in resume_checkpoint:
            scheduler.load_state_dict(resume_checkpoint["scheduler"])
        if "group_lrs" in resume_checkpoint:
            group_lrs = resume_checkpoint["group_lrs"]
            for group in opt.param_groups:
                group["lr"] = float(group_lrs[group["role"]])
        val_metrics = resume_checkpoint.get("val_metrics", {})
        best = float(resume_checkpoint.get("best", val_metrics.get("mse", best)))
        stale = int(resume_checkpoint.get("stale", 0))
        checkpoint_epoch = resume_checkpoint.get("epoch", 0)
        if isinstance(checkpoint_epoch, bool) or not isinstance(checkpoint_epoch, int) or checkpoint_epoch < 0:
            raise RuntimeError("GraphReportTS trainer checkpoint epoch must be a non-negative integer")
        start_epoch = checkpoint_epoch + 1
        best_epoch = int(resume_checkpoint.get("best_epoch", resume_checkpoint.get("epoch", 0)))
        stopped_epoch = checkpoint_epoch
        epoch_seconds = _reconcile_epoch_history(
            out_dir / "epoch_history.jsonl",
            checkpoint_epoch=checkpoint_epoch,
            checkpoint_epoch_seconds=resume_checkpoint.get("epoch_seconds"),
        )
        print(f"resumed from {resume_path} at epoch {start_epoch}; best val_mse={best:.6f}")

    for epoch in range(start_epoch, profile.max_epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        if model_cfg.freeze_text and text_backbone is not None:
            text_backbone.eval()
        scheduler.start_epoch(epoch)
        epoch_lrs = graph_report_group_lrs(opt)
        meters = {k: AverageMeter() for k in ["total", "reg", "align"]}
        gate_stats = _empty_gate_stats()
        weights = _loss_weights(args, profile, epoch)
        pbar = tqdm(train_loader, desc=f"{args.variant}/{args.dataset} epoch {epoch}/{profile.max_epochs}")
        for batch in pbar:
            batch = to_device(batch, device)
            out = _model_forward(model, batch)
            loss = graph_report_loss(out, batch["y"], batch["mask"], weights, loss_type=args.loss)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), profile.gradient_clip)
            opt.step()
            n = _batch_size(batch)
            for k, v in loss.items():
                meters[k].update(float(v.detach()), n)
            _update_gate_stats(gate_stats, out.get("gate"))
            pbar.set_postfix({"loss": meters["total"].avg})
        train_gate = _finalize_gate_stats(gate_stats)
        val = evaluate(model, val_loader, device, weights, args.loss, gate_path=out_dir / f"val_gates_epoch_{epoch}.csv")
        epoch_duration = float(time.perf_counter() - epoch_started)
        epoch_seconds.append(epoch_duration)
        stopped_epoch = epoch
        scheduler.step_validation(epoch, val["mse"])
        score = val["mse"]
        improved = score < best
        if improved:
            best = score
            best_epoch = epoch
        stale = update_graph_report_stale(epoch, stale, improved, profile)
        checkpoint = {
            "model": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "args": vars(args),
            "training_strategy_version": training_strategy_version,
            "training_profile": profile.__dict__,
            "ablation_suite_version": args.ablation_suite_version,
            "trainable_parameter_count": trainable_parameter_count,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "group_lrs": graph_report_group_lrs(opt),
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best": best,
            "stale": stale,
            "val_metrics": val,
            "epoch_seconds": epoch_seconds,
        }
        epoch_row = {
            "epoch": epoch,
            "epoch_seconds": epoch_duration,
            "training_strategy_version": training_strategy_version,
            "core_lr": epoch_lrs["core"],
            "semantic_lr": epoch_lrs["semantic"],
            "align_weight": weights["align"],
            "train_loss": meters["total"].avg,
            "train_reg": meters["reg"].avg,
            "train_align": meters["align"].avg,
            **{f"train_{k}": v for k, v in train_gate.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        with (out_dir / "epoch_history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(epoch_row, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            f"epoch={epoch} val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} "
            f"val_loss={val['total']:.6f} gate_mean={val['gate_mean']:.4f} align_w={weights['align']:.5f} "
            f"core_lr={epoch_lrs['core']:.8f} semantic_lr={epoch_lrs['semantic']:.8f} "
            f"training_strategy_version={training_strategy_version}"
        )
        if improved:
            torch.save(checkpoint, out_dir / "best.pt")
            save_json(val, out_dir / "val_metrics.json")
        torch.save(checkpoint, out_dir / "last.pt")
        if should_stop_graph_report(epoch, stale, profile):
            print(f"early stopping at epoch {epoch}; best val_mse={best:.6f}")
            break
    save_json(
        _run_summary_payload(
            best_epoch=best_epoch,
            stopped_epoch=stopped_epoch,
            epoch_seconds=epoch_seconds,
            trainable_parameter_count=trainable_parameter_count,
            training_strategy_version=training_strategy_version,
            ablation_suite_version=args.ablation_suite_version,
        ),
        out_dir / "run_summary.json",
    )
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(model, test_loader, device, eval_weights, args.loss, gate_path=out_dir / "test_gates.csv")
    save_json(test, out_dir / "test_metrics.json")
    print("test metrics:", test)


if __name__ == "__main__":
    main()
