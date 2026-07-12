from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
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
    from .general_results import GeneralRunWriter
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
    from general_results import GeneralRunWriter
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


def _git_source_commit() -> str:
    """Resolve the exact checked-out main-model source revision for provenance."""

    root = Path(__file__).resolve().parents[1]
    return subprocess.check_output(["git", "rev-parse", "--short=7", "HEAD"], cwd=root, text=True).strip()


def build_general_result_spec(dataset: str, dataset_checksum: str) -> Dict[str, Any]:
    """Build the immutable formal protocol identity for one main-model run."""

    if dataset not in {"ETTm1", "ETTm2", "ETTh1", "ETTh2", "ECL", "Weather"}:
        raise ValueError(f"unknown formal general dataset: {dataset}")
    if len(dataset_checksum) != 64:
        raise ValueError("general dataset checksum must be a SHA-256 digest")
    return {
        "dataset": dataset,
        "dataset_checksum": dataset_checksum,
        "source_commit": _git_source_commit(),
        "protocol": {"input_len": 36, "features": "M", "horizons": [24, 36, 48, 60]},
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_result_output_dir(path: Path, variant: str) -> Path:
    """Preserve legacy battery creation while reserving the general final path for atomic rename."""

    return path if variant == "general" else ensure_dir(path)


def _general_dataset_checksum(dataset: str, data_path: Path) -> str:
    """Fail closed unless the exact trainer input matches the frozen catalog."""

    try:
        from .general_experiment_config import load_general_experiment_spec
    except ImportError:
        from general_experiment_config import load_general_experiment_spec
    config_path = Path(__file__).resolve().parents[1] / "configs" / "general_forecasting" / "experiment_matrix.yaml"
    spec = load_general_experiment_spec(config_path)
    expected = next(item.raw_sha256 for item in spec.datasets if item.name == dataset)
    observed = _sha256_file(data_path)
    if observed != expected:
        raise ValueError(f"formal general data checksum mismatch for {dataset}: {data_path}")
    return observed


def parse_args():
    p = argparse.ArgumentParser(description="Train Battery/General GraphReportTS")
    p.add_argument("--variant", choices=["battery", "general"], default="battery")
    p.add_argument("--dataset", type=str, default="mit")
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_dir", type=str, default="runs/graph_report_ts")
    p.add_argument("--input_len", type=int)
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
    if args.input_len is None:
        args.input_len = 36 if args.variant == "general" else 96
    if args.variant == "general":
        if args.input_len != 36:
            p.error("general forecasting requires --input_len 36")
        if args.pred_len not in (24, 36, 48, 60):
            p.error("general forecasting requires --pred_len one of 24, 36, 48, 60")
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
            precomputed_cache_dir=args.precomputed_cache_dir,
            require_precomputed_cache=args.require_precomputed_cache,
            seed=args.seed,
            max_cycles=args.max_cycles,
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


def _empty_prompt_audit_stats() -> Dict[str, float]:
    return {"token_sum": 0.0, "limit_sum": 0.0, "count": 0.0, "truncated_count": 0.0}


def _update_prompt_audit_stats(stats: Dict[str, float], audit: Optional[List[Dict[str, Any]]]) -> None:
    if audit is None:
        return
    for row in audit:
        stats["token_sum"] += float(row["token_count"])
        stats["limit_sum"] += float(row["token_limit"])
        stats["count"] += 1.0
        stats["truncated_count"] += float(bool(row["truncated"]))


def _finalize_prompt_audit_stats(stats: Dict[str, float]) -> Dict[str, float]:
    count = stats["count"]
    if count <= 0:
        return {}
    return {
        "encoder_token_count_mean": stats["token_sum"] / count,
        "encoder_truncated_count": stats["truncated_count"],
        "encoder_truncated_rate": stats["truncated_count"] / count,
        "encoder_token_limit": stats["limit_sum"] / count,
    }


def _model_forward(model: GraphReportTS, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    steps = None
    if model.cfg.variant == "battery" and not model.cfg.use_relative_steps:
        steps = batch.get("target_steps")
    return model(
        batch["maps"],
        batch["prompt"],
        batch["horizon"],
        steps=steps,
        history_features=batch.get("history_scaled", batch.get("history_features")),
        variable_mask=batch.get("variable_mask"),
    )


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
    collect_standardized_predictions: bool = False,
) -> Dict[str, float]:
    model.eval()
    meters = {k: AverageMeter() for k in ["total", "reg", "align"]}
    gate_stats = _empty_gate_stats()
    prompt_audit_stats = _empty_prompt_audit_stats()
    gate_rows: List[Dict[str, Any]] = []
    predictions: List[torch.Tensor] = []
    targets: List[torch.Tensor] = []
    sample_indices: List[torch.Tensor] = []
    step_indices: List[torch.Tensor] = []
    mse_sum = mae_sum = count = 0.0
    for batch in loader:
        batch = to_device(batch, device)
        out = _model_forward(model, batch)
        loss = graph_report_loss(out, batch["y"], batch["mask"], weights, loss_type=loss_type)
        n = batch["maps"].size(0)
        for k, v in loss.items():
            meters[k].update(float(v.detach()), n)
        _update_gate_stats(gate_stats, out.get("gate"))
        _update_prompt_audit_stats(prompt_audit_stats, out.get("prompt_audit"))
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
        if collect_standardized_predictions:
            predictions.append(out["pred"].detach().cpu())
            targets.append(batch["y"].detach().cpu())
            sample_indices.append(batch["start_index"].detach().cpu())
            step_indices.append(batch["target_steps"].detach().cpu())
    mse = mse_sum / max(count, 1.0)
    mae = mae_sum / max(count, 1.0)
    out = {k: m.avg for k, m in meters.items()}
    out.update({"mse": mse, "mae": mae, "rmse": mse**0.5})
    out.update(_finalize_gate_stats(gate_stats))
    out.update(_finalize_prompt_audit_stats(prompt_audit_stats))
    if gate_path is not None:
        _write_gate_rows(gate_path, gate_rows)
    if collect_standardized_predictions:
        if not predictions:
            raise ValueError("cannot collect predictions from an empty loader")
        out["standardized_predictions"] = torch.cat(predictions, dim=0).numpy()
        out["standardized_targets"] = torch.cat(targets, dim=0).numpy()
        out["sample_indices"] = torch.cat(sample_indices, dim=0).numpy()
        out["step_indices"] = torch.cat(step_indices, dim=0).numpy()
    return out


def main():
    args = parse_args()
    if args.variant == "battery":
        require_formal_battery_protocol(
            observed_cycles=args.history_len,
            prediction_cycles=args.pred_len,
            batch_size=args.batch_size,
            stage="main",
            context="GraphReportTS battery trainer",
        )
    seed_everything(args.seed)
    cfg = ExperimentConfig()
    ensure_research_dirs(cfg)
    out_dir = prepare_result_output_dir(Path(args.out_dir) / args.variant / args.dataset, args.variant)
    device = torch.device(args.device)
    started_at = time.monotonic()

    train_loader, val_loader, test_loader, output_dim = build_loaders(args)
    general_writer: Optional[GeneralRunWriter] = None
    general_data_path: Optional[Path] = None
    if args.variant == "general":
        general_data_path = Path(train_loader.dataset.source_path)
        general_writer = GeneralRunWriter(
            out_dir,
            build_general_result_spec(args.dataset, _general_dataset_checksum(args.dataset, general_data_path)),
        )
        out_dir = general_writer.path
    model_cfg = GraphReportTSConfig(
        variant=args.variant,
        output_dim=output_dim,
        max_steps=60 if args.variant == "general" else 1024,
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
    resume_path = None
    resume_checkpoint = None
    if args.variant == "general":
        # A final general result bundle is immutable; an interrupted .partial run is deliberately rejected.
        args.no_resume = True
    if not args.no_resume:
        last_path = out_dir / "last.pt"
        best_path = out_dir / "best.pt"
        resume_path = last_path if last_path.exists() else best_path
        if resume_path.exists():
            resume_checkpoint = torch.load(resume_path, map_location=device)
            require_checkpoint_strategy_version(resume_checkpoint, "GraphReportTS trainer")
            if "training_profile" in resume_checkpoint:
                profile = MainTrainingProfile(**resume_checkpoint["training_profile"])

    opt = build_graph_report_optimizer(model, profile)
    scheduler = GraphReportScheduler(opt, profile)
    eval_weights = _loss_weights(args, profile, profile.max_epochs)

    run_config = {
        "args": vars(args),
        "model_cfg": model_cfg.__dict__,
        "output_dim": output_dim,
        "training_strategy_version": training_strategy_version,
        "training_profile": profile.__dict__,
    }
    if general_writer is None:
        save_json(run_config, out_dir / "run_config.json")
    else:
        general_writer.write_run_config({"model": "GraphReportTS", "dataset": args.dataset, "seed": args.seed, "metrics_space": "standardized", **run_config})
    best = float("inf")
    stale = 0
    start_epoch = 1
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
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1 if "epoch" in resume_checkpoint else 1
        print(f"resumed from {resume_path} at epoch {start_epoch}; best val_mse={best:.6f}")

    stopped_early = False
    for epoch in range(start_epoch, profile.max_epochs + 1):
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
            n = batch["maps"].size(0)
            for k, v in loss.items():
                meters[k].update(float(v.detach()), n)
            _update_gate_stats(gate_stats, out.get("gate"))
            pbar.set_postfix({"loss": meters["total"].avg})
        train_gate = _finalize_gate_stats(gate_stats)
        val = evaluate(model, val_loader, device, weights, args.loss, gate_path=out_dir / f"val_gates_epoch_{epoch}.csv")
        scheduler.step_validation(epoch, val["mse"])
        score = val["mse"]
        improved = score < best
        if improved:
            best = score
        stale = update_graph_report_stale(epoch, stale, improved, profile)
        checkpoint = {
            "model": model.state_dict(),
            "model_cfg": model_cfg.__dict__,
            "args": vars(args),
            "training_strategy_version": training_strategy_version,
            "training_profile": profile.__dict__,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "group_lrs": graph_report_group_lrs(opt),
            "epoch": epoch,
            "best": best,
            "stale": stale,
            "val_metrics": val,
        }
        epoch_row = {
            "epoch": epoch,
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
        if general_writer is not None:
            general_writer.append_history(epoch_row)
        print(
            f"epoch={epoch} val_mse={val['mse']:.6f} val_mae={val['mae']:.6f} "
            f"val_loss={val['total']:.6f} gate_mean={val['gate_mean']:.4f} align_w={weights['align']:.5f} "
            f"core_lr={epoch_lrs['core']:.8f} semantic_lr={epoch_lrs['semantic']:.8f} "
            f"training_strategy_version={training_strategy_version}"
        )
        if improved:
            if general_writer is None:
                torch.save(checkpoint, out_dir / "best.pt")
                save_json(val, out_dir / "val_metrics.json")
            else:
                general_writer.record_validation(epoch=epoch, mse=val["mse"], checkpoint=checkpoint)
        if general_writer is None:
            torch.save(checkpoint, out_dir / "last.pt")
        if should_stop_graph_report(epoch, stale, profile):
            print(f"early stopping at epoch {epoch}; best val_mse={best:.6f}")
            stopped_early = True
            break
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate(
        model,
        test_loader,
        device,
        eval_weights,
        args.loss,
        gate_path=out_dir / "test_gates.csv",
        collect_standardized_predictions=general_writer is not None,
    )
    if general_writer is None:
        save_json(test, out_dir / "test_metrics.json")
    else:
        general_writer.record_test(
            test.pop("standardized_predictions"),
            test.pop("standardized_targets"),
            sample_indices=test.pop("sample_indices"),
            step_indices=test.pop("step_indices"),
            variable_indices=range(output_dim),
        )
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        general_writer.write_environment({"device": str(device), "cuda_initialized": torch.cuda.is_initialized()})
        general_writer.complete(
            {
                "dataset_checksum": _general_dataset_checksum(args.dataset, general_data_path),
                "source_commit": _git_source_commit(),
                "protocol": {"input_len": 36, "features": "M", "horizons": [24, 36, 48, 60]},
                "source": {"url": "local:GraphReportTS", "commit": _git_source_commit()},
                "runtime": {
                    "wall_time_seconds": time.monotonic() - started_at,
                    "peak_gpu_memory_bytes": int(peak_memory),
                    "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                    "epochs": epoch,
                    "early_stop_reason": "patience" if stopped_early else "max_epochs",
                    "data_path": str(general_data_path),
                },
                "prompt_audit": {
                    "validation": {key: value for key, value in val.items() if key.startswith("encoder_")},
                    "test": {key: value for key, value in test.items() if key.startswith("encoder_")},
                },
            }
        )
    print("test metrics:", test)


if __name__ == "__main__":
    main()
