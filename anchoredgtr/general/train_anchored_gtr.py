"""Train one audited L36 AnchoredGTR cell."""

from __future__ import annotations

import argparse
from pathlib import Path

from anchoredgtr.core import train_general

from .anchored_gtr import ANCHORED_GTR_MODEL_NAME, AnchoredGTR
from .strategy_registry import DATASETS, HORIZONS, resolve_strategy


TRAINING_PROTOCOL_PREFIX = "anchored-gtr-l36-historical-best-registry-v1"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--horizon", type=int, choices=HORIZONS, required=True)
    parser.add_argument("--data-root", default=str(root / "data" / "general"))
    parser.add_argument(
        "--output-root",
        default=str(root / "artifacts" / "general" / "anchored_gtr" / "runs"),
        help="Destination for new runs; curated historical best artifacts remain read-only siblings.",
    )
    parser.add_argument("--provenance-manifest")
    parser.add_argument("--text-model", default=str(root / "hf_models" / "distilbert-base-uncased"))
    parser.add_argument(
        "--text-cache-root",
        default=str(root / "data" / "general" / "cache" / "distilbert_hidden"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mode", choices=("preflight", "formal"), default="formal")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    args = parser.parse_args(argv)
    if args.mode == "formal" and not args.provenance_manifest:
        parser.error("formal mode requires --provenance-manifest")
    return args


def build_general_argv(args: argparse.Namespace) -> list[str]:
    strategy = resolve_strategy(args.dataset, args.horizon)
    seed = strategy.seed if args.seed is None else int(args.seed)
    batch_size = strategy.batch_size if args.batch_size is None else int(args.batch_size)
    workers = strategy.num_workers if args.num_workers is None else int(args.num_workers)
    output = Path(args.output_root).expanduser().resolve() / args.dataset / f"H{args.horizon}" / f"seed-{seed}"
    protocol = f"{TRAINING_PROTOCOL_PREFIX}:{strategy.name}"
    command = [
        "--dataset", args.dataset,
        "--data_root", str(Path(args.data_root).expanduser().resolve()),
        "--input_len", "36",
        "--prompt_len", "36",
        "--horizon", str(args.horizon),
        "--mode", args.mode,
        "--epochs", str(strategy.epochs),
        "--patience", str(strategy.patience),
        "--batch_size", str(batch_size),
        "--num_workers", str(workers),
        "--seed", str(seed),
        "--device", args.device,
        "--amp", "bf16" if args.device.startswith("cuda") else "none",
        "--text_model", str(Path(args.text_model).expanduser().resolve()),
        "--text_backend", "distilbert",
        "--text_hidden_cache_size", "65536",
        "--text_hidden_cache_max_bytes", "17179869184",
        "--text_hidden_precompute_cache",
        "--text_hidden_cache_root", str(Path(args.text_cache_root).expanduser().resolve()),
        "--text_hidden_precompute_batch_size", "128",
        "--prompt_build_workers", "16",
        "--graph_embedding_variant", "series_context_decomp",
        "--linear_anchor_init", "ridge",
        "--linear_anchor_ridge", "1e-4",
        "--correction_gate_mode", strategy.correction_gate_mode,
        "--model_name", ANCHORED_GTR_MODEL_NAME,
        "--training_protocol", protocol,
        "--output", str(output),
    ]
    if args.provenance_manifest:
        command.extend(["--provenance_manifest", str(Path(args.provenance_manifest).expanduser().resolve())])
    if strategy.freeze_linear_anchor:
        command.append("--freeze_linear_anchor")
    if workers > 0:
        command.extend(["--pin_memory", "--persistent_workers", "--prefetch_factor", "2"])
    for name, value in strategy.training_overrides.items():
        command.extend([f"--{name}", str(value)])
    if args.max_train_batches is not None:
        command.extend(["--max_train_batches", str(args.max_train_batches)])
    if args.max_eval_batches is not None:
        command.extend(["--max_eval_batches", str(args.max_eval_batches)])
    last = output / "last.pt"
    if args.resume and last.is_file():
        command.extend(["--resume", str(last)])
    return command


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    train_general.main(build_general_argv(args), model_factory=AnchoredGTR)


if __name__ == "__main__":
    main()
