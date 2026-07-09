from __future__ import annotations

import argparse
import importlib
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .train_battery_baselines import BatterySequenceDataset, FEATURES
    from .utils import AverageMeter, ensure_dir, save_json, seed_everything
except ImportError:
    from train_battery_baselines import BatterySequenceDataset, FEATURES
    from utils import AverageMeter, ensure_dir, save_json, seed_everything


OFFICIAL_REPO_DIRS = {
    "patchtst": Path("patchtst") / "PatchTST_supervised",
    "itransformer": Path("itransformer"),
    "timesnet": Path("timesnet"),
    "dlinear": Path("dlinear"),
    "timecma": Path("timecma"),
    "time_llm": Path("time_llm"),
}


@contextmanager
def external_import_path(root: Path):
    root = root.resolve()
    old_path = list(sys.path)
    sys.path.insert(0, str(root))
    try:
        yield
    finally:
        sys.path[:] = old_path


def git_commit(path: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def require_external_root(external_root: Path, name: str) -> Path:
    rel = OFFICIAL_REPO_DIRS[name]
    root = external_root / rel
    if not root.exists():
        raise FileNotFoundError(f"Official baseline source for {name} not found under {root}")
    return root


def base_forecast_config(args, num_features: int) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=args.input_len,
        label_len=0,
        pred_len=args.pred_len,
        enc_in=num_features,
        dec_in=num_features,
        c_out=num_features,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        fc_dropout=args.dropout,
        head_dropout=args.dropout,
        factor=1,
        embed=args.embed,
        freq=args.freq,
        activation="gelu",
        output_attention=False,
        use_norm=True,
        class_strategy="projection",
        moving_avg=args.moving_avg,
        individual=False,
        patch_len=args.patch_len,
        stride=args.stride,
        padding_patch="end",
        revin=True,
        affine=False,
        subtract_last=False,
        decomposition=False,
        kernel_size=args.moving_avg,
        top_k=args.top_k,
        num_kernels=args.num_kernels,
        llm_model=args.llm_model,
        llm_dim=args.llm_dim,
        llm_layers=args.llm_layers,
        prompt_domain=1,
        content=(
            "Battery state-of-health forecasting from historical SOH and capacity-summary "
            "cycle sequences."
        ),
    )


def select_target(pred: torch.Tensor) -> torch.Tensor:
    if pred.ndim == 3:
        return pred[:, :, 0]
    return pred


class TimeCMAPromptEmbeddings(nn.Module):
    """Generate TimeCMA-style prompt embeddings with a frozen local HF encoder."""

    def __init__(self, model_path: str, d_llm: int = 768, max_length: int = 96):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.cls_token
        self.model = AutoModel.from_pretrained(model_path, local_files_only=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        hidden = int(getattr(self.model.config, "hidden_size", getattr(self.model.config, "n_embd", d_llm)))
        self.proj = nn.Identity() if hidden == d_llm else nn.Linear(hidden, d_llm)
        self.max_length = int(max_length)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.eval()
        return self

    def _prompts(self, x: torch.Tensor) -> List[str]:
        prompts: List[str] = []
        names = ["SOH", "capacity summary"]
        x_cpu = x.detach().float().cpu()
        for b in range(x_cpu.size(0)):
            for j in range(x_cpu.size(2)):
                values = x_cpu[b, :, j]
                trend = float(values[-1] - values[0])
                prompts.append(
                    "Battery cycle history for {name}: min {mn:.6f}, max {mx:.6f}, "
                    "median {med:.6f}, trend {trend:.6f}. Forecast future SOH.".format(
                        name=names[j] if j < len(names) else f"feature {j}",
                        mn=float(values.min()),
                        mx=float(values.max()),
                        med=float(values.median()),
                        trend=trend,
                    )
                )
        return prompts

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, n_vars = x.shape
        tok = self.tokenizer(
            self._prompts(x),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        device = next(self.parameters()).device
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.model(**tok).last_hidden_state
        mask = tok.get("attention_mask", torch.ones(out.shape[:2], device=device)).bool()
        last_idx = mask.long().sum(dim=1).clamp(min=1) - 1
        pooled = out[torch.arange(out.size(0), device=device), last_idx]
        pooled = self.proj(pooled)
        return pooled.view(bsz, n_vars, -1).permute(0, 2, 1).contiguous()


def patch_timellm_hf_paths(module, gpt2_path: str, bert_path: str) -> None:
    mapping = {
        "openai-community/gpt2": gpt2_path,
        "google-bert/bert-base-uncased": bert_path,
    }

    def patch_class(cls):
        original = cls.from_pretrained

        def mapped_from_pretrained(name, *args, **kwargs):
            return original(mapping.get(name, name), *args, **kwargs)

        cls.from_pretrained = staticmethod(mapped_from_pretrained)

    for attr in ["GPT2Config", "GPT2Model", "GPT2Tokenizer", "BertConfig", "BertModel", "BertTokenizer"]:
        if hasattr(module, attr):
            patch_class(getattr(module, attr))


class OfficialBaseline(nn.Module):
    def __init__(self, name: str, args, num_features: int):
        super().__init__()
        self.name = name
        self.args = args
        self.num_features = num_features
        self.model = self._build(name, args, num_features)
        self.timecma_prompt = None
        if name == "timecma":
            self.timecma_prompt = TimeCMAPromptEmbeddings(args.hf_gpt2_model, d_llm=args.llm_dim)

    def _build(self, name: str, args, num_features: int) -> nn.Module:
        external_root = Path(args.external_root)
        cfg = base_forecast_config(args, num_features)
        if name == "patchtst":
            root = require_external_root(external_root, name)
            with external_import_path(root):
                return importlib.import_module("models.PatchTST").Model(cfg)
        if name == "itransformer":
            root = require_external_root(external_root, name)
            with external_import_path(root):
                return importlib.import_module("model.iTransformer").Model(cfg)
        if name == "timesnet":
            root = require_external_root(external_root, name)
            with external_import_path(root):
                return importlib.import_module("models.TimesNet").Model(cfg)
        if name == "dlinear":
            root = require_external_root(external_root, name)
            with external_import_path(root):
                return importlib.import_module("models.DLinear").Model(cfg)
        if name == "timecma":
            root = require_external_root(external_root, name)
            with external_import_path(root):
                dual = importlib.import_module("models.TimeCMA").Dual
                return dual(
                    device=args.device,
                    channel=args.timecma_channel,
                    num_nodes=num_features,
                    seq_len=args.input_len,
                    pred_len=args.pred_len,
                    dropout_n=args.dropout,
                    d_llm=args.llm_dim,
                    e_layer=args.e_layers,
                    d_layer=args.d_layers,
                    d_ff=args.d_ff,
                    head=args.n_heads,
                )
        if name == "time_llm":
            root = require_external_root(external_root, name)
            with external_import_path(root):
                module = importlib.import_module("models.TimeLLM")
                patch_timellm_hf_paths(module, args.hf_gpt2_model, args.hf_bert_model)
                return module.Model(cfg)
        raise ValueError(f"Unknown official baseline: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.name == "patchtst":
            return select_target(self.model(x))
        if self.name in {"itransformer", "timesnet"}:
            return select_target(self.model(x, None, None, None))
        if self.name == "dlinear":
            try:
                return select_target(self.model(x, None, None, None))
            except TypeError:
                return select_target(self.model(x))
        if self.name == "timecma":
            mark = torch.zeros(x.size(0), x.size(1), 1, dtype=x.dtype, device=x.device)
            emb = self.timecma_prompt(x)
            return select_target(self.model(x, mark, emb))
        if self.name == "time_llm":
            if x.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    return select_target(self.model(x, None, None, None)).float()
            return select_target(self.model(x, None, None, None))
        raise ValueError(f"Unknown official baseline: {self.name}")


def metrics(pred: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    diff = pred.detach() - y.detach()
    mse = float((diff**2).mean().cpu())
    mae = float(diff.abs().mean().cpu())
    return {"mse": mse, "mae": mae, "rmse": mse**0.5}


def run_epoch(model, loader, device, opt=None) -> Dict[str, float]:
    train = opt is not None
    model.train(train)
    loss_meter = AverageMeter()
    mse_sum = mae_sum = count = 0.0
    with torch.set_grad_enabled(train):
        for batch in tqdm(loader, desc="train" if train else "eval"):
            x = batch["x"].to(device)
            y = batch["y"].to(device)
            pred = model(x)
            loss = F.smooth_l1_loss(pred, y)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
            n = x.size(0)
            loss_meter.update(float(loss.detach()), n)
            m = metrics(pred, y)
            elems = y.numel()
            mse_sum += m["mse"] * elems
            mae_sum += m["mae"] * elems
            count += elems
    mse = mse_sum / max(count, 1.0)
    mae = mae_sum / max(count, 1.0)
    return {"loss": loss_meter.avg, "mse": mse, "mae": mae, "rmse": mse**0.5}


def parse_args():
    p = argparse.ArgumentParser(description="Train official battery SOH baselines on shared splits")
    p.add_argument("--model", choices=["patchtst", "itransformer", "timecma", "timesnet", "dlinear", "time_llm"], required=True)
    p.add_argument("--dataset", choices=["mit", "calce", "xjtu"], required=True)
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_dir", type=str, default="runs/full_hf/baselines")
    p.add_argument("--external_root", type=str, default="external")
    p.add_argument("--input_len", type=int, default=32)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--d_ff", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--e_layers", type=int, default=2)
    p.add_argument("--d_layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--patch_len", type=int, default=8)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--moving_avg", type=int, default=5)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--num_kernels", type=int, default=3)
    p.add_argument("--embed", type=str, default="fixed")
    p.add_argument("--freq", type=str, default="h")
    p.add_argument("--llm_model", choices=["GPT2", "BERT"], default="GPT2")
    p.add_argument("--llm_dim", type=int, default=768)
    p.add_argument("--llm_layers", type=int, default=6)
    p.add_argument("--timecma_channel", type=int, default=128)
    p.add_argument("--hf_gpt2_model", type=str, default="openai-community/gpt2")
    p.add_argument("--hf_bert_model", type=str, default="google-bert/bert-base-uncased")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--no_resume", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    ds_kwargs = dict(
        dataset_name=args.dataset,
        data_root=args.data_root,
        input_len=args.input_len,
        pred_len=args.pred_len,
        seed=args.seed,
        max_cycles=args.max_cycles,
    )
    train_ds = BatterySequenceDataset(split="train", **ds_kwargs)
    val_ds = BatterySequenceDataset(split="val", **ds_kwargs)
    test_ds = BatterySequenceDataset(split="test", **ds_kwargs)
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise RuntimeError("Official baseline train/test split is empty. Check data and input_len/pred_len.")
    val_eval_ds = val_ds if len(val_ds) else test_ds
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs.update({"prefetch_factor": 2})
    train_loader = DataLoader(train_ds, shuffle=True, persistent_workers=args.num_workers > 0, **loader_kwargs)
    val_loader = DataLoader(val_eval_ds, shuffle=False, persistent_workers=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, persistent_workers=False, **loader_kwargs)

    model = OfficialBaseline(args.model, args, len(FEATURES)).to(device)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    out_dir = ensure_dir(Path(args.out_dir) / args.dataset / args.model)
    source_dir = Path(args.external_root) / OFFICIAL_REPO_DIRS[args.model].parts[0]
    save_json(
        {
            "args": vars(args),
            "features": FEATURES,
            "num_train": len(train_ds),
            "num_val": len(val_ds),
            "num_test": len(test_ds),
            "official_source_dir": str(source_dir),
            "official_source_commit": git_commit(source_dir),
            "adapter": "project-side battery sequence adapter with official model class",
        },
        out_dir / "run_config.json",
    )
    best = float("inf")
    stale = 0
    start_epoch = 1
    if not args.no_resume:
        last_path = out_dir / "last.pt"
        best_path = out_dir / "best.pt"
        resume_path = last_path if last_path.exists() else best_path
        if resume_path.exists():
            ckpt = torch.load(resume_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                opt.load_state_dict(ckpt["optimizer"])
            val_metrics = ckpt.get("val_metrics", {})
            best = float(ckpt.get("best", val_metrics.get("mse", best)))
            stale = int(ckpt.get("stale", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1 if "epoch" in ckpt else 1
            print(f"resumed from {resume_path} at epoch {start_epoch}; best val_mse={best:.6f}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_m = run_epoch(model, train_loader, device, opt)
        val_m = run_epoch(model, val_loader, device)
        print(f"epoch={epoch} train_loss={train_m['loss']:.6f} val_mse={val_m['mse']:.6f} val_mae={val_m['mae']:.6f}")
        if val_m["mse"] < best:
            best = val_m["mse"]
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "best": best,
                    "stale": stale,
                    "val_metrics": val_m,
                },
                out_dir / "best.pt",
            )
            save_json(val_m, out_dir / "val_metrics.json")
        else:
            stale += 1
        torch.save(
            {
                "model": model.state_dict(),
                "args": vars(args),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best": best,
                "stale": stale,
                "val_metrics": val_m,
            },
            out_dir / "last.pt",
        )
        if stale >= args.early_stop_patience:
            print(f"early stopping at epoch {epoch}; best val_mse={best:.6f}")
            break
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = run_epoch(model, test_loader, device)
    save_json(test_m, out_dir / "test_metrics.json")
    print("test metrics:", test_m)


if __name__ == "__main__":
    main()
