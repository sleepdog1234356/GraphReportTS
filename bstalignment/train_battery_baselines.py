from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .battery_protocol import fit_cycle_scale, fit_processed_cycle_scale, split_processed_items
    from .data_mit import add_cycle_features, load_mit_battery_pkls, split_cells
    from .utils import AverageMeter, ensure_dir, save_json, seed_everything
except ImportError:
    from battery_protocol import fit_cycle_scale, fit_processed_cycle_scale, split_processed_items
    from data_mit import add_cycle_features, load_mit_battery_pkls, split_cells
    from utils import AverageMeter, ensure_dir, save_json, seed_everything


FEATURES = ["capacity_summary", "capacity_delta", "internal_resistance", "charge_time", "cycle_ratio"]


def _as_float_series(values: np.ndarray | None, n: int, fill: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full(n, float(fill), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(arr) < n:
        out = np.full(n, float(fill), dtype=np.float32)
        out[: len(arr)] = arr
        return out
    return arr[:n].astype(np.float32)


def _cycle_ratio(cycle_ids: np.ndarray | None, n: int, cycle_scale: float) -> np.ndarray:
    if cycle_ids is None:
        arr = np.arange(1, n + 1, dtype=np.float32)
    else:
        arr = _as_float_series(cycle_ids, n, fill=0.0)
    return (arr / max(float(cycle_scale), 1.0)).astype(np.float32)


class BatterySequenceDataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        data_root: str = "bstalignment/data",
        split: str = "train",
        input_len: int = 32,
        pred_len: int = 20,
        seed: int = 42,
        max_cycles: int | None = None,
    ):
        self.dataset_name = dataset_name.lower()
        self.data_root = Path(data_root)
        self.split = split
        self.input_len = int(input_len)
        self.pred_len = int(pred_len)
        self.series: Dict[str, np.ndarray] = {}
        self.targets: Dict[str, np.ndarray] = {}
        self.samples: List[Tuple[str, int]] = []
        self.cycle_scale = 1.0
        if self.dataset_name == "mit":
            self._load_mit(seed, max_cycles)
        else:
            self._load_processed(seed, max_cycles)

    def _add_series(
        self,
        cell_id: str,
        soh: np.ndarray,
        capacity: np.ndarray | None,
        internal_resistance: np.ndarray | None = None,
        charge_time: np.ndarray | None = None,
        cycle_ids: np.ndarray | None = None,
    ) -> None:
        soh = np.asarray(soh, dtype=np.float32).reshape(-1)
        n = len(soh)
        capacity = _as_float_series(capacity, n)
        internal_resistance = _as_float_series(internal_resistance, n)
        charge_time = _as_float_series(charge_time, n)
        values = np.stack(
            [
                capacity,
                np.diff(capacity, prepend=capacity[0]).astype(np.float32) if n else capacity,
                internal_resistance,
                charge_time,
                _cycle_ratio(cycle_ids, n, self.cycle_scale),
            ],
            axis=-1,
        )
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        target = np.nan_to_num(soh, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        if len(values) < self.input_len + self.pred_len:
            return
        self.series[cell_id] = values
        self.targets[cell_id] = target
        for start in range(0, len(values) - self.input_len - self.pred_len + 1):
            self.samples.append((cell_id, start))

    def _load_mit(self, seed: int, max_cycles: int | None) -> None:
        records = load_mit_battery_pkls(self.data_root / "mit")
        train, val, test = split_cells(records, seed=seed)
        self.cycle_scale = fit_cycle_scale(
            (record.summary["cycle"].to_numpy(dtype=np.float64) for record in train),
            max_cycles,
        )
        selected = {"train": train, "val": val, "test": test, "all": records}[self.split]
        for rec in selected:
            df = add_cycle_features(rec.summary)
            if max_cycles is not None:
                df = df.iloc[: int(max_cycles)].copy()
            self._add_series(
                rec.cell_id,
                df["SOH"].to_numpy(np.float32),
                df["QD"].to_numpy(np.float32),
                df["IR"].to_numpy(np.float32),
                df["chargetime"].to_numpy(np.float32),
                df["cycle"].to_numpy(np.float32),
            )

    def _load_processed(self, seed: int, max_cycles: int | None) -> None:
        root = self.data_root / "processed" / "battery" / self.dataset_name
        files = sorted(root.glob("*.npz"))
        if not files:
            raise FileNotFoundError(f"No processed {self.dataset_name} npz files under {root}")
        splits = split_processed_items(files, seed=seed)
        self.cycle_scale = fit_processed_cycle_scale(splits["train"], max_cycles)
        for path in splits[self.split]:
            data = np.load(path, allow_pickle=True)
            soh = np.asarray(data["soh"], dtype=np.float32)
            capacity = np.asarray(data["capacity_summary"], dtype=np.float32) if "capacity_summary" in data else None
            if capacity is None and "capacity" in data:
                capacity_arr = np.asarray(data["capacity"], dtype=np.float32)
                capacity = capacity_arr[:, -1] if capacity_arr.ndim == 2 else capacity_arr.reshape(len(soh), -1)[:, -1]
            internal_resistance = np.asarray(data["internal_resistance"], dtype=np.float32) if "internal_resistance" in data else None
            charge_time = np.asarray(data["charge_time"], dtype=np.float32) if "charge_time" in data else None
            cycle_ids = np.asarray(data["cycle_id"], dtype=np.float32) if "cycle_id" in data else None
            if max_cycles is not None:
                soh = soh[: int(max_cycles)]
                capacity = capacity[: int(max_cycles)] if capacity is not None else None
                internal_resistance = internal_resistance[: int(max_cycles)] if internal_resistance is not None else None
                charge_time = charge_time[: int(max_cycles)] if charge_time is not None else None
                cycle_ids = cycle_ids[: int(max_cycles)] if cycle_ids is not None else None
            self._add_series(path.stem, soh, capacity, internal_resistance, charge_time, cycle_ids)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        cell_id, start = self.samples[idx]
        values = self.series[cell_id]
        target = self.targets[cell_id]
        x = values[start : start + self.input_len]
        y = target[start + self.input_len : start + self.input_len + self.pred_len]
        return {
            "x": torch.tensor(x, dtype=torch.float32),
            "y": torch.tensor(y, dtype=torch.float32),
            "cell_id": cell_id,
            "start": torch.tensor(start, dtype=torch.long),
        }


class MovingAverage(nn.Module):
    def __init__(self, kernel_size: int = 5):
        super().__init__()
        self.kernel_size = int(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.avg_pool1d(x.transpose(1, 2), self.kernel_size, stride=1, padding=pad, count_include_pad=False).transpose(1, 2)


class DLinearBaseline(nn.Module):
    def __init__(self, input_len: int, pred_len: int, num_features: int):
        super().__init__()
        self.decomp = MovingAverage(5)
        self.trend = nn.Linear(input_len, pred_len)
        self.residual = nn.Linear(input_len, pred_len)
        self.proj = nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        trend = self.decomp(x)
        residual = x - trend
        out = self.trend(trend.transpose(1, 2)) + self.residual(residual.transpose(1, 2))
        return self.proj(out.transpose(1, 2)).squeeze(-1)


class PatchTSTBaseline(nn.Module):
    def __init__(self, input_len: int, pred_len: int, num_features: int, d_model: int, patch_len: int = 8, stride: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.embed = nn.Linear(patch_len * num_features, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).reshape(x.size(0), patches.size(1), -1)
        z = self.encoder(self.embed(patches)).mean(dim=1)
        return self.head(z)


class ITransformerBaseline(nn.Module):
    def __init__(self, input_len: int, pred_len: int, num_features: int, d_model: int):
        super().__init__()
        self.value_proj = nn.Linear(input_len, d_model)
        self.feature_embed = nn.Parameter(torch.randn(num_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.value_proj(x.transpose(1, 2)) + self.feature_embed.unsqueeze(0)
        z = self.encoder(z)
        return self.head(z[:, 0, :])


class TimesBlock(nn.Module):
    def __init__(self, channels: int, d_model: int):
        super().__init__()
        self.in_proj = nn.Linear(channels, d_model)
        self.conv3 = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x).transpose(1, 2)
        z = (self.conv3(z) + self.conv5(z) + self.conv7(z)) / 3.0
        return self.norm(F.gelu(z).transpose(1, 2))


class TimesNetBaseline(nn.Module):
    def __init__(self, pred_len: int, num_features: int, d_model: int):
        super().__init__()
        self.block1 = TimesBlock(num_features, d_model)
        self.block2 = TimesBlock(d_model, d_model)
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.block2(self.block1(x)).mean(dim=1)
        return self.head(z)


class TimeCMABaseline(nn.Module):
    """Lightweight TimeCMA-style adapter: inverted tokens plus trend/residual fusion."""

    def __init__(self, input_len: int, pred_len: int, num_features: int, d_model: int):
        super().__init__()
        self.inverted = ITransformerBaseline(input_len, pred_len, num_features, d_model)
        self.trend = DLinearBaseline(input_len, pred_len, num_features)
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.gate)
        return g * self.inverted(x) + (1.0 - g) * self.trend(x)


def build_model(name: str, input_len: int, pred_len: int, num_features: int, d_model: int) -> nn.Module:
    name = name.lower()
    if name == "dlinear":
        return DLinearBaseline(input_len, pred_len, num_features)
    if name == "patchtst":
        return PatchTSTBaseline(input_len, pred_len, num_features, d_model)
    if name == "itransformer":
        return ITransformerBaseline(input_len, pred_len, num_features, d_model)
    if name == "timesnet":
        return TimesNetBaseline(pred_len, num_features, d_model)
    if name == "timecma":
        return TimeCMABaseline(input_len, pred_len, num_features, d_model)
    raise ValueError(f"Unknown baseline: {name}")


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
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
    p = argparse.ArgumentParser(description="Train battery SOH baselines on shared processed splits")
    p.add_argument("--model", choices=["patchtst", "itransformer", "timecma", "timesnet", "dlinear"], required=True)
    p.add_argument("--dataset", choices=["mit", "calce", "xjtu"], required=True)
    p.add_argument("--data_root", type=str, default="bstalignment/data")
    p.add_argument("--out_dir", type=str, default="runs/baselines")
    p.add_argument("--input_len", type=int, default=32)
    p.add_argument("--pred_len", type=int, default=20)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--early_stop_patience", type=int, default=8)
    p.add_argument("--max_cycles", type=int, default=None)
    p.add_argument("--no_resume", action="store_true", help="Disable automatic resume from last.pt/best.pt in the output directory")
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
        raise RuntimeError("Baseline train/test split is empty. Check data and input_len/pred_len.")
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
    model = build_model(args.model, args.input_len, args.pred_len, len(FEATURES), args.d_model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir = ensure_dir(Path(args.out_dir) / args.dataset / args.model)
    save_json({"args": vars(args), "features": FEATURES, "num_train": len(train_ds), "num_val": len(val_ds), "num_test": len(test_ds)}, out_dir / "run_config.json")
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
            break
    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test_m = run_epoch(model, test_loader, device)
    save_json(test_m, out_dir / "test_metrics.json")
    print("test metrics:", test_m)


if __name__ == "__main__":
    main()
