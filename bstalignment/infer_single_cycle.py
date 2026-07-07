from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

try:
    from .data_mit import (
        MITBatteryCycleDataset,
        StandardScalerTorch,
        collate_cycle_batch,
        load_mit_battery_pkls,
        split_cells,
    )
    from .models import BatteryCycleLLMAssist, BatteryCycleLLMAssistConfig
    from .utils import seed_everything, to_device
except ImportError:
    from data_mit import (
        MITBatteryCycleDataset,
        StandardScalerTorch,
        collate_cycle_batch,
        load_mit_battery_pkls,
        split_cells,
    )
    from models import BatteryCycleLLMAssist, BatteryCycleLLMAssistConfig
    from utils import seed_everything, to_device


def parse_args():
    p = argparse.ArgumentParser(description="Run one-cycle SOH inference and print the generated prompt")
    p.add_argument("--checkpoint", type=str, default="runs/mit_bstalign/best.pt")
    p.add_argument("--data_dir", type=str, default="bstalignment/data/mit")
    p.add_argument("--split", type=str, choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--cell_id", type=str, required=True, help="Cell id, for example batch1_b1c6")
    p.add_argument("--cycle", type=int, required=True, help="Input cycle number")
    p.add_argument("--forecast_horizon", type=int, default=None, help="Future cycles to predict; default uses checkpoint max_horizon")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def select_records(records, split: str, seed: int):
    if split == "all":
        return records
    train_recs, val_recs, test_recs = split_cells(records, seed=seed)
    return {"train": train_recs, "val": val_recs, "test": test_recs}[split]


def find_sample(ds: MITBatteryCycleDataset, cell_id: str, cycle: int) -> dict:
    for sample in ds.samples:
        if sample["cell_id"] != cell_id:
            continue
        _, df = ds.cells[sample["cell_id"]]
        row = df.iloc[int(sample["idx"])]
        if int(row["cycle"]) == int(cycle):
            return sample
    available = sorted(ds.cells.keys())
    preview = ", ".join(available[:10])
    raise RuntimeError(
        f"No sample found for cell_id={cell_id}, cycle={cycle}. "
        f"Available cell examples: {preview}"
    )


@torch.no_grad()
def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = BatteryCycleLLMAssistConfig(**ckpt["cfg"])
    model = BatteryCycleLLMAssist(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    records = load_mit_battery_pkls(args.data_dir)
    recs = select_records(records, args.split, seed=ckpt.get("args", {}).get("seed", args.seed))
    scaler = StandardScalerTorch.from_state_dict(ckpt["scaler"])
    requested_horizon = cfg.max_horizon if args.forecast_horizon is None else min(args.forecast_horizon, cfg.max_horizon)
    ds = MITBatteryCycleDataset(
        recs,
        max_horizon=requested_horizon,
        min_history=ckpt.get("args", {}).get("min_history", 5),
        features=ckpt["features"],
        scaler=scaler,
        fit_scaler=False,
        random_horizon=False,
        max_cycles=ckpt.get("args", {}).get("max_cycles", None),
        seed=args.seed,
    )
    ds.samples = [find_sample(ds, args.cell_id, args.cycle)]

    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_cycle_batch)
    batch = next(iter(loader))
    batch = to_device(batch, device)
    out = model(batch["x"], batch["prompt"], batch["horizon"])

    horizon = int(batch["horizon"][0].detach().cpu())
    soh_now = float(out["soh_now"][0].detach().cpu())
    soh_future = out["soh_future"][0, :horizon].detach().cpu().tolist()
    future_cycles = batch["future_cycles"][0, :horizon].detach().cpu().tolist()

    print("Generated prompt:")
    print(batch["prompt"][0])
    print()
    print(f"Input: cell_id={args.cell_id}, cycle={int(batch['cycle'][0].detach().cpu())}, forecast_horizon={horizon}")
    print(f"soh_now_pred={soh_now:.6f}")
    print("soh_future_pred:")
    for cyc, pred in zip(future_cycles, soh_future):
        print(f"  cycle {int(cyc)}: {float(pred):.6f}")


if __name__ == "__main__":
    main()
    # python -m bstalignment.infer_single_cycle --checkpoint runs/mit_bstalign/best.pt --cell_id batch1_b1c6 --cycle 120 --forecast_horizon 20