from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class RawCycleRecord:
    cell_id: str
    cycle_id: int
    channels: Dict[str, np.ndarray]
    target: float
    metadata: Dict[str, str]


def resample_1d(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(x) == 0:
        return np.zeros(length, dtype=np.float32)
    if len(x) == length:
        return x.astype(np.float32)
    src = np.linspace(0.0, 1.0, len(x), dtype=np.float32)
    dst = np.linspace(0.0, 1.0, length, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def robust_scale(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    med = np.nanmedian(x)
    q25 = np.nanpercentile(x, 25)
    q75 = np.nanpercentile(x, 75)
    scale = max(float(q75 - q25), eps)
    return ((x - med) / scale).astype(np.float32)


def current_to_capacity(time: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Integrate current to Ah using trapezoidal increments.

    If time is already normalized or unavailable, the output is still a useful
    monotone proxy after resampling, but exact Ah requires time in seconds.
    """
    t = np.asarray(time, dtype=np.float32).reshape(-1)
    i = np.asarray(current, dtype=np.float32).reshape(-1)
    if len(t) != len(i) or len(t) < 2:
        return np.zeros_like(i, dtype=np.float32)
    dt = np.diff(t, prepend=t[0])
    dq = i * dt / 3600.0
    return np.cumsum(dq).astype(np.float32)


def smooth_gradient(y: np.ndarray, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    if len(y) != len(x) or len(y) < 3:
        return np.zeros_like(y, dtype=np.float32)
    dx = np.gradient(x)
    dy = np.gradient(y)
    return (dy / np.where(np.abs(dx) < eps, eps, dx)).astype(np.float32)


def hankel_map(x: np.ndarray, delay_dim: int, delay_lag: int) -> np.ndarray:
    """Convert a 1D sequence to a 2D delay/Hankel map.

    Rows are delayed views, columns are local states. This preserves local
    dynamics without forcing the downstream model to see a very long raw vector.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    delay_dim = max(int(delay_dim), 1)
    delay_lag = max(int(delay_lag), 1)
    width = len(x) - (delay_dim - 1) * delay_lag
    if width <= 0:
        x = resample_1d(x, delay_dim * delay_lag + 1)
        width = len(x) - (delay_dim - 1) * delay_lag
    rows = [x[i * delay_lag : i * delay_lag + width] for i in range(delay_dim)]
    return np.stack(rows, axis=0).astype(np.float32)


def derivative_maps(x: np.ndarray, delay_dim: int, delay_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    d1 = np.diff(x, prepend=x[0])
    d2 = np.diff(d1, prepend=d1[0])
    return hankel_map(d1, delay_dim, delay_lag), hankel_map(d2, delay_dim, delay_lag)


def build_multiview_maps(
    channels: Dict[str, np.ndarray],
    resample_len: int = 128,
    delay_dim: int = 8,
    delay_lag: int = 1,
    include_derivatives: bool = True,
    include_hankel: bool = True,
    include_ic_dv: bool = False,
) -> Tuple[np.ndarray, List[str]]:
    """Build multi-view 2D maps from raw channels.

    Output shape: [C_map, delay_dim, width]. The maps are intentionally generic:
    Hankel maps and derivative maps work for batteries, weather, traffic, and
    electricity. IC/DV is only enabled for battery SOH experiments.
    """
    base = {}
    for name, values in channels.items():
        if values is None:
            continue
        base[name] = robust_scale(resample_1d(values, resample_len))

    if include_ic_dv and "capacity" in base and "voltage" in base:
        cap = base["capacity"]
        volt = base["voltage"]
        base["ic_dqdv"] = robust_scale(smooth_gradient(cap, volt))
        base["dv_dvdq"] = robust_scale(smooth_gradient(volt, cap))

    maps: List[np.ndarray] = []
    names: List[str] = []
    for name, values in base.items():
        if include_hankel:
            maps.append(hankel_map(values, delay_dim, delay_lag))
            names.append(f"{name}:hankel")
        if include_derivatives:
            d1, d2 = derivative_maps(values, delay_dim, delay_lag)
            maps.extend([d1, d2])
            names.extend([f"{name}:d1", f"{name}:d2"])
    if not maps:
        maps.append(np.zeros((delay_dim, resample_len - delay_dim + 1), dtype=np.float32))
        names.append("empty:hankel")
    min_width = min(m.shape[1] for m in maps)
    maps = [m[:, :min_width] for m in maps]
    return np.stack(maps, axis=0).astype(np.float32), names


def build_variable_maps(
    x: np.ndarray,
    resample_len: int = 128,
    delay_dim: int = 8,
    delay_lag: int = 1,
    include_derivatives: bool = True,
    include_hankel: bool = True,
) -> np.ndarray:
    """Build independent multi-view maps shaped [variables, views, height, width]."""
    values = np.asarray(x, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected history [time,variables], got {values.shape}")
    variable_maps = []
    for variable_index in range(values.shape[1]):
        maps, _ = build_multiview_maps(
            {f"x{variable_index}": values[:, variable_index]},
            resample_len=resample_len,
            delay_dim=delay_dim,
            delay_lag=delay_lag,
            include_derivatives=include_derivatives,
            include_hankel=include_hankel,
            include_ic_dv=False,
        )
        variable_maps.append(maps)
    if not variable_maps:
        raise ValueError("Variable maps require at least one input variable")
    return np.stack(variable_maps, axis=0).astype(np.float32)


VARIABLE_GRAPH_STATISTICS = ("mean", "std", "min", "max", "q25", "q75")


def aggregate_variable_maps(variable_maps: np.ndarray) -> np.ndarray:
    """Aggregate variable-wise maps to fixed graph channels.

    The input variable axis is reduced independently for every map view and
    pixel.  The resulting channel count is therefore independent of the raw
    feature count while the unaggregated history remains available to the
    variable-specific numeric encoder.
    """
    values = np.asarray(variable_maps, dtype=np.float32)
    if values.ndim != 4 or values.shape[0] < 1:
        raise ValueError(
            "Expected variable maps [variables,views,height,width] with at least one variable"
        )
    statistics = (
        values.mean(axis=0),
        values.std(axis=0),
        values.min(axis=0),
        values.max(axis=0),
        np.quantile(values, 0.25, axis=0),
        np.quantile(values, 0.75, axis=0),
    )
    return np.concatenate(statistics, axis=0).astype(np.float32, copy=False)


def maps_to_patch_nodes(
    maps: torch.Tensor,
    patch_size: int = 8,
    patch_stride: int = 4,
) -> torch.Tensor:
    """Convert [B,C,H,W] maps to patch-node features [B,N,P]."""
    if maps.ndim != 4:
        raise ValueError(f"Expected maps [B,C,H,W], got {tuple(maps.shape)}")
    b, c, h, w = maps.shape
    patch_size = max(int(patch_size), 1)
    patch_stride = max(int(patch_stride), 1)
    if h < patch_size or w < patch_size:
        pad_h = max(patch_size - h, 0)
        pad_w = max(patch_size - w, 0)
        maps = torch.nn.functional.pad(maps, (0, pad_w, 0, pad_h))
    patches = torch.nn.functional.unfold(maps, kernel_size=patch_size, stride=patch_stride)
    return patches.transpose(1, 2).contiguous()


def maps_to_channel_patch_nodes(
    maps: torch.Tensor,
    patch_size: int = 8,
    patch_stride: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int, int]]:
    """Convert [B,C,H,W] maps to per-channel patch nodes.

    Unlike `maps_to_patch_nodes`, this preserves map/channel identity. Nodes are
    ordered as channel-major: channel c, patch row r, patch col w.

    Returns:
      nodes: [B, C*R*W, patch_size*patch_size]
      meta: [N, 3] with channel, patch_row, patch_col
      grid: (C, R, W)
    """
    if maps.ndim != 4:
        raise ValueError(f"Expected maps [B,C,H,W], got {tuple(maps.shape)}")
    b, c, h, w = maps.shape
    patch_size = max(int(patch_size), 1)
    patch_stride = max(int(patch_stride), 1)
    if h < patch_size or w < patch_size:
        pad_h = max(patch_size - h, 0)
        pad_w = max(patch_size - w, 0)
        maps = torch.nn.functional.pad(maps, (0, pad_w, 0, pad_h))
        h, w = maps.shape[-2:]
    per_channel = maps.reshape(b * c, 1, h, w)
    patches = torch.nn.functional.unfold(per_channel, kernel_size=patch_size, stride=patch_stride)
    patch_dim, num_patches = patches.shape[1], patches.shape[2]
    patch_rows = (h - patch_size) // patch_stride + 1
    patch_cols = (w - patch_size) // patch_stride + 1
    patches = patches.transpose(1, 2).reshape(b, c * num_patches, patch_dim)

    meta_rows = []
    for ci in range(c):
        for ri in range(patch_rows):
            for wi in range(patch_cols):
                meta_rows.append((ci, ri, wi))
    meta = torch.tensor(meta_rows, dtype=torch.long, device=maps.device)
    return patches.contiguous(), meta, (c, patch_rows, patch_cols)


def build_report_from_array(
    x: np.ndarray,
    domain: str,
    horizon: int,
    variables: Optional[Iterable[str]] = None,
) -> str:
    arr = np.asarray(x, dtype=np.float32)
    flat = arr.reshape(-1, arr.shape[-1]) if arr.ndim > 1 else arr.reshape(-1, 1)
    trend = flat[-1] - flat[0] if len(flat) > 1 else np.zeros(flat.shape[-1], dtype=np.float32)
    volatility = np.nanstd(flat, axis=0)
    var_names = list(variables) if variables is not None else [f"x{i}" for i in range(flat.shape[-1])]
    trend_txt = ", ".join(f"{n}:{float(v):.4f}" for n, v in zip(var_names[:6], trend[:6]))
    vol_txt = ", ".join(f"{n}:{float(v):.4f}" for n, v in zip(var_names[:6], volatility[:6]))
    return (
        f"Task: forecast target variables. Domain: {domain}. "
        f"Variables: {', '.join(var_names[:12])}. "
        f"Historical summary: trend_delta={{ {trend_txt} }}; volatility={{ {vol_txt} }}. "
        f"Forecast request: predict next {int(horizon)} steps."
    )
