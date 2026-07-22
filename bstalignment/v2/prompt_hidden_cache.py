from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import torch

from .semantic import FrozenDistilBERTEncoder


PROMPT_HIDDEN_CACHE_SCHEMA = "graph-report-ts-v2-general-prompt-hidden-v1"
PROMPT_HIDDEN_CHUNK_SIZE = 2_048
BATTERY_PROMPT_HIDDEN_CACHE_ROOT = "data/battery/cache/distilbert_hidden"


def battery_prompt_hidden_dataset_name(dataset: str) -> str:
    """Return the shared formal battery prompt-cache namespace."""

    normalized = str(dataset).strip().lower()
    if normalized not in {"mit", "xjtu"}:
        raise ValueError(f"unsupported battery prompt cache dataset: {dataset}")
    return f"battery-shared-{normalized}"


def _stable_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return normalized or "dataset"


def local_text_model_identity(model_path: str | Path) -> dict[str, object]:
    """Fingerprint local tokenizer/config/weight files used by DistilBERT."""

    root = Path(model_path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"local text model directory does not exist: {root}")
    fixed_candidates = (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    candidates = sorted(
        set(fixed_candidates)
        | {path.name for path in root.glob("*.safetensors")}
        | {path.name for path in root.glob("pytorch_model*.bin")}
    )
    files: list[dict[str, object]] = []
    for name in candidates:
        path = root / name
        if path.is_file():
            files.append(
                {
                    "name": name,
                    "size": path.stat().st_size,
                    "sha256": _file_sha256(path),
                }
            )
    if not any(
        str(item["name"]).endswith((".safetensors", ".bin"))
        or str(item["name"]).endswith(("safetensors.index.json", "bin.index.json"))
        for item in files
    ):
        raise FileNotFoundError(f"local text model has no recognized weight file: {root}")
    revision = None
    config_path = root / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        revision = config.get("_commit_hash") or config.get("revision")
    payload: dict[str, object] = {
        "path": str(root),
        "revision": revision,
        "files": files,
    }
    payload["sha256"] = _stable_digest(payload)
    return payload


def prompt_hidden_base_identity(
    *,
    model_path: str | Path,
    text_max_length: int,
    prompt_schema: Mapping[str, object],
    dataset_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": PROMPT_HIDDEN_CACHE_SCHEMA,
        "model": local_text_model_identity(model_path),
        "text_max_length": int(text_max_length),
        "prompt_schema": dict(prompt_schema),
        "dataset_identity": dict(dataset_identity),
        "storage_dtype": "bfloat16",
        "backbone_compute_dtype": "bfloat16_cuda_or_float32_cpu",
        "cache_boundary": "backbone.last_hidden_state+attention_mask",
    }


def _unique_prompts(prompts: Sequence[str]) -> tuple[list[str], list[str]]:
    unique: dict[str, str] = {}
    sequence_hashes: list[str] = []
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        previous = unique.get(digest)
        if previous is not None and previous != prompt:  # pragma: no cover - cryptographic invariant
            raise RuntimeError("SHA-256 collision between two prompt texts")
        unique.setdefault(digest, prompt)
        sequence_hashes.append(digest)
    return list(unique.values()), sequence_hashes


def _split_identity(
    base_identity: Mapping[str, object],
    *,
    split: str,
    prompts: Sequence[str],
) -> tuple[dict[str, object], list[str]]:
    unique, sequence_hashes = _unique_prompts(prompts)
    identity = {
        **dict(base_identity),
        "split": split,
        "prompt_count": len(prompts),
        "unique_prompt_count": len(unique),
        "prompt_hashes_sha256": _stable_digest(sequence_hashes),
        "unique_prompt_hashes": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in unique],
    }
    return identity, unique


def _load_manifest_if_valid(
    manifest_path: Path,
    expected_identity: Mapping[str, object],
) -> dict[str, object] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PROMPT_HIDDEN_CACHE_SCHEMA:
            return None
        if manifest.get("identity") != expected_identity:
            return None
        chunks = manifest.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return None
        for item in chunks:
            if not isinstance(item, dict):
                return None
            path = manifest_path.parent / str(item["file"])
            if not path.is_file() or _file_sha256(path) != item.get("sha256"):
                return None
        return manifest
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def ensure_prompt_hidden_split_cache(
    encoder: FrozenDistilBERTEncoder,
    prompts: Sequence[str],
    *,
    split: str,
    cache_root: str | Path,
    dataset_name: str,
    horizon: int,
    base_identity: Mapping[str, object],
    precompute_batch_size: int,
    chunk_size: int = PROMPT_HIDDEN_CHUNK_SIZE,
) -> dict[str, object]:
    """Validate or atomically rebuild one split's persistent hidden cache."""

    if split not in ("train", "val", "test"):
        raise ValueError("prompt hidden-cache split must be train, val, or test")
    if precompute_batch_size < 1 or chunk_size < 1:
        raise ValueError("prompt hidden-cache batch and chunk sizes must be positive")
    identity, unique_prompts = _split_identity(base_identity, split=split, prompts=prompts)
    if not unique_prompts:
        raise ValueError(f"cannot precompute an empty {split} prompt cache")
    split_dir = Path(cache_root).expanduser().resolve() / _slug(dataset_name) / f"H{int(horizon)}" / split
    manifest_path = split_dir / "manifest.json"
    valid = _load_manifest_if_valid(manifest_path, identity)
    if valid is not None:
        return {**valid, "manifest_path": str(manifest_path), "cache_status": "reused"}

    generation = _stable_digest(identity)[:16]
    chunk_records: list[dict[str, object]] = []
    pending: list[dict[str, object]] = []
    chunk_index = 0
    for offset in range(0, len(unique_prompts), precompute_batch_size):
        batch_prompts = unique_prompts[offset : offset + precompute_batch_size]
        states = encoder.encode_hidden_states(batch_prompts)
        for prompt, state in zip(batch_prompts, states):
            pending.append(
                {
                    "prompt": prompt,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "hidden": state.hidden.to(device="cpu", dtype=torch.bfloat16).contiguous(),
                    "mask": state.mask.to(device="cpu", dtype=torch.bool).contiguous(),
                    "token_count": state.token_count,
                }
            )
            if len(pending) >= chunk_size:
                name = f"chunk-{generation}-{chunk_index:05d}.pt"
                path = split_dir / name
                _atomic_torch_save(
                    path,
                    {
                        "schema": PROMPT_HIDDEN_CACHE_SCHEMA,
                        "identity_sha256": _stable_digest(identity),
                        "entries": pending,
                    },
                )
                chunk_records.append(
                    {"file": name, "entries": len(pending), "sha256": _file_sha256(path)}
                )
                pending = []
                chunk_index += 1
    if pending:
        name = f"chunk-{generation}-{chunk_index:05d}.pt"
        path = split_dir / name
        _atomic_torch_save(
            path,
            {
                "schema": PROMPT_HIDDEN_CACHE_SCHEMA,
                "identity_sha256": _stable_digest(identity),
                "entries": pending,
            },
        )
        chunk_records.append({"file": name, "entries": len(pending), "sha256": _file_sha256(path)})

    manifest: dict[str, object] = {
        "schema": PROMPT_HIDDEN_CACHE_SCHEMA,
        "identity": identity,
        "identity_sha256": _stable_digest(identity),
        "chunks": chunk_records,
    }
    _atomic_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path), "cache_status": "rebuilt"}


def _read_cache_entries(
    manifest: Mapping[str, object],
) -> Iterable[tuple[str, torch.Tensor, torch.Tensor, int]]:
    manifest_path = Path(str(manifest["manifest_path"]))
    identity_sha256 = str(manifest["identity_sha256"])
    for item in manifest["chunks"]:
        path = manifest_path.parent / str(item["file"])
        if not path.is_file() or _file_sha256(path) != item["sha256"]:
            raise RuntimeError(f"prompt hidden-cache chunk is missing or corrupt: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("schema") != PROMPT_HIDDEN_CACHE_SCHEMA
            or payload.get("identity_sha256") != identity_sha256
        ):
            raise RuntimeError(f"prompt hidden-cache chunk provenance mismatch: {path}")
        for entry in payload.get("entries", []):
            prompt = str(entry["prompt"])
            if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != entry["prompt_sha256"]:
                raise RuntimeError(f"prompt hidden-cache entry hash mismatch: {path}")
            hidden = entry["hidden"]
            mask = entry["mask"]
            token_count = int(entry["token_count"])
            if hidden.dtype != torch.bfloat16 or hidden.device.type != "cpu":
                raise RuntimeError(f"prompt hidden-cache state is not CPU BF16: {path}")
            if mask.dtype != torch.bool or mask.device.type != "cpu":
                raise RuntimeError(f"prompt hidden-cache mask is not CPU bool: {path}")
            if token_count < int(mask.sum()):
                raise RuntimeError(f"prompt hidden-cache token count is invalid: {path}")
            yield prompt, hidden, mask, token_count


def install_prompt_hidden_caches(
    encoder: FrozenDistilBERTEncoder,
    manifests: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    """Load train/validation/test cache entries and require complete RAM coverage."""

    # Precomputation itself may have populated the LRU. Release those entries
    # before reading the authoritative disk chunks to avoid a 2x CPU-RAM peak.
    encoder.clear_hidden_cache()
    entries: list[tuple[str, torch.Tensor, torch.Tensor, int]] = []
    for manifest in manifests:
        entries.extend(_read_cache_entries(manifest))
    return encoder.install_hidden_states(entries, replace=True, require_all=True)
