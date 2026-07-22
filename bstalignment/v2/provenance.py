from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

import torch


PROVENANCE_SCHEMA = "graph-report-ts-v2-provenance-v1"
DEFAULT_EXCLUDES = {
    ".git",
    ".migration-transfer",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    "data",
    "datasets",
    "checkpoints",
    "hf_models",
    "runs",
    "external",
}


def _git_value(root: Path, *arguments: str) -> str | None:
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), *arguments],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def iter_tree_files(root: str | Path, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> Iterable[Path]:
    base = Path(root).resolve()
    excluded = set(excludes)
    for directory, names, filenames in os.walk(base):
        names[:] = sorted(name for name in names if name not in excluded)
        relative_directory = Path(directory).relative_to(base)
        for filename in sorted(filenames):
            if filename in excluded or filename.endswith((".pyc", ".pyo")):
                continue
            yield relative_directory / filename


def tree_digest(root: str | Path, excludes: Iterable[str] = DEFAULT_EXCLUDES) -> tuple[str, int, int]:
    base = Path(root).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"source tree not found: {base}")
    digest = hashlib.sha256()
    file_count = 0
    byte_count = 0
    for relative in iter_tree_files(base, excludes):
        path = base / relative
        size = path.stat().st_size
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        file_count += 1
        byte_count += size
    return digest.hexdigest(), file_count, byte_count


def source_snapshot(root: str | Path, name: str | None = None) -> dict[str, Any]:
    path = Path(root).resolve()
    digest, files, size = tree_digest(path)
    commit = _git_value(path, "rev-parse", "HEAD")
    status = _git_value(path, "status", "--porcelain", "--untracked-files=no")
    return {
        "name": name or path.name,
        "path": str(path),
        "identity_kind": "git_commit_and_tree" if commit else "tree_snapshot",
        "git_commit": commit,
        "tracked_dirty": bool(status) if status is not None else None,
        "tree_sha256": digest,
        "file_count": files,
        "byte_count": size,
    }


def hardware_snapshot() -> dict[str, Any]:
    memory = None
    if Path("/proc/meminfo").exists():
        values: dict[str, int] = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
        memory = {key: values.get(key) for key in ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")}
    gpus = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            free, total = torch.cuda.mem_get_info(index)
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_memory": int(total),
                    "free_memory": int(free),
                    "capability": f"{props.major}.{props.minor}",
                    "bf16": bool(torch.cuda.is_bf16_supported()),
                }
            )
    disk = shutil.disk_usage(Path.cwd())
    return {
        "schema": PROVENANCE_SCHEMA,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "working_directory_disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "gpus": gpus,
    }


def write_provenance(
    output: str | Path,
    *,
    project_root: str | Path,
    external_root: str | Path | None = None,
) -> Path:
    project = Path(project_root).resolve()
    sources = []
    if external_root is not None:
        external = Path(external_root).resolve()
        if external.exists():
            for child in sorted(external.iterdir()):
                if child.is_dir():
                    sources.append(source_snapshot(child))
    payload = {
        "schema": PROVENANCE_SCHEMA,
        "project": source_snapshot(project, "GraphReportTS"),
        "external_sources": sources,
        "hardware": hardware_snapshot(),
    }
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record GraphReportTS-v2 source and hardware provenance")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--external_root", default="external")
    parser.add_argument("--output", default="runs/graph_report_ts_v2/provenance.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    path = write_provenance(args.output, project_root=args.project_root, external_root=args.external_root)
    print(path)


if __name__ == "__main__":
    main()
