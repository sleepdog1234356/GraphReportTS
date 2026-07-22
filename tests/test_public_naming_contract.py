from __future__ import annotations

from pathlib import Path

from anchoredgtr.battery import BatteryGTR
from anchoredgtr.core.contracts import GTRConfig
from anchoredgtr.general import AnchoredGTR


_FORBIDDEN_TEXT = (
    "bst" + "alignment",
    "graph" + "reportts",
    "graph" + "_report_ts",
    "graph" + "-report-ts",
    "graph" + "_" + "report" + "_" + "v" + "2",
)
_TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".sh"}
_IGNORED_PARTS = {
    ".git",
    ".codex_tmp",
    ".pytest_cache",
    ".vendor",
    "__pycache__",
    "artifacts",
    "data",
    "experiments",
    "external",
    "hf_models",
    "results_archive",
    "runs",
    "superpowers",
}


def test_final_public_imports() -> None:
    assert AnchoredGTR.__name__ == "AnchoredGTR"
    assert BatteryGTR.__name__ == "BatteryGTR"
    assert GTRConfig.__name__ == "GTRConfig"


def test_public_tree_has_no_temporary_main_identity() -> None:
    root = Path(__file__).resolve().parents[1]
    offenders: set[str] = set()
    for path in root.rglob("*"):
        if any(part in _IGNORED_PARTS for part in path.parts):
            continue
        relative = path.relative_to(root).as_posix().lower()
        if any(token in relative for token in _FORBIDDEN_TEXT):
            offenders.add(relative)
        if path.is_file() and path.suffix.lower() in _TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8").lower()
            if any(token in text for token in _FORBIDDEN_TEXT):
                offenders.add(relative)
    assert sorted(offenders) == []
