from __future__ import annotations

from pathlib import Path

import pytest
import torch

from bstalignment.general.anchored_gtr import (
    ANCHORED_GTR_MODEL_NAME,
    AnchoredGTR,
    canonical_general_model_name,
)
from bstalignment.general.strategy_registry import DATASETS, HORIZONS, STRATEGY_REGISTRY, resolve_strategy
from bstalignment.general.train_anchored_gtr import build_general_argv, parse_args
from bstalignment.v2.contracts import GraphReportTSv2Config
from bstalignment.v2.heads import FixedLogitGate
from bstalignment.v2.train_battery import parse_args as parse_battery_args
from bstalignment.v2.train_general import _dataset_identity_matches


def test_name_mapping_only_changes_legacy_general_prefix() -> None:
    assert canonical_general_model_name("GraphReportTS-v2-DRF") == "AnchoredGTR"
    assert canonical_general_model_name("GraphReportTS-v2-DRF-FixedCorrection") == (
        "AnchoredGTR-FixedCorrection"
    )
    assert canonical_general_model_name("GraphReportTS-v2-BatteryDRF") == (
        "GraphReportTS-v2-BatteryDRF"
    )
    assert ANCHORED_GTR_MODEL_NAME == "AnchoredGTR"


def test_registry_covers_exact_l36_matrix() -> None:
    assert set(STRATEGY_REGISTRY) == {(dataset, horizon) for dataset in DATASETS for horizon in HORIZONS}
    assert resolve_strategy("ETTh2", 24).correction_gate_mode == "fixed_one"
    assert resolve_strategy("ETTm2", 24).freeze_linear_anchor is True
    assert resolve_strategy("Weather", 36).seed == 43
    assert resolve_strategy("Weather", 60).name == "weather_validation_calibrated_a1"
    with pytest.raises(ValueError):
        resolve_strategy("ECL", 96)


def test_anchored_gtr_requires_general_decomposition_encoder() -> None:
    model = AnchoredGTR(
        GraphReportTSv2Config(
            domain="general",
            input_len=36,
            pred_len=24,
            graph_embedding_variant="series_context_decomp",
            text_backend="simple",
            correction_gate_mode="fixed_one",
        )
    )
    assert isinstance(model.head.correction_gate, FixedLogitGate)
    gate = torch.sigmoid(model.head.correction_gate(torch.zeros(2, 3, 4)))
    assert torch.equal(gate, torch.ones_like(gate))
    with pytest.raises(ValueError, match="series_context_decomp"):
        AnchoredGTR(
            GraphReportTSv2Config(
                domain="general",
                input_len=36,
                pred_len=24,
                graph_embedding_variant="patch",
                text_backend="simple",
            )
        )


def test_main_cli_builds_canonical_identity_and_relative_roots(tmp_path: Path) -> None:
    provenance = tmp_path / "provenance.json"
    args = parse_args(
        [
            "--dataset",
            "ETTh2",
            "--horizon",
            "24",
            "--mode",
            "preflight",
            "--data-root",
            str(tmp_path / "data"),
            "--output-root",
            str(tmp_path / "artifacts"),
            "--text-model",
            str(tmp_path / "distilbert"),
            "--text-cache-root",
            str(tmp_path / "cache"),
            "--max-train-batches",
            "1",
            "--max-eval-batches",
            "1",
        ]
    )
    command = build_general_argv(args)
    assert command[command.index("--model_name") + 1] == "AnchoredGTR"
    assert command[command.index("--graph_embedding_variant") + 1] == "series_context_decomp"
    assert command[command.index("--correction_gate_mode") + 1] == "fixed_one"
    assert "--freeze_linear_anchor" not in command
    assert "/root/autodl-tmp/GraphReportTS" not in " ".join(command)


def test_main_defaults_use_organized_data_and_non_overwriting_run_roots() -> None:
    root = Path(__file__).resolve().parents[1]
    general = parse_args(["--dataset", "ETTh1", "--horizon", "24", "--mode", "preflight"])
    battery = parse_battery_args(["--dataset", "mit", "--cache_dir", "cache"])

    assert Path(general.data_root) == root / "data" / "general"
    assert Path(general.output_root) == root / "artifacts" / "general" / "anchored_gtr" / "runs"
    assert Path(battery.output) == Path("artifacts/battery/graphreportts_v2/runs")


def test_project_launchers_resolve_root_from_their_location() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in (
        Path("projects/general/anchored_gtr/run_matrix.sh"),
        Path("projects/battery/battery_gtr/run_matrix.sh"),
    ):
        text = (root / relative).read_text(encoding="utf-8")
        assert "BASH_SOURCE[0]" in text
        assert "/root/autodl-tmp/GraphReportTS" not in text


def test_dataset_identity_allows_only_a_relocated_equal_csv() -> None:
    original = {
        "name": "ECL",
        "source_csv": {"path": "/old/electricity.csv", "sha256": "a" * 64},
        "row_count": 100,
        "variable_count": 321,
        "columns_sha256": "b" * 64,
    }
    relocated = {
        **original,
        "source_csv": {"path": "/new/electricity.csv", "sha256": "a" * 64},
    }
    changed = {
        **relocated,
        "source_csv": {"path": "/new/electricity.csv", "sha256": "c" * 64},
    }
    assert _dataset_identity_matches(original, relocated)
    assert not _dataset_identity_matches(original, changed)
