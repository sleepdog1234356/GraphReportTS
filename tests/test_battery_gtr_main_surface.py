from __future__ import annotations

from pathlib import Path

from anchoredgtr.battery.battery_gtr import (
    BATTERY_GTR_MODEL_NAME,
    BatteryGTR,
)
from anchoredgtr.battery.train_battery_gtr import build_battery_argv
from anchoredgtr.core.contracts import GTRConfig
from anchoredgtr.core.heads import BatterySOHHead
from anchoredgtr.core.model import BatteryGTRCore


def battery_config() -> GTRConfig:
    return GTRConfig(
        domain="battery",
        input_len=32,
        pred_len=20,
        text_backend="simple",
        graph_embedding_variant="patch",
    )


def test_battery_gtr_uses_final_identity() -> None:
    assert BATTERY_GTR_MODEL_NAME == "BatteryGTR"


def test_battery_gtr_is_checkpoint_compatible_and_anchor_free() -> None:
    legacy = BatteryGTRCore(battery_config())
    model = BatteryGTR(battery_config())
    incompatible = model.load_state_dict(legacy.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    assert isinstance(model.shared.head, BatterySOHHead)
    assert not any("linear_anchor" in name for name, _ in model.named_parameters())


def test_battery_gtr_rejects_non_patch_graphs() -> None:
    config = GTRConfig(
        domain="battery",
        input_len=32,
        pred_len=20,
        text_backend="simple",
        graph_embedding_variant="series_context_decomp",
    )
    try:
        BatteryGTR(config)
    except ValueError as error:
        assert "graph_embedding_variant='patch'" in str(error)
    else:
        raise AssertionError("BatteryGTR accepted a non-patch graph")


def test_battery_gtr_cli_injects_new_default_output() -> None:
    argv = build_battery_argv(["--cache_dir", "cache", "--disable_text"])
    assert argv[argv.index("--output") + 1] == "artifacts/battery/battery_gtr/runs"
    explicit = build_battery_argv(
        ["--cache_dir", "cache", "--disable_text", "--output", "custom"]
    )
    assert explicit[explicit.index("--output") + 1] == "custom"


def test_battery_gtr_launcher_is_portable() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = root / "projects" / "battery" / "battery_gtr" / "run_matrix.sh"
    text = launcher.read_text(encoding="utf-8")
    assert "anchoredgtr.battery.train_battery_gtr" in text
    assert "artifacts/battery/battery_gtr/runs" in text
    assert "BASH_SOURCE[0]" in text
    assert "/root/autodl-tmp/AnchoredGTR" not in text
