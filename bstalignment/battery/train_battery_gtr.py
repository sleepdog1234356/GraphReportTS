"""Train the anchor-free BatteryGTR battery SOH model."""

from __future__ import annotations

import sys

from .battery_gtr import BATTERY_GTR_MODEL_NAME, BatteryGTR


DEFAULT_OUTPUT = "artifacts/battery/battery_gtr/runs"


def build_battery_argv(argv: list[str] | None = None) -> list[str]:
    """Return CLI arguments with the BatteryGTR output default injected."""

    command = list(sys.argv[1:] if argv is None else argv)
    if "--output" not in command:
        command.extend(["--output", DEFAULT_OUTPUT])
    return command


def main(argv: list[str] | None = None) -> None:
    from bstalignment.v2.train_battery import (
        BATTERY_V2_TRAINING_PROTOCOL,
        main as train_battery_v2,
    )

    train_battery_v2(
        build_battery_argv(argv),
        model_factory=BatteryGTR,
        model_name=BATTERY_GTR_MODEL_NAME,
        training_protocol=BATTERY_V2_TRAINING_PROTOCOL,
    )


if __name__ == "__main__":
    main()
