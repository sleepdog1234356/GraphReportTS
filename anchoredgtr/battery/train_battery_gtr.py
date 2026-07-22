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
    from anchoredgtr.core.train_battery import (
        BATTERY_GTR_TRAINING_PROTOCOL,
        main as train_battery_gtr,
    )

    train_battery_gtr(
        build_battery_argv(argv),
        model_factory=BatteryGTR,
        model_name=BATTERY_GTR_MODEL_NAME,
        training_protocol=BATTERY_GTR_TRAINING_PROTOCOL,
    )


if __name__ == "__main__":
    main()
