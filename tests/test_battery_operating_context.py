from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

import numpy as np

from anchoredgtr.core.battery_cache import (
    BatteryCellFeatures,
    BatteryFeatureCache,
    BatteryOperatingContext,
)
from anchoredgtr.core.battery_features import (
    BASE_FEATURE_NAMES,
    CURVE_POINTS,
    DV_NORMALIZED_Q_AXIS,
    IC_VOLTAGE_AXIS,
)
from anchoredgtr.core.battery_data import BatteryForecastDataset, BatterySplit
import anchoredgtr.core.precompute_battery as battery_precompute
from anchoredgtr.core.precompute_battery import (
    mit_operating_context,
    source_operating_contexts,
    xjtu_operating_context,
)
from anchoredgtr.core.prompts import build_battery_prompt, fit_battery_prompt_thresholds


XJTU_FILES = (
    "2C_battery-1.npz",
    "3C_battery-1.npz",
    "R2.5_battery-1.npz",
    "R3_battery-1.npz",
    "RW_battery-1.npz",
    "Sim_satellite_battery-1.npz",
)


def _prompt_features(scale: float = 1.0) -> np.ndarray:
    values = np.zeros((32, len(BASE_FEATURE_NAMES)), dtype=np.float32)
    index = {name: position for position, name in enumerate(BASE_FEATURE_NAMES)}
    progress = np.linspace(0.0, 1.0, 32, dtype=np.float32)
    values[:, index["cc_cv_ratio"]] = 2.0 + 0.1 * progress
    values[:, index["charge_duration_s"]] = 1000.0 + scale * 250.0 * progress
    values[:, index["discharge_duration_s"]] = 800.0 + scale * 200.0 * progress
    values[:, index["discharge_integral_ah"]] = 0.8 + scale * 0.15 * progress
    values[:, index["t_mean"]] = 25.0 + progress
    values[:, index["t_max"]] = 28.0 + 1.5 * progress
    values[:, index["current_squared_integral_a2s"]] = scale * (100.0 + progress)
    values[:, index["charge_tail_voltage_duration_s"]] = scale * (120.0 + progress)
    values[:, index["charge_tail_current_duration_s"]] = scale * (90.0 + progress)
    values[:, index["ic_primary_position"]] = 3.6 - 0.01 * progress
    values[:, index["dv_primary_position"]] = 0.5 + 0.01 * progress
    return values


def _cell(cell_id: str, context: BatteryOperatingContext | None) -> BatteryCellFeatures:
    observations = 84
    base = np.tile(_prompt_features(), (3, 1))[:observations]
    curve_mask = np.ones((observations, CURVE_POINTS), dtype=bool)
    return BatteryCellFeatures(
        cell_id=cell_id,
        observation_ids=np.arange(1, observations + 1, dtype=np.int64),
        time_coverage=np.ones(observations, dtype=np.float32),
        base_values=base,
        base_observed_mask=np.ones_like(base, dtype=bool),
        base_reliability=np.ones_like(base, dtype=np.float32),
        ic_curve=np.zeros((observations, CURVE_POINTS), dtype=np.float32),
        ic_curve_axis=np.tile(IC_VOLTAGE_AXIS, (observations, 1)),
        ic_curve_mask=curve_mask,
        ic_quality=np.ones(observations, dtype=np.float32),
        dv_curve=np.zeros((observations, CURVE_POINTS), dtype=np.float32),
        dv_curve_axis=np.tile(DV_NORMALIZED_Q_AXIS, (observations, 1)),
        dv_curve_mask=curve_mask.copy(),
        dv_quality=np.ones(observations, dtype=np.float32),
        soh_labels=np.linspace(1.0, 0.9, observations, dtype=np.float32),
        operating_context=context,
    )


class XJTUOperatingContextTests(unittest.TestCase):
    def test_six_protocols_share_one_cell_specification_and_are_distinct(self):
        contexts = [xjtu_operating_context(name) for name in XJTU_FILES]

        self.assertTrue(all(context is not None for context in contexts))
        self.assertEqual(
            {
                (
                    context.manufacturer,
                    context.form_factor,
                    context.chemistry,
                    context.nominal_capacity_ah,
                    context.nominal_voltage_v,
                    context.voltage_window_v,
                )
                for context in contexts
            },
            {("LISHEN", "18650", "NCM523", 2.0, 3.6, (2.5, 4.2))},
        )
        self.assertEqual(len({(context.charge_protocol, context.discharge_protocol) for context in contexts}), 6)
        self.assertEqual(
            [(context.charge_protocol, context.discharge_protocol) for context in contexts],
            [
                ("2C CC-CV to 4.2V", "1C constant-current to 2.5V"),
                ("3C CC-CV to 4.2V", "1C constant-current to 2.5V"),
                ("2C CC-CV to 4.2V", "0.5/1/2/3/5C cyclic discharge to 2.5V"),
                ("2C CC-CV to 4.2V", "0.5/1/2/3/5C cyclic discharge to 3.0V"),
                (
                    "staged 0.5/1/3C CC-CV to 4.2V",
                    "random-walk 2-8A for 2-6min; 3.0V safety cutoff",
                ),
                (
                    "2C CC-CV to 4.2V",
                    "0.667C GEO-shadow with scheduled variable duration and DOD below 80%",
                ),
            ],
        )

    def test_prompts_include_protocol_but_not_dataset_or_cell_identity(self):
        thresholds = fit_battery_prompt_thresholds([_prompt_features(0.8), _prompt_features(1.0), _prompt_features(1.2)])
        prompts = [
            build_battery_prompt(
                _prompt_features(),
                None,
                "sensor_only",
                operating_context=xjtu_operating_context(name),
                thresholds=thresholds,
            ).text
            for name in XJTU_FILES
        ]

        self.assertEqual(len(set(prompts)), 6)
        for text in prompts:
            self.assertIn("Cell: LISHEN 18650 NCM523 2Ah 3.6V 2.5-4.2V.", text)
            self.assertIn("Charge:", text)
            self.assertIn("Discharge:", text)
            lowered = text.lower()
            for forbidden in ("xjtu", "batch-", "battery-1", "cell_id", "cycle 84"):
                self.assertNotIn(forbidden, lowered)

    def test_unknown_context_falls_back_to_sensor_summary(self):
        prompt = build_battery_prompt(_prompt_features(), None, "sensor_only", operating_context=None)

        self.assertIn("Cell: unavailable.", prompt.text)
        self.assertIn("Charge: unavailable (", prompt.text)
        self.assertIn("Discharge: unavailable (", prompt.text)


class MITOperatingContextTests(unittest.TestCase):
    def test_mit_cell_spec_and_protocol_use_the_generic_schema(self):
        context = mit_operating_context("3.6C(80%)-3.6C")

        self.assertEqual(context.manufacturer, "A123")
        self.assertEqual(context.model, "APR18650M1A")
        self.assertEqual(context.chemistry, "LFP/graphite")
        self.assertEqual(context.form_factor, "18650")
        self.assertEqual(context.nominal_capacity_ah, 1.1)
        self.assertEqual(context.nominal_voltage_v, 3.3)
        self.assertEqual(context.voltage_window_v, (2.0, 3.6))
        self.assertEqual(context.charge_protocol, "3.6C(80%)-3.6C")
        self.assertEqual(context.discharge_protocol, "4C CC-CV to 2.0V with C/50 cutoff")

    def test_metadata_only_mit_reader_does_not_build_records_cycles_or_features(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = {
                "b1c0": {
                    "summary": {},
                    "charge_policy": "3.6C(80%)-3.6C",
                    "cycles": {"must_not_be_read": object()},
                }
            }
            with (root / "batch1.pkl").open("wb") as handle:
                pickle.dump(batch, handle)

            with (
                mock.patch.object(
                    battery_precompute,
                    "_load_mit_records_compatible",
                    side_effect=AssertionError("full records must not be constructed"),
                ),
                mock.patch.object(
                    battery_precompute,
                    "_summary_to_frame",
                    side_effect=AssertionError("summary conversion must not run"),
                ),
                mock.patch.object(
                    battery_precompute,
                    "_mit_raw_cycles",
                    side_effect=AssertionError("cycle conversion must not run"),
                ),
                mock.patch.object(
                    battery_precompute,
                    "build_cell_features",
                    side_effect=AssertionError("feature extraction must not run"),
                ),
            ):
                contexts = source_operating_contexts("mit", root)

        self.assertEqual(set(contexts), {"batch1_b1c0"})
        self.assertEqual(contexts["batch1_b1c0"].charge_protocol, "3.6C(80%)-3.6C")


class BatteryOperatingContextCacheTests(unittest.TestCase):
    def test_context_round_trips_and_manifest_can_be_enriched_without_rewriting_npz(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            cache = BatteryFeatureCache.create(
                root,
                {"cell-a": _cell("cell-a", None)},
                {"source_boundary": "test"},
            )
            cell_file = root / cache.manifest["cells"]["cell-a"]["file"]
            before = cell_file.read_bytes()
            context = xjtu_operating_context("2C_battery-1")

            enriched = cache.enrich_operating_context({"cell-a": context})

            self.assertEqual(cell_file.read_bytes(), before)
            self.assertEqual(enriched.load_cell("cell-a").operating_context, context)
            self.assertNotEqual(enriched.manifest_hash, cache.manifest_hash)

    def test_context_rejects_identity_fields(self):
        with self.assertRaisesRegex(ValueError, "unsupported operating-context fields"):
            BatteryOperatingContext.from_json({"dataset": "xjtu", "cell_id": "battery-1"})


class XJTUProtocolSplitTests(unittest.TestCase):
    @staticmethod
    def _cells(*, missing_context: bool = False, drop_last: bool = False):
        cells: dict[str, BatteryCellFeatures] = {}
        for group_index, filename in enumerate(XJTU_FILES):
            count = 15 if group_index == 1 else 8
            for cell_index in range(count):
                if drop_last and group_index == len(XJTU_FILES) - 1 and cell_index == count - 1:
                    continue
                cell_id = f"opaque-{group_index}-{cell_index:02d}"
                context = xjtu_operating_context(filename)
                if missing_context and group_index == 0 and cell_index == 0:
                    context = None
                cells[cell_id] = _cell(cell_id, context)
        return cells

    @staticmethod
    def _protocol_counts(cache: BatteryFeatureCache, ids: tuple[str, ...]):
        counts: dict[tuple[str, str], int] = {}
        for cell_id in ids:
            context = cache.load_cell(cell_id).operating_context
            assert context is not None
            key = (str(context.charge_protocol), str(context.discharge_protocol))
            counts[key] = counts.get(key, 0) + 1
        return sorted(counts.values())

    def test_split_is_deterministic_disjoint_and_covers_all_six_protocols(self):
        with TemporaryDirectory() as temporary:
            cache = BatteryFeatureCache.create(
                Path(temporary) / "xjtu",
                self._cells(),
                {"dataset": "xjtu", "source_boundary": "test"},
            )

            split = BatterySplit.from_cache(cache, seed=42)
            repeated = BatterySplit.from_cache(cache, seed=42)
            other_seed = BatterySplit.from_cache(cache, seed=43)

            self.assertEqual((len(split.train), len(split.val), len(split.test)), (38, 8, 9))
            self.assertEqual(self._protocol_counts(cache, split.train), [6, 6, 6, 6, 6, 8])
            self.assertEqual(self._protocol_counts(cache, split.val), [1, 1, 1, 1, 1, 3])
            self.assertEqual(self._protocol_counts(cache, split.test), [1, 1, 1, 1, 1, 4])
            self.assertEqual(split, repeated)
            self.assertNotEqual(split, other_seed)
            self.assertFalse(set(split.train) & set(split.val))
            self.assertFalse(set(split.train) & set(split.test))
            self.assertFalse(set(split.val) & set(split.test))
            self.assertEqual(set(split.train) | set(split.val) | set(split.test), set(cache.cell_ids))

    def test_invalid_xjtu_context_or_group_shape_fails_loud(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_context = BatteryFeatureCache.create(
                root / "missing",
                self._cells(missing_context=True),
                {"dataset": "xjtu", "source_boundary": "test"},
            )
            bad_shape = BatteryFeatureCache.create(
                root / "shape",
                self._cells(drop_last=True),
                {"dataset": "xjtu", "source_boundary": "test"},
            )

            with self.assertRaisesRegex(ValueError, "requires operating context"):
                BatterySplit.from_cache(missing_context, seed=42)
            with self.assertRaisesRegex(ValueError, "six canonical protocol groups"):
                BatterySplit.from_cache(bad_shape, seed=42)

    def test_non_xjtu_cache_keeps_legacy_cell_random_split(self):
        with TemporaryDirectory() as temporary:
            cells = {cell_id: _cell(cell_id, None) for cell_id in ("a", "b", "c", "d")}
            cache = BatteryFeatureCache.create(
                Path(temporary) / "mit",
                cells,
                {"dataset": "mit", "source_boundary": "test"},
            )

            self.assertEqual(
                BatterySplit.from_cache(cache, seed=42),
                BatterySplit.from_cell_ids(cache.cell_ids, seed=42),
            )


class BatteryPromptThresholdTests(unittest.TestCase):
    def test_thresholds_depend_only_on_supplied_training_windows(self):
        training = [_prompt_features(0.7), _prompt_features(1.0), _prompt_features(1.3)]
        expected = fit_battery_prompt_thresholds(training)

        validation_or_future = _prompt_features(50.0)
        actual = fit_battery_prompt_thresholds(training)

        self.assertEqual(actual, expected)
        self.assertNotEqual(actual, fit_battery_prompt_thresholds(training + [validation_or_future]))

    def test_dataset_thresholds_ignore_validation_and_test_cells(self):
        context = xjtu_operating_context("2C_battery-1")
        ids = ("cell-a", "cell-b", "cell-c")
        split = BatterySplit.from_cell_ids(ids, seed=42)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = {cell_id: _cell(cell_id, context) for cell_id in ids}
            changed = dict(original)
            feature_index = {name: position for position, name in enumerate(BASE_FEATURE_NAMES)}
            for cell_id in (*split.val, *split.test):
                cell = changed[cell_id]
                values = cell.base_values.copy()
                alternating = np.where(np.arange(len(values)) % 2 == 0, 1.0, 10000.0)
                values[:, feature_index["charge_duration_s"]] = alternating
                values[:, feature_index["discharge_integral_ah"]] = alternating
                changed[cell_id] = replace(cell, base_values=values)
            cache_a = BatteryFeatureCache.create(root / "a", original, {"source_boundary": "test"})
            cache_b = BatteryFeatureCache.create(root / "b", changed, {"source_boundary": "test"})

            dataset_a = BatteryForecastDataset(cache_a, split="train", seed=42)
            dataset_b = BatteryForecastDataset(cache_b, split="train", seed=42)

            self.assertEqual(dataset_a.prompt_thresholds, dataset_b.prompt_thresholds)
            metadata = dataset_a[0]["metadata"]
            self.assertEqual(metadata["dataset_split"], "train")
            self.assertEqual(metadata["sample_index"], 0)
            self.assertEqual(metadata["window_key"], dataset_a.sample_keys[0])
            repeated = BatteryForecastDataset(cache_a, split="train", seed=42)
            self.assertEqual(metadata["window_key"], repeated.sample_keys[0])

    def test_recent_graph_cycles_and_future_targets_do_not_change_prompt(self):
        context = xjtu_operating_context("RW_battery-1")
        ids = ("cell-a", "cell-b", "cell-c")
        thresholds = {
            "charge_duration_variability": (0.05, 0.20),
            "discharge_current_variability": (0.05, 0.20),
        }
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = {cell_id: _cell(cell_id, context) for cell_id in ids}
            changed: dict[str, BatteryCellFeatures] = {}
            for cell_id, cell in original.items():
                values = cell.base_values.copy()
                labels = cell.soh_labels.copy()
                values[32:, :] = 9999.0
                labels[64:] = 0.25
                changed[cell_id] = replace(cell, base_values=values, soh_labels=labels)
            cache_a = BatteryFeatureCache.create(root / "a", original, {"source_boundary": "test"})
            cache_b = BatteryFeatureCache.create(root / "b", changed, {"source_boundary": "test"})

            dataset_a = BatteryForecastDataset(
                cache_a,
                split="train",
                seed=42,
                prompt_thresholds=thresholds,
                max_samples=1,
            )
            dataset_b = BatteryForecastDataset(
                cache_b,
                split="train",
                seed=42,
                prompt_thresholds=thresholds,
                max_samples=1,
            )

            self.assertEqual(dataset_a[0]["prompt"], dataset_b[0]["prompt"])


if __name__ == "__main__":
    unittest.main()
