from __future__ import annotations

import unittest

import numpy as np

from anchoredgtr.core.general_data import GeneralForecastDataset
from anchoredgtr.core.prompts import (
    GENERAL_PROMPT_METRICS,
    build_general_prompt,
    fit_general_prompt_thresholds,
)


class GeneralPromptTests(unittest.TestCase):
    @staticmethod
    def thresholds() -> dict[str, tuple[float, float, float, float]]:
        return {name: (0.1, 0.2, 0.4, 0.8) for name in GENERAL_PROMPT_METRICS}

    def test_zero_missingness_is_not_described_as_high(self) -> None:
        context = np.zeros((36, 7), dtype=np.float32)
        prompt = build_general_prompt(context, [f"v{i}" for i in range(7)], "1 hour", 24, self.thresholds())
        self.assertNotIn("missing", prompt.text.lower())
        self.assertNotIn("outliers", prompt.text.lower())

    def test_different_histories_produce_different_compact_prompts(self) -> None:
        time = np.linspace(-1.0, 1.0, 36, dtype=np.float32)[:, None]
        rising = np.repeat(time, 7, axis=1)
        falling = -rising
        first = build_general_prompt(rising, [f"v{i}" for i in range(7)], "1 hour", 24, self.thresholds())
        second = build_general_prompt(falling, [f"v{i}" for i in range(7)], "1 hour", 24, self.thresholds())
        self.assertNotEqual(first.text, second.text)
        self.assertIn("Trend: up", first.text)
        self.assertIn("Trend: down", second.text)
        self.assertLess(len(first.text.split()), 100)

    def test_prompt_statistics_use_only_supplied_context(self) -> None:
        rng = np.random.default_rng(7)
        context = rng.normal(size=(36, 21)).astype(np.float32)
        first = build_general_prompt(context, [f"v{i}" for i in range(21)], "10 minutes", 24, self.thresholds())
        changed = context.copy()
        changed[-1] += 4.0
        second = build_general_prompt(changed, [f"v{i}" for i in range(21)], "10 minutes", 24, self.thresholds())
        self.assertNotEqual(first.text, second.text)
        self.assertEqual(first.metadata["context_length"], 36)

    def test_training_threshold_fit_tracks_the_gtr_metric_schema(self) -> None:
        rng = np.random.default_rng(11)
        values = rng.normal(size=(180, 7)).astype(np.float32)
        thresholds = fit_general_prompt_thresholds(values, [36, 72, 108, 144])
        self.assertEqual(set(thresholds), set(GENERAL_PROMPT_METRICS))

    def test_dataset_item_reuses_prompt_result_for_text_and_metadata(self) -> None:
        dataset = GeneralForecastDataset.__new__(GeneralForecastDataset)
        dataset.samples = [36]
        dataset.raw_values = np.arange(96 * 7, dtype=np.float32).reshape(96, 7)
        dataset.values = dataset.raw_values.copy()
        dataset.columns = [f"v{i}" for i in range(7)]
        dataset.frequency = "1 hour"
        dataset.pred_len = 24
        dataset.prompt_thresholds = self.thresholds()
        dataset.cache_prompts = True
        dataset._prompt_cache = {}
        dataset.dataset_name = "unit-test"

        item = dataset[0]
        self.assertEqual(item["prompt"], dataset.prompt_at(0))
        self.assertEqual(item["metadata"]["prompt_metadata"]["context_length"], 36)
        self.assertEqual(item["metadata"]["target_start"], 72)


if __name__ == "__main__":
    unittest.main()
