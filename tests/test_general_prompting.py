from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import re
import unittest

import numpy as np
import pandas as pd
import torch

try:
    from bstalignment.general_prompting import build_general_prompt, build_general_prompt_result
except ModuleNotFoundError:
    build_general_prompt = None
    build_general_prompt_result = None


class GeneralPromptingTests(unittest.TestCase):
    def require_prompting(self):
        self.assertIsNotNone(build_general_prompt, "general prompting must define build_general_prompt")
        self.assertIsNotNone(build_general_prompt_result, "general prompting must define build_general_prompt_result")

    def test_seven_variable_prompt_matches_the_stable_generic_template(self):
        self.require_prompting()
        history = np.tile(np.arange(7, dtype=np.float32), (36, 1))
        columns = tuple(f"x{index}" for index in range(7))

        prompt = build_general_prompt(history, columns, "15 minutes", 96)

        self.assertEqual(
            prompt,
            "Task: multivariate time-series forecasting.\n"
            "Observation: 36 past steps sampled every 15 minutes; 7 variables are observed.\n"
            "Window summary: aggregate mean=3.0000, standard deviation=2.0000, "
            "mean absolute change=0.0000, trend balance=0 increasing/0 decreasing/7 approximately flat.\n"
            "Variable summaries: x0(last=0.0000, trend=0.0000, volatility=0.0000); "
            "x1(last=1.0000, trend=0.0000, volatility=0.0000); "
            "x2(last=2.0000, trend=0.0000, volatility=0.0000); "
            "x3(last=3.0000, trend=0.0000, volatility=0.0000); "
            "x4(last=4.0000, trend=0.0000, volatility=0.0000); "
            "x5(last=5.0000, trend=0.0000, volatility=0.0000); "
            "x6(last=6.0000, trend=0.0000, volatility=0.0000).\n"
            "Instruction: predict all 7 variables for the next 96 steps.\n"
            "Use only the observed window.",
        )

    def test_twenty_one_variable_prompt_selects_twelve_deterministic_summaries(self):
        self.require_prompting()
        history = np.arange(36, dtype=np.float32)[:, None] * np.arange(21, dtype=np.float32)[None, :]
        columns = tuple(f"weather_{index}" for index in range(21))

        prompt = build_general_prompt(history, columns, "10 minutes", 192)
        names = re.findall(r"([\w-]+)\(last=", prompt)

        self.assertEqual(names, [*columns[:6], *columns[-6:]])
        self.assertIn("21 variables are observed.", prompt)
        self.assertIn("next 192 steps", prompt)

    def test_high_dimensional_selection_uses_absolute_trend_then_canonical_index(self):
        self.require_prompting()
        columns = tuple(f"load_{index}" for index in range(321))
        trends = np.arange(321, dtype=np.float32)
        history = np.arange(36, dtype=np.float32)[:, None] * trends[None, :]

        prompt = build_general_prompt(history, columns, "1 hour", 720)
        names = re.findall(r"([\w-]+)\(last=", prompt)

        self.assertEqual(names, [*(f"load_{index}" for index in range(6)), *(f"load_{index}" for index in range(315, 321))])
        self.assertEqual(len(names), 12)
        self.assertIn("321 variables are observed.", prompt)
        self.assertIn("next 720 steps", prompt)

    def test_result_metadata_is_stable_and_records_a_bounded_prompt(self):
        self.require_prompting()
        history = np.zeros((36, 7), dtype=np.float32)

        result = build_general_prompt_result(history, tuple(f"x{index}" for index in range(7)), "1 hour", 336)

        self.assertEqual(result.metadata["token_count"], len(result.prompt.split()))
        self.assertEqual(result.metadata["token_budget"], 256)
        self.assertFalse(result.metadata["truncated"])
        self.assertEqual(result.metadata["frequency"], "1 hour")
        self.assertEqual(result.metadata["variable_count"], 7)
        self.assertEqual(result.metadata["summary_count"], 7)

    def test_rejects_nonformal_history_or_horizon(self):
        self.require_prompting()
        history = np.zeros((35, 7), dtype=np.float32)
        columns = tuple(f"x{index}" for index in range(7))

        with self.assertRaisesRegex(ValueError, "36"):
            build_general_prompt(history, columns, "1 hour", 96)
        with self.assertRaisesRegex(ValueError, "formal"):
            build_general_prompt(np.zeros((36, 7), dtype=np.float32), columns, "1 hour", 95)

    def test_prompt_vocabulary_contains_no_battery_specific_terms(self):
        self.require_prompting()
        prompt = build_general_prompt(
            np.zeros((36, 7), dtype=np.float32),
            tuple(f"signal_{index}" for index in range(7)),
            "1 hour",
            96,
        ).lower()

        for prohibited in ("soh", "capacity", "cycle", "chemistry", "degradation", "future", "train", "validation", "test"):
            self.assertNotIn(prohibited, prompt)


class GeneralPromptDatasetIntegrationTests(unittest.TestCase):
    def test_general_collator_retains_optional_prompt_metadata_without_breaking_legacy_samples(self):
        from bstalignment.data_general import collate_general_graph_batch

        sample = {
            "maps": torch.zeros(1, 3, 4, 8),
            "y": torch.zeros(96, 1),
            "mask": torch.ones(96, 1, dtype=torch.bool),
            "horizon": torch.tensor(96),
            "prompt": "legacy general prompt",
            "series_id": "synthetic",
            "start_index": 0,
            "target_steps": torch.arange(36, 132),
        }

        try:
            batch = collate_general_graph_batch([sample])
        except KeyError as error:
            self.fail(f"general collator must accept legacy samples without prompt metadata: {error}")

        self.assertEqual(batch["prompt_metadata"], [None])

    def test_general_dataset_uses_only_standardized_history_and_exposes_prompt_audit_metadata(self):
        from bstalignment.data_general import GeneralForecastGraphDataset

        with TemporaryDirectory() as directory:
            root = Path(directory)
            data_dir = root / "processed" / "general" / "ECL"
            data_dir.mkdir(parents=True)
            values = np.arange(1_000, dtype=np.float32)
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2020-01-01", periods=len(values), freq="h"),
                    "load": values,
                    "temperature": values * 2,
                }
            )
            path = data_dir / "ECL.csv"
            frame.to_csv(path, index=False)
            original = GeneralForecastGraphDataset("ECL", data_root=str(root), split="val", pred_len=96, fit_scaler=True)[0]

            frame.loc[700:, ["load", "temperature"]] += 1_000_000
            frame.to_csv(path, index=False)
            changed_future = GeneralForecastGraphDataset("ECL", data_root=str(root), split="val", pred_len=96, fit_scaler=True)[0]

        self.assertEqual(original["prompt"], changed_future["prompt"])
        self.assertIn("prompt_metadata", original)
        self.assertIn("prompt_metadata", changed_future)
        self.assertEqual(original["prompt_metadata"], changed_future["prompt_metadata"])
        self.assertEqual(original["prompt_metadata"]["frequency"], "1 hour")
        self.assertEqual(original["prompt_metadata"]["token_budget"], 256)
        self.assertFalse(original["prompt_metadata"]["truncated"])


if __name__ == "__main__":
    unittest.main()
