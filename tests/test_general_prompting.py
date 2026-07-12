from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import re
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import bstalignment.models as text_models
from bstalignment.graph_report_model import GraphReportTS, GraphReportTSConfig
from bstalignment.train_graph_report import evaluate

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

    def test_all_tied_high_dimensional_trends_select_twelve_distinct_canonical_variables(self):
        self.require_prompting()
        columns = tuple(f"tie_{index}" for index in range(21))

        prompt = build_general_prompt(np.zeros((36, 21), dtype=np.float32), columns, "1 hour", 96)
        names = re.findall(r"([\w-]+)\(last=", prompt)

        self.assertEqual(names, list(columns[:12]))
        self.assertEqual(len(names), len(set(names)))

    def test_result_metadata_is_stable_and_records_a_bounded_prompt(self):
        self.require_prompting()
        history = np.zeros((36, 7), dtype=np.float32)

        result = build_general_prompt_result(history, tuple(f"x{index}" for index in range(7)), "1 hour", 336)

        self.assertIn("pretoken_word_count", result.metadata)
        self.assertIn("pretoken_word_budget", result.metadata)
        self.assertIn("pretoken_word_truncated", result.metadata)
        self.assertEqual(result.metadata["pretoken_word_count"], len(result.prompt.split()))
        self.assertEqual(result.metadata["pretoken_word_budget"], 192)
        self.assertFalse(result.metadata["pretoken_word_truncated"])
        self.assertEqual(result.metadata["frequency"], "1 hour")
        self.assertEqual(result.metadata["variable_count"], 7)
        self.assertEqual(result.metadata["summary_count"], 7)

    def test_fallback_audit_truthfully_reports_when_the_192_word_budget_drops_summaries(self):
        self.require_prompting()
        columns = tuple(f"signal_{index} {'word ' * 200}".strip() for index in range(7))

        result = build_general_prompt_result(np.zeros((36, 7), dtype=np.float32), columns, "1 hour", 96)

        self.assertIn("pretoken_word_count", result.metadata)
        self.assertIn("pretoken_word_truncated", result.metadata)
        self.assertLessEqual(result.metadata["pretoken_word_count"], 192)
        self.assertTrue(result.metadata["pretoken_word_truncated"])
        self.assertEqual(result.metadata["summary_count"], 0)

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
        self.assertIn("pretoken_word_budget", original["prompt_metadata"])
        self.assertIn("pretoken_word_truncated", original["prompt_metadata"])
        self.assertEqual(original["prompt_metadata"]["pretoken_word_budget"], 192)
        self.assertFalse(original["prompt_metadata"]["pretoken_word_truncated"])


class EncoderPromptAuditTests(unittest.TestCase):
    def test_tokenizer_helper_reports_true_untruncated_count_and_limit(self):
        helper = getattr(text_models, "tokenizer_prompt_audit", None)
        self.assertIsNotNone(helper, "models must expose tokenizer_prompt_audit")

        class FakeTokenizer:
            def __call__(self, prompts, **_kwargs):
                return {"input_ids": [list(range(len(prompt.split()))) for prompt in prompts]}

        audit = helper(FakeTokenizer(), ["token " * 193], max_length=192)

        self.assertEqual(audit, [{"token_count": 193, "token_limit": 192, "truncated": True}])

    def test_simple_encoder_reports_its_actual_192_token_truncation(self):
        encoder = text_models.SimpleTextEncoder(d_model=4, max_length=192)
        try:
            encoder(["token " * 193], audit=True)
        except TypeError as error:
            self.fail(f"simple text encoder must support actual prompt audit: {error}")

        self.assertEqual(
            encoder.last_prompt_audit,
            [{"token_count": 193, "token_limit": 192, "truncated": True}],
        )

    def test_general_model_exposes_actual_encoder_audit_without_battery_output_field(self):
        general = GraphReportTS(
            GraphReportTSConfig(
                variant="general",
                output_dim=2,
                d_model=8,
                patch_size=4,
                patch_stride=4,
                graph_layers=1,
                dropout=0.0,
                use_hf_text_encoder=False,
                use_numeric_history=False,
            )
        ).eval()
        battery = GraphReportTS(
            GraphReportTSConfig(
                variant="battery",
                output_dim=1,
                d_model=8,
                patch_size=4,
                patch_stride=4,
                graph_layers=1,
                dropout=0.0,
                use_hf_text_encoder=False,
                use_numeric_history=False,
            )
        ).eval()

        general_output = general(
            torch.zeros(1, 2, 3, 4, 8),
            ["token " * 193],
            torch.tensor([2]),
            variable_mask=torch.ones(1, 2, dtype=torch.bool),
        )
        battery_output = battery(torch.zeros(1, 3, 4, 8), ["token " * 193], torch.tensor([2]))

        self.assertIn("prompt_audit", general_output)
        self.assertEqual(general_output["prompt_audit"][0]["token_count"], 193)
        self.assertTrue(general_output["prompt_audit"][0]["truncated"])
        self.assertNotIn("prompt_audit", battery_output)

    def test_evaluation_aggregates_general_encoder_audit_into_final_metrics(self):
        class AuditModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.cfg = SimpleNamespace(variant="general", use_relative_steps=True)

            def forward(self, maps, prompts, horizon, **_kwargs):
                return {
                    "pred": maps.new_zeros(maps.size(0), int(horizon.max()), maps.size(1)),
                    "prompt_audit": [
                        {"token_count": 100, "token_limit": 192, "truncated": False},
                        {"token_count": 193, "token_limit": 192, "truncated": True},
                    ],
                }

        batch = {
            "maps": torch.zeros(2, 1, 3, 4, 8),
            "prompt": ["one", "two"],
            "horizon": torch.tensor([2, 2]),
            "y": torch.zeros(2, 2, 1),
            "mask": torch.ones(2, 2, 1, dtype=torch.bool),
            "variable_mask": torch.ones(2, 1, dtype=torch.bool),
        }

        metrics = evaluate(AuditModel(), [batch], torch.device("cpu"), weights={"align": 0.0}, loss_type="mse")

        self.assertIn("encoder_token_count_mean", metrics)
        self.assertIn("encoder_truncated_count", metrics)
        self.assertIn("encoder_truncated_rate", metrics)
        self.assertIn("encoder_token_limit", metrics)
        self.assertEqual(metrics["encoder_token_count_mean"], 146.5)
        self.assertEqual(metrics["encoder_truncated_count"], 1.0)
        self.assertEqual(metrics["encoder_truncated_rate"], 0.5)
        self.assertEqual(metrics["encoder_token_limit"], 192.0)


if __name__ == "__main__":
    unittest.main()
