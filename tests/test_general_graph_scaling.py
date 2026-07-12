from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np
import torch

import bstalignment.graph_report_model as graph_model
import bstalignment.infer_graph_report as graph_inference
import bstalignment.raw_signal as raw_signal
from bstalignment.data_general import collate_general_graph_batch


def _sample(variable_count: int, horizon: int = 24) -> dict[str, object]:
    return {
        "maps": torch.arange(18 * 4 * 8, dtype=torch.float32).reshape(18, 4, 8),
        "y": torch.zeros(horizon, variable_count),
        "mask": torch.ones(horizon, variable_count, dtype=torch.bool),
        "horizon": torch.tensor(horizon),
        "prompt": "forecast all variables",
        "series_id": "synthetic",
        "start_index": 0,
        "target_steps": torch.arange(36, 36 + horizon),
        "history_scaled": torch.arange(36 * variable_count, dtype=torch.float32).reshape(36, variable_count),
    }


def _general_model(use_prompt: bool = False) -> graph_model.GraphReportTS:
    return graph_model.GraphReportTS(
        graph_model.GraphReportTSConfig(
            variant="general",
            d_model=8,
            max_steps=60,
            patch_size=4,
            patch_stride=4,
            graph_layers=1,
            topk_edges=2,
            dropout=0.0,
            use_report_prompt=use_prompt,
            use_cross_modal_fusion=use_prompt,
            use_hf_text_encoder=False,
        )
    ).eval()


class GeneralVariableMapTests(unittest.TestCase):
    def test_variable_axis_aggregation_is_fixed_and_exact(self):
        maps = np.arange(3 * 2 * 2 * 2, dtype=np.float32).reshape(3, 2, 2, 2)

        actual = raw_signal.aggregate_variable_maps(maps)

        expected = np.concatenate(
            (
                maps.mean(axis=0),
                maps.std(axis=0),
                maps.min(axis=0),
                maps.max(axis=0),
                np.quantile(maps, 0.25, axis=0),
                np.quantile(maps, 0.75, axis=0),
            ),
            axis=0,
        ).astype(np.float32)
        self.assertEqual(actual.shape, (12, 2, 2))
        np.testing.assert_allclose(actual, expected)

    def test_fixed_graph_shape_is_independent_of_variable_count(self):
        shapes = [
            raw_signal.aggregate_variable_maps(np.zeros((count, 4, 8, 29), dtype=np.float32)).shape
            for count in (1, 7, 21, 321)
        ]

        self.assertEqual(shapes, [(24, 8, 29)] * 4)

    def test_general_collator_preserves_history_and_masks_padding(self):
        batch = collate_general_graph_batch([_sample(3), _sample(5)])

        self.assertEqual(tuple(batch["maps"].shape), (2, 18, 4, 8))
        self.assertEqual(tuple(batch["history_scaled"].shape), (2, 36, 5))
        self.assertEqual(tuple(batch["y"].shape), (2, 24, 5))
        self.assertEqual(
            batch["variable_mask"].tolist(),
            [[True, True, True, False, False], [True, True, True, True, True]],
        )
        torch.testing.assert_close(batch["history_scaled"][0, :, :3], _sample(3)["history_scaled"])
        self.assertEqual(torch.count_nonzero(batch["history_scaled"][0, :, 3:]).item(), 0)
        self.assertFalse(batch["mask"][0, :, 3:].any())


class GeneralGraphReportShapeTests(unittest.TestCase):
    def test_formal_short_horizons_retain_every_variable(self):
        for variable_count, horizon in ((1, 24), (7, 36), (21, 48), (321, 60)):
            with self.subTest(variable_count=variable_count, horizon=horizon):
                model = _general_model()
                output = model(
                    torch.randn(1, 18, 4, 8),
                    ["forecast all variables"],
                    torch.tensor([horizon]),
                    history_features=torch.randn(1, 36, variable_count),
                    variable_mask=torch.ones(1, variable_count, dtype=torch.bool),
                )
                self.assertIsInstance(model.graph_encoder, graph_model.GraphMapEncoder)
                self.assertEqual(tuple(output["pred"].shape), (1, horizon, variable_count))

    def test_general_variable_permutation_is_equivariant(self):
        torch.manual_seed(31)
        model = _general_model()
        graph = torch.randn(1, 18, 4, 8)
        history = torch.randn(1, 36, 3)
        mask = torch.ones(1, 3, dtype=torch.bool)
        permutation = torch.tensor([2, 0, 1])

        original = model(graph, ["forecast"], torch.tensor([36]), history_features=history, variable_mask=mask)
        permuted = model(
            graph,
            ["forecast"],
            torch.tensor([36]),
            history_features=history[:, :, permutation],
            variable_mask=mask[:, permutation],
        )

        torch.testing.assert_close(permuted["pred"], original["pred"][:, :, permutation])

    def test_prompt_path_ignores_padded_history_variables(self):
        torch.manual_seed(23)
        model = _general_model(use_prompt=True)
        graph = torch.randn(1, 18, 4, 8)
        history = torch.randn(1, 36, 3)
        valid_mask = torch.ones(1, 3, dtype=torch.bool)
        padded_history = torch.cat((history, torch.randn(1, 36, 2) * 100.0), dim=-1)
        padded_mask = torch.tensor([[True, True, True, False, False]])

        valid = model(graph, ["forecast every variable"], 24, history_features=history, variable_mask=valid_mask)
        padded = model(
            graph,
            ["forecast every variable"],
            24,
            history_features=padded_history,
            variable_mask=padded_mask,
        )

        torch.testing.assert_close(padded["pred"][..., :3], valid["pred"])
        self.assertEqual(torch.count_nonzero(padded["pred"][..., 3:]).item(), 0)

    def test_backward_reaches_shared_variable_history_encoder(self):
        torch.manual_seed(29)
        model = _general_model()
        graph = torch.randn(1, 18, 4, 8, requires_grad=True)
        history = torch.randn(1, 36, 4, requires_grad=True)
        mask = torch.tensor([[True, True, True, False]])

        output = model(graph, ["forecast"], 24, history_features=history, variable_mask=mask)
        output["pred"].square().sum().backward()

        self.assertTrue(torch.all(history.grad[:, :, :3].abs().sum(dim=1) > 0))
        self.assertEqual(torch.count_nonzero(history.grad[:, :, 3:]).item(), 0)
        self.assertGreater(float(graph.grad.abs().sum()), 0.0)
        self.assertTrue(hasattr(model, "variable_decoder"))
        self.assertTrue(hasattr(model, "variable_history_encoder"))


class BatteryIsolationTests(unittest.TestCase):
    def test_battery_still_uses_graph_map_encoder(self):
        battery = graph_model.GraphReportTS(
            graph_model.GraphReportTSConfig(
                variant="battery",
                output_dim=1,
                d_model=8,
                patch_size=4,
                patch_stride=4,
                graph_layers=1,
                dropout=0.0,
                use_report_prompt=False,
                use_cross_modal_fusion=False,
                use_hf_text_encoder=False,
                use_numeric_history=False,
            )
        ).eval()

        self.assertIsInstance(battery.graph_encoder, graph_model.GraphMapEncoder)
        self.assertIsNone(battery.variable_history_encoder)


class GeneralInferenceMaskTests(unittest.TestCase):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(use_relative_steps=True)
            self.variable_mask = None
            self.history_features = None

        def forward(self, maps, prompts, horizon, steps=None, history_features=None, variable_mask=None):
            self.variable_mask = variable_mask
            self.history_features = history_features
            return {
                "pred": maps.new_zeros(maps.size(0), int(horizon.max()), variable_mask.size(1))
            }

    def test_prediction_collection_forwards_general_variable_mask(self):
        model = self.RecordingModel()
        variable_mask = torch.tensor([[True, True, False]])
        batch = {
            "maps": torch.zeros(1, 18, 4, 8),
            "history_scaled": torch.zeros(1, 36, 3),
            "variable_mask": variable_mask,
            "prompt": ["forecast"],
            "horizon": torch.tensor([2]),
            "y": torch.zeros(1, 2, 3),
            "mask": torch.tensor([[[True, True, False], [True, True, False]]]),
            "target_steps": torch.tensor([[36, 37]]),
            "series_id": ["synthetic"],
            "start_index": torch.tensor([0]),
        }

        graph_inference.collect_predictions(model, [batch], torch.device("cpu"), "general")

        self.assertIs(model.variable_mask, variable_mask)
        self.assertIs(model.history_features, batch["history_scaled"])


if __name__ == "__main__":
    unittest.main()
