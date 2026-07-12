from __future__ import annotations

import inspect
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import bstalignment.graph_report_model as graph_model
import bstalignment.infer_graph_report as graph_inference
import bstalignment.raw_signal as raw_signal
from bstalignment.data_general import collate_general_graph_batch
from bstalignment.train_graph_report import parse_args


CUDA_ECL_PEAK_LIMIT_BYTES = 20 * 1024**3
CUDA_ECL_REQUIRED_FREE_BYTES = 22 * 1024**3


def _require_variable_interfaces():
    build_variable_maps = getattr(raw_signal, "build_variable_maps", None)
    variable_encoder = getattr(graph_model, "VariableGraphEncoder", None)
    if build_variable_maps is None or variable_encoder is None:
        raise AssertionError("bounded general variable-map interfaces are not implemented")
    return build_variable_maps, variable_encoder


def _sample(variable_count: int, horizon: int = 96) -> dict[str, object]:
    maps = torch.arange(variable_count * 3 * 4 * 8, dtype=torch.float32).reshape(variable_count, 3, 4, 8)
    return {
        "maps": maps,
        "y": torch.zeros(horizon, variable_count),
        "mask": torch.ones(horizon, variable_count, dtype=torch.bool),
        "horizon": torch.tensor(horizon),
        "prompt": "forecast all variables",
        "series_id": "synthetic",
        "start_index": 0,
        "target_steps": torch.arange(36, 36 + horizon),
    }


def _general_model(
    variable_count: int,
    chunk_size: int = 32,
    use_prompt: bool = False,
) -> graph_model.GraphReportTS:
    config = graph_model.GraphReportTSConfig(
        variant="general",
        output_dim=variable_count,
        d_model=8,
        max_steps=720,
        patch_size=4,
        patch_stride=4,
        graph_layers=1,
        topk_edges=2,
        dropout=0.0,
        use_report_prompt=use_prompt,
        use_cross_modal_fusion=use_prompt,
        use_hf_text_encoder=False,
        use_numeric_history=False,
        variable_chunk_size=chunk_size,
    )
    return graph_model.GraphReportTS(config).eval()


class GeneralVariableMapTests(unittest.TestCase):
    def test_variable_maps_preserve_all_321_variables_and_existing_map_values(self):
        build_variable_maps, _ = _require_variable_interfaces()
        history = np.arange(36 * 321, dtype=np.float32).reshape(36, 321)

        variable_maps = build_variable_maps(history, resample_len=36, delay_dim=4, delay_lag=1)
        legacy_maps, _ = raw_signal.build_multiview_maps(
            {f"x{i}": history[:, i] for i in range(321)},
            resample_len=36,
            delay_dim=4,
            delay_lag=1,
        )

        self.assertEqual(variable_maps.shape, (321, 3, 4, 33))
        np.testing.assert_array_equal(variable_maps.reshape(-1, 4, 33), legacy_maps)

    def test_general_collator_pads_variables_and_marks_padding_invalid(self):
        batch = collate_general_graph_batch([_sample(3), _sample(5)])

        self.assertEqual(tuple(batch["maps"].shape), (2, 5, 3, 4, 8))
        self.assertEqual(tuple(batch["y"].shape), (2, 96, 5))
        self.assertEqual(tuple(batch["mask"].shape), (2, 96, 5))
        self.assertEqual(
            batch["variable_mask"].tolist(),
            [[True, True, True, False, False], [True, True, True, True, True]],
        )
        self.assertFalse(batch["mask"][0, :, 3:].any())


class VariableGraphEncoderTests(unittest.TestCase):
    def test_padding_mask_makes_padded_variable_values_invariant(self):
        _, variable_encoder = _require_variable_interfaces()
        torch.manual_seed(7)
        encoder = variable_encoder(
            d_model=8,
            patch_size=4,
            patch_stride=4,
            graph_layers=1,
            topk_edges=2,
            dropout=0.0,
            variable_chunk_size=8,
        ).eval()
        valid_maps = torch.randn(1, 3, 3, 4, 8)
        padded_maps = torch.cat([valid_maps, torch.randn(1, 2, 3, 4, 8)], dim=1)

        valid = encoder(valid_maps, torch.ones(1, 3, dtype=torch.bool))
        padded = encoder(padded_maps, torch.tensor([[True, True, True, False, False]]))

        torch.testing.assert_close(padded["tokens"][:, :3], valid["tokens"])
        torch.testing.assert_close(padded["repr"], valid["repr"])
        self.assertEqual(torch.count_nonzero(padded["tokens"][:, 3:]).item(), 0)

    def test_321_variables_are_encoded_in_fixed_chunks(self):
        _, variable_encoder = _require_variable_interfaces()
        chunk_size = 16
        encoder = variable_encoder(
            d_model=8,
            patch_size=4,
            patch_stride=4,
            graph_layers=1,
            topk_edges=2,
            dropout=0.0,
            variable_chunk_size=chunk_size,
        ).eval()
        observed_batches: list[int] = []
        handle = encoder.map_encoder.register_forward_pre_hook(
            lambda _module, inputs: observed_batches.append(inputs[0].shape[0])
        )
        try:
            with torch.inference_mode():
                output = encoder(torch.zeros(1, 321, 3, 4, 8), torch.ones(1, 321, dtype=torch.bool))
        finally:
            handle.remove()

        self.assertEqual(tuple(output["tokens"].shape), (1, 321, 8))
        self.assertEqual(tuple(output["repr"].shape), (1, 8))
        self.assertEqual(len(observed_batches), math.ceil(321 / chunk_size))
        self.assertLessEqual(max(observed_batches), chunk_size)

    def test_chunk_size_does_not_change_predictions_and_diagnostic_is_bounded_summary(self):
        torch.manual_seed(11)
        maps = torch.randn(1, 5, 3, 4, 8)
        variable_mask = torch.ones(1, 5, dtype=torch.bool)
        chunk_one = _general_model(5, chunk_size=1)
        chunk_four = _general_model(5, chunk_size=4)
        with torch.inference_mode():
            chunk_one(maps, ["forecast"], torch.tensor([6]), variable_mask=variable_mask)
            chunk_four(maps, ["forecast"], torch.tensor([6]), variable_mask=variable_mask)
        chunk_four.load_state_dict(chunk_one.state_dict())

        with torch.inference_mode():
            one = chunk_one(maps, ["forecast"], torch.tensor([6]), variable_mask=variable_mask)
            four = chunk_four(maps, ["forecast"], torch.tensor([6]), variable_mask=variable_mask)

        torch.testing.assert_close(one["pred"], four["pred"], rtol=1e-5, atol=1e-6)
        self.assertIsInstance(one["graph_attn"], dict)
        self.assertEqual(one["graph_attn"]["mode"], "chunked_summary")
        self.assertEqual(one["graph_attn"]["chunk_count"], 5)
        self.assertEqual(four["graph_attn"]["chunk_count"], 2)
        self.assertNotIn("attention", one["graph_attn"])


class GeneralGraphReportShapeTests(unittest.TestCase):
    def test_formal_horizons_retain_every_variable(self):
        build_variable_maps, variable_encoder = _require_variable_interfaces()
        cases = ((7, 96), (21, 192), (7, 336), (321, 720))
        for variable_count, horizon in cases:
            with self.subTest(variable_count=variable_count, horizon=horizon):
                history = np.linspace(-1.0, 1.0, 36 * variable_count, dtype=np.float32).reshape(36, variable_count)
                maps = torch.from_numpy(
                    build_variable_maps(history, resample_len=36, delay_dim=4, delay_lag=1)
                ).unsqueeze(0)
                model = _general_model(variable_count)

                with torch.inference_mode():
                    output = model(
                        maps,
                        ["forecast all variables"],
                        torch.tensor([horizon]),
                        variable_mask=torch.ones(1, variable_count, dtype=torch.bool),
                    )

                self.assertIsInstance(model.graph_encoder, variable_encoder)
                self.assertEqual(tuple(output["pred"].shape), (1, horizon, variable_count))

    def test_legacy_general_flag_restores_global_graph_and_battery_layout_is_unchanged(self):
        _, variable_encoder = _require_variable_interfaces()
        legacy = graph_model.GraphReportTS(
            graph_model.GraphReportTSConfig(
                variant="general",
                output_dim=3,
                d_model=8,
                patch_size=4,
                patch_stride=4,
                graph_layers=1,
                dropout=0.0,
                use_report_prompt=False,
                use_cross_modal_fusion=False,
                use_hf_text_encoder=False,
                use_numeric_history=False,
                legacy_general_graph=True,
            )
        ).eval()
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

        self.assertIsInstance(legacy.graph_encoder, graph_model.GraphMapEncoder)
        self.assertNotIsInstance(legacy.graph_encoder, variable_encoder)
        self.assertIsInstance(battery.graph_encoder, graph_model.GraphMapEncoder)
        self.assertIn("graph_encoder.input_proj.weight", battery.state_dict())
        self.assertFalse(any(key.startswith("graph_encoder.map_encoder.") for key in battery.state_dict()))

    def test_general_cli_defaults_to_scalable_path_and_exposes_diagnostic_legacy_flag(self):
        with patch("sys.argv", ["train_graph_report", "--variant", "general", "--pred_len", "96"]):
            scalable = parse_args()
        with patch(
            "sys.argv",
            ["train_graph_report", "--variant", "general", "--pred_len", "96", "--legacy_general_graph"],
        ):
            legacy = parse_args()

        self.assertFalse(scalable.legacy_general_graph)
        self.assertEqual(scalable.variable_chunk_size, 32)
        self.assertTrue(legacy.legacy_general_graph)


class GeneralVariableIdentityTests(unittest.TestCase):
    def test_prompt_path_is_permutation_equivariant(self):
        torch.manual_seed(17)
        model = _general_model(4, chunk_size=2, use_prompt=True)
        maps = torch.randn(1, 4, 3, 4, 8)
        variable_mask = torch.tensor([[True, True, False, True]])
        permutation = torch.tensor([3, 0, 2, 1])

        with torch.inference_mode():
            original = model(maps, ["forecast every variable"], 7, variable_mask=variable_mask)
            permuted = model(
                maps.index_select(1, permutation),
                ["forecast every variable"],
                7,
                variable_mask=variable_mask.index_select(1, permutation),
            )

        inverse_permutation = torch.argsort(permutation)
        torch.testing.assert_close(
            permuted["pred"].index_select(-1, inverse_permutation),
            original["pred"],
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(permuted["context"], original["context"], rtol=1e-5, atol=1e-6)

    def test_identical_variable_inputs_have_identical_output_channels(self):
        torch.manual_seed(19)
        model = _general_model(3, use_prompt=True)
        one_variable = torch.randn(1, 1, 3, 4, 8)
        maps = one_variable.expand(-1, 3, -1, -1, -1).clone()

        with torch.inference_mode():
            pred = model(maps, ["forecast every variable"], 5, variable_mask=torch.ones(1, 3, dtype=torch.bool))[
                "pred"
            ]

        torch.testing.assert_close(pred[..., 0], pred[..., 1])
        torch.testing.assert_close(pred[..., 1], pred[..., 2])

    def test_full_prompt_path_ignores_padded_variable_slots(self):
        torch.manual_seed(23)
        model = _general_model(5, chunk_size=2, use_prompt=True)
        valid_maps = torch.randn(1, 3, 3, 4, 8)
        padded_maps = torch.cat([valid_maps, torch.randn(1, 2, 3, 4, 8) * 100.0], dim=1)
        valid_mask = torch.ones(1, 3, dtype=torch.bool)
        padded_mask = torch.tensor([[True, True, True, False, False]])

        with torch.inference_mode():
            valid = model(valid_maps, ["forecast every variable"], 6, variable_mask=valid_mask)
            padded = model(padded_maps, ["forecast every variable"], 6, variable_mask=padded_mask)

        self.assertEqual(tuple(valid["pred"].shape), (1, 6, 3))
        self.assertEqual(tuple(padded["pred"].shape), (1, 6, 5))
        torch.testing.assert_close(padded["pred"][..., :3], valid["pred"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(padded["context"], valid["context"], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(padded["cross_attn"][:, :3], valid["cross_attn"], rtol=1e-5, atol=1e-6)
        self.assertEqual(torch.count_nonzero(padded["pred"][..., 3:]).item(), 0)

    def test_backward_reaches_each_valid_variable_and_shared_future_head(self):
        torch.manual_seed(29)
        model = _general_model(4, chunk_size=2)
        maps = torch.randn(1, 4, 3, 4, 8, requires_grad=True)
        variable_mask = torch.tensor([[True, True, True, False]])

        output = model(maps, ["forecast"], 4, variable_mask=variable_mask)
        output["pred"].square().sum().backward()

        self.assertTrue(hasattr(model, "variable_decoder"))
        valid_gradient = maps.grad[:, :3].abs().flatten(2).sum(dim=-1)
        self.assertTrue(torch.all(valid_gradient > 0))
        self.assertEqual(torch.count_nonzero(maps.grad[:, 3:]).item(), 0)
        for parameter in model.variable_decoder.parameters():
            if parameter.requires_grad:
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(
            sum(float(parameter.grad.abs().sum()) for parameter in model.variable_decoder.parameters()),
            0.0,
        )


class GeneralInferenceMaskTests(unittest.TestCase):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.cfg = SimpleNamespace(use_relative_steps=True)
            self.variable_mask = None

        def forward(
            self,
            maps,
            prompts,
            horizon,
            steps=None,
            history_features=None,
            variable_mask=None,
        ):
            self.variable_mask = variable_mask
            return {"pred": maps.new_zeros(maps.size(0), int(horizon.max()), maps.size(1))}

    def test_prediction_collection_forwards_general_variable_mask(self):
        model = self.RecordingModel()
        variable_mask = torch.tensor([[True, True, False]])
        batch = {
            "maps": torch.zeros(1, 3, 3, 4, 8),
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

    def test_inference_lazy_initialization_uses_the_shared_mask_forwarding_helper(self):
        helper = getattr(graph_inference, "forward_inference_batch", None)
        self.assertIsNotNone(helper)
        source = inspect.getsource(graph_inference.main)
        self.assertIn("forward_inference_batch(model, init_batch, variant)", source)


class CudaECLSmokeTests(unittest.TestCase):
    def test_ecl_forward_stays_below_4090_safe_peak(self):
        if os.environ.get("RUN_CUDA_ECL_SMOKE") != "1":
            self.skipTest("set RUN_CUDA_ECL_SMOKE=1 during readiness QA to enable the CUDA ECL smoke test")
        if not torch.cuda.is_available():
            self.skipTest("CUDA is unavailable")
        free_bytes, _ = torch.cuda.mem_get_info()
        if free_bytes < CUDA_ECL_REQUIRED_FREE_BYTES:
            self.skipTest(
                f"GPU headroom is unavailable: {free_bytes / 1024**3:.1f} GiB free; "
                f"requires {CUDA_ECL_REQUIRED_FREE_BYTES / 1024**3:.0f} GiB"
            )

        build_variable_maps, _ = _require_variable_interfaces()
        device = torch.device("cuda")
        history = np.zeros((36, 321), dtype=np.float32)
        maps = torch.from_numpy(build_variable_maps(history)).unsqueeze(0).to(device)
        model = graph_model.GraphReportTS(
            graph_model.GraphReportTSConfig(
                variant="general",
                output_dim=321,
                max_steps=720,
                use_report_prompt=False,
                use_cross_modal_fusion=False,
                use_hf_text_encoder=False,
                use_numeric_history=False,
                variable_chunk_size=32,
            )
        ).to(device).eval()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.inference_mode():
            output = model(
                maps,
                ["forecast all variables"],
                torch.tensor([720], device=device),
                variable_mask=torch.ones(1, 321, dtype=torch.bool, device=device),
            )
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device)

        self.assertEqual(tuple(output["pred"].shape), (1, 720, 321))
        self.assertLess(peak, CUDA_ECL_PEAK_LIMIT_BYTES)


if __name__ == "__main__":
    unittest.main()
