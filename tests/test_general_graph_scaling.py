from __future__ import annotations

import math
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

import bstalignment.graph_report_model as graph_model
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


def _general_model(variable_count: int, chunk_size: int = 32) -> graph_model.GraphReportTS:
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
        use_report_prompt=False,
        use_cross_modal_fusion=False,
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
