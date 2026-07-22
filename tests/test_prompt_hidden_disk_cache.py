from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

from anchoredgtr.models import HFTextEncoder
from anchoredgtr.core.prompt_hidden_cache import (
    BATTERY_PROMPT_HIDDEN_CACHE_ROOT,
    battery_prompt_hidden_dataset_name,
    ensure_prompt_hidden_split_cache,
    install_prompt_hidden_caches,
    local_text_model_identity,
)
from anchoredgtr.core.semantic import FrozenDistilBERTEncoder
from anchoredgtr.core.train_general import _dataset_prompts, parse_args


class _FakeTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self,
        prompt: str,
        *,
        padding: bool,
        truncation: bool,
        add_special_tokens: bool,
        max_length: int | None = None,
    ) -> dict[str, list[int]]:
        del padding, add_special_tokens
        self.calls += 1
        tokens = [101, *(3 + ord(character) % 17 for character in prompt), 102]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


class _CountingBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(4), requires_grad=False))
        self.calls = 0

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        hidden = input_ids.float().unsqueeze(-1) * self.scale.view(1, 1, -1)
        return SimpleNamespace(last_hidden_state=hidden)


def _fake_hf_init(
    encoder: HFTextEncoder,
    model_name: str,
    d_model: int,
    freeze: bool,
    max_length: int,
) -> None:
    nn.Module.__init__(encoder)
    encoder.tokenizer = _FakeTokenizer()
    encoder.backbone = _CountingBackbone()
    encoder.hidden_size = 4
    encoder.proj = nn.Linear(4, d_model, bias=False)
    encoder.max_length = max_length
    encoder.last_prompt_audit = None
    encoder.freeze = freeze


class PersistentPromptHiddenCacheTests(unittest.TestCase):
    def test_battery_shared_cache_namespace_is_neutral(self) -> None:
        self.assertEqual(
            BATTERY_PROMPT_HIDDEN_CACHE_ROOT,
            "data/battery/cache/distilbert_hidden",
        )
        self.assertEqual(battery_prompt_hidden_dataset_name("mit"), "battery-shared-mit")
        self.assertEqual(battery_prompt_hidden_dataset_name("xjtu"), "battery-shared-xjtu")
        with self.assertRaisesRegex(ValueError, "battery prompt cache"):
            battery_prompt_hidden_dataset_name("calce")

    def build_encoder(self, *, entries: int = 32, max_bytes: int = 1 << 20) -> FrozenDistilBERTEncoder:
        with mock.patch.object(HFTextEncoder, "__init__", new=_fake_hf_init):
            return FrozenDistilBERTEncoder(
                "fake-distilbert",
                d_model=3,
                max_length=12,
                token_cache_size=32,
                hidden_cache_size=entries,
                hidden_cache_max_bytes=max_bytes,
            )

    @staticmethod
    def base_identity() -> dict[str, object]:
        return {
            "schema": "gtr-general-prompt-hidden-v1",
            "model": {"path": "fake-distilbert", "sha256": "model-sha"},
            "text_max_length": 12,
            "prompt_schema": {"schema": "anchored-gtr-prompt-v1", "sha256": "prompt-schema-sha"},
            "dataset_identity": {"name": "synthetic", "sha256": "dataset-sha"},
            "storage_dtype": "bfloat16",
            "cache_boundary": "backbone.last_hidden_state+attention_mask",
        }

    def build_split(self, root: Path, encoder: FrozenDistilBERTEncoder, prompts: list[str]):
        return ensure_prompt_hidden_split_cache(
            encoder,
            prompts,
            split="train",
            cache_root=root,
            dataset_name="synthetic",
            horizon=24,
            base_identity=self.base_identity(),
            precompute_batch_size=2,
            chunk_size=2,
        )

    def test_precompute_reuses_disk_and_keeps_projection_outside_cache(self) -> None:
        encoder = self.build_encoder()
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.build_split(Path(temporary), encoder, ["alpha", "beta", "alpha"])
            self.assertEqual(manifest["cache_status"], "rebuilt")
            self.assertEqual(encoder.backbone.calls, 1)
            identity = manifest["identity"]
            self.assertEqual(identity["split"], "train")
            self.assertEqual(identity["prompt_count"], 3)
            self.assertEqual(identity["unique_prompt_count"], 2)

            encoder.clear_hidden_cache()
            installed = install_prompt_hidden_caches(encoder, [manifest])
            self.assertEqual(installed["installed_entries"], 2)
            tokenizer_calls = encoder.tokenizer.calls
            encoder.proj.zero_grad(set_to_none=True)
            tokens, pooled, _ = encoder(["beta", "alpha"])
            (tokens.sum() + pooled.sum()).backward()
            self.assertEqual(encoder.backbone.calls, 1)
            self.assertEqual(encoder.tokenizer.calls, tokenizer_calls)
            self.assertGreater(float(encoder.proj.weight.grad.abs().sum()), 0.0)

            reused = self.build_split(Path(temporary), encoder, ["alpha", "beta", "alpha"])
            self.assertEqual(reused["cache_status"], "reused")
            self.assertEqual(encoder.backbone.calls, 1)

    def test_changed_prompt_identity_and_corrupt_chunk_are_rebuilt(self) -> None:
        encoder = self.build_encoder()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = self.build_split(root, encoder, ["alpha", "beta"])
            changed = self.build_split(root, encoder, ["alpha", "gamma"])
            self.assertEqual(changed["cache_status"], "rebuilt")
            self.assertNotEqual(original["identity_sha256"], changed["identity_sha256"])

            chunk = Path(changed["manifest_path"]).parent / changed["chunks"][0]["file"]
            chunk.write_bytes(b"corrupt")
            repaired = self.build_split(root, encoder, ["alpha", "gamma"])
            self.assertEqual(repaired["cache_status"], "rebuilt")
            self.assertGreater(chunk.stat().st_size, len(b"corrupt"))

    def test_install_fails_clearly_when_memory_cache_is_too_small(self) -> None:
        producer = self.build_encoder()
        consumer = self.build_encoder(entries=1)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = self.build_split(Path(temporary), producer, ["alpha", "beta"])
            with self.assertRaisesRegex(RuntimeError, "capacity is insufficient"):
                install_prompt_hidden_caches(consumer, [manifest])

    def test_local_model_identity_changes_when_weight_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text('{"_commit_hash":"revision-a"}', encoding="utf-8")
            (root / "tokenizer.json").write_text("tokenizer", encoding="utf-8")
            weights = root / "model.safetensors"
            weights.write_bytes(b"first")
            first = local_text_model_identity(root)
            weights.write_bytes(b"second")
            second = local_text_model_identity(root)
            self.assertEqual(first["revision"], "revision-a")
            self.assertNotEqual(first["sha256"], second["sha256"])

    def test_cli_exposes_precompute_controls(self) -> None:
        args = parse_args(["--dataset", "synthetic", "--output", "out"])
        self.assertTrue(args.text_hidden_precompute_cache)
        self.assertEqual(args.text_hidden_precompute_batch_size, 64)
        self.assertEqual(args.text_hidden_cache_root, "data/general/cache/distilbert_hidden")

        disabled = parse_args(
            ["--dataset", "synthetic", "--output", "out", "--no-text-hidden-precompute-cache"]
        )
        self.assertFalse(disabled.text_hidden_precompute_cache)

    def test_precompute_prompt_collection_does_not_read_targets(self) -> None:
        class _PromptOnlyDataset:
            def __len__(self) -> int:
                return 2

            def prompt_at(self, index: int) -> str:
                return f"causal-{index}"

            def __getitem__(self, index: int):
                raise AssertionError(f"target-bearing item {index} must not be read")

        self.assertEqual(_dataset_prompts(_PromptOnlyDataset()), ["causal-0", "causal-1"])


if __name__ == "__main__":
    unittest.main()
