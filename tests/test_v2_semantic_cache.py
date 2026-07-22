from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch
import torch.nn as nn

from bstalignment.models import HFTextEncoder
from bstalignment.v2.contracts import GraphReportTSv2Config
from bstalignment.v2.semantic import FrozenDistilBERTEncoder


class _FakeTokenizer:
    pad_token_id = 0

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
        tokens = [101, *(3 + (ord(character) % 29) for character in prompt), 102]
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}


class _CountingBackbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.register_parameter("scale", nn.Parameter(torch.ones(hidden_size), requires_grad=False))
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, input_ids: torch.Tensor, **_: torch.Tensor) -> SimpleNamespace:
        self.calls += 1
        self.batch_sizes.append(int(input_ids.size(0)))
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
    encoder.backbone = _CountingBackbone(hidden_size=4)
    encoder.hidden_size = 4
    encoder.proj = nn.Linear(4, d_model, bias=False)
    encoder.max_length = max_length
    encoder.last_prompt_audit = None
    encoder.freeze = freeze


class FrozenDistilBERTHiddenCacheTests(unittest.TestCase):
    def build_encoder(
        self,
        *,
        hidden_cache_size: int = 8,
        hidden_cache_max_bytes: int = 1 << 20,
    ) -> FrozenDistilBERTEncoder:
        with mock.patch.object(HFTextEncoder, "__init__", new=_fake_hf_init):
            return FrozenDistilBERTEncoder(
                "fake-distilbert",
                d_model=3,
                max_length=12,
                token_cache_size=16,
                hidden_cache_size=hidden_cache_size,
                hidden_cache_max_bytes=hidden_cache_max_bytes,
            )

    def test_second_forward_reuses_cpu_hidden_state_and_keeps_projection_trainable(self) -> None:
        encoder = self.build_encoder()
        encoder.train()

        encoder(["alpha", "beta"])
        encoder.proj.zero_grad(set_to_none=True)
        tokens, pooled, mask = encoder(["beta", "alpha"])
        (tokens.sum() + pooled.sum()).backward()

        self.assertEqual(encoder.backbone.calls, 1)
        self.assertEqual(tuple(mask.shape), tuple(tokens.shape[:2]))
        self.assertIsNotNone(encoder.proj.weight.grad)
        self.assertGreater(float(encoder.proj.weight.grad.abs().sum()), 0.0)
        self.assertFalse(encoder.backbone.scale.requires_grad)
        self.assertIsNone(encoder.backbone.scale.grad)
        cached = next(iter(encoder._hidden_cache.values()))
        self.assertEqual(cached.hidden.device.type, "cpu")
        self.assertEqual(cached.hidden.dtype, torch.bfloat16)

    def test_mixed_hits_only_encode_unique_missing_prompts(self) -> None:
        encoder = self.build_encoder()
        encoder(["alpha", "beta"])

        tokens, _, mask = encoder(["beta", "gamma", "alpha", "gamma"])

        self.assertEqual(encoder.backbone.batch_sizes, [2, 1])
        self.assertTrue(torch.equal(tokens[1], tokens[3]))
        self.assertTrue(torch.equal(mask[1], mask[3]))

    def test_entry_and_byte_limits_are_independent_and_bounded(self) -> None:
        one_entry = self.build_encoder(hidden_cache_size=1)
        one_entry(["alpha"])
        one_entry(["beta"])
        one_entry(["alpha"])
        self.assertEqual(one_entry.backbone.calls, 3)
        self.assertEqual(one_entry.hidden_cache_entries, 1)
        self.assertLessEqual(one_entry.hidden_cache_current_bytes, one_entry.hidden_cache_max_bytes)

        no_room = self.build_encoder(hidden_cache_size=8, hidden_cache_max_bytes=1)
        no_room(["alpha"])
        no_room(["alpha"])
        self.assertEqual(no_room.backbone.calls, 2)
        self.assertEqual(no_room.hidden_cache_entries, 0)
        self.assertEqual(no_room.hidden_cache_current_bytes, 0)

    def test_config_rejects_negative_hidden_cache_budgets(self) -> None:
        for field in ("text_hidden_cache_size", "text_hidden_cache_max_bytes"):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                GraphReportTSv2Config(
                    domain="general",
                    input_len=36,
                    pred_len=24,
                    use_text=False,
                    **{field: -1},
                )


if __name__ == "__main__":
    unittest.main()
