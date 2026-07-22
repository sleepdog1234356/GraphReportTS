from __future__ import annotations

import math
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ..models import HFTextEncoder


@dataclass
class SemanticOutput:
    variable_tokens: torch.Tensor
    router_tokens: torch.Tensor
    context: torch.Tensor
    gate: torch.Tensor
    cross_attention: torch.Tensor
    align_graph: torch.Tensor
    align_text: torch.Tensor
    prompt_audit: list[dict[str, int | bool]] | None


@dataclass(frozen=True)
class CachedHiddenState:
    """One prompt's frozen-backbone output before the trainable projection."""

    hidden: torch.Tensor
    mask: torch.Tensor
    token_count: int
    nbytes: int


class FrozenDistilBERTEncoder(HFTextEncoder):
    """Local-only frozen DistilBERT with its train/eval invariant inherited from v1."""

    def __init__(
        self,
        model_path: str,
        d_model: int,
        max_length: int = 128,
        token_cache_size: int = 16_384,
        hidden_cache_size: int = 4_096,
        hidden_cache_max_bytes: int = 512 * 1024 * 1024,
    ) -> None:
        super().__init__(model_path, d_model=d_model, freeze=True, max_length=max_length)
        self.model_path = str(model_path)
        self.token_cache_size = max(int(token_cache_size), 0)
        self._token_cache: OrderedDict[tuple[str, int, str], tuple[dict[str, list[int]], int]] = OrderedDict()
        self.hidden_cache_size = max(int(hidden_cache_size), 0)
        self.hidden_cache_max_bytes = max(int(hidden_cache_max_bytes), 0)
        self._hidden_cache: OrderedDict[tuple[str, int, str], CachedHiddenState] = OrderedDict()
        self._hidden_cache_current_bytes = 0

    @property
    def hidden_cache_entries(self) -> int:
        return len(self._hidden_cache)

    @property
    def hidden_cache_current_bytes(self) -> int:
        return self._hidden_cache_current_bytes

    def clear_hidden_cache(self) -> None:
        self._hidden_cache.clear()
        self._hidden_cache_current_bytes = 0

    def _hidden_cache_get(self, key: tuple[str, int, str]) -> CachedHiddenState | None:
        cached = self._hidden_cache.get(key)
        if cached is not None:
            self._hidden_cache.move_to_end(key)
        return cached

    @staticmethod
    def _tensor_nbytes(value: torch.Tensor) -> int:
        return int(value.numel() * value.element_size())

    def _hidden_cache_put(
        self,
        key: tuple[str, int, str],
        hidden: torch.Tensor,
        mask: torch.Tensor,
        token_count: int | None = None,
    ) -> CachedHiddenState | None:
        cached_hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
        cached_mask = mask.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()
        key_bytes = len(key[0].encode("utf-8")) + len(key[2].encode("utf-8")) + 16
        nbytes = self._tensor_nbytes(cached_hidden) + self._tensor_nbytes(cached_mask) + key_bytes + 8
        entry = CachedHiddenState(
            cached_hidden,
            cached_mask,
            int(token_count) if token_count is not None else int(cached_mask.sum()),
            nbytes,
        )
        if not self.hidden_cache_size or not self.hidden_cache_max_bytes or nbytes > self.hidden_cache_max_bytes:
            return None
        previous = self._hidden_cache.pop(key, None)
        if previous is not None:
            self._hidden_cache_current_bytes -= previous.nbytes
        while self._hidden_cache and (
            len(self._hidden_cache) >= self.hidden_cache_size
            or self._hidden_cache_current_bytes + nbytes > self.hidden_cache_max_bytes
        ):
            _, evicted = self._hidden_cache.popitem(last=False)
            self._hidden_cache_current_bytes -= evicted.nbytes
        self._hidden_cache[key] = entry
        self._hidden_cache_current_bytes += nbytes
        return entry

    def cached_hidden_state(self, prompt: str) -> CachedHiddenState | None:
        """Return a CPU BF16 backbone state without applying ``self.proj``."""

        return self._hidden_cache_get((self.model_path, self.max_length, prompt))

    def install_hidden_states(
        self,
        entries: list[tuple[str, torch.Tensor, torch.Tensor, int]],
        *,
        replace: bool = True,
        require_all: bool = True,
    ) -> dict[str, int]:
        """Prime the in-memory cache from validated persistent entries.

        Capacity is checked before mutating the cache so a formal run cannot
        silently fall back to repeated DistilBERT inference after LRU eviction.
        """

        unique: OrderedDict[tuple[str, int, str], CachedHiddenState] = OrderedDict()
        for prompt, hidden, mask, token_count in entries:
            key = (self.model_path, self.max_length, prompt)
            cached_hidden = hidden.detach().to(device="cpu", dtype=torch.bfloat16).contiguous()
            cached_mask = mask.detach().to(device="cpu", dtype=torch.bool).contiguous().clone()
            if cached_hidden.ndim != 2 or cached_hidden.size(1) != self.hidden_size:
                raise ValueError("persistent text hidden state has an incompatible hidden width")
            if cached_mask.ndim != 1 or cached_mask.size(0) != cached_hidden.size(0):
                raise ValueError("persistent text hidden mask has an incompatible shape")
            if not 1 <= cached_hidden.size(0) <= self.max_length or not cached_mask.any():
                raise ValueError("persistent text hidden state has an invalid token length")
            if int(token_count) < int(cached_mask.sum()):
                raise ValueError("persistent text hidden token count is invalid")
            key_bytes = len(key[0].encode("utf-8")) + len(key[2].encode("utf-8")) + 16
            nbytes = self._tensor_nbytes(cached_hidden) + self._tensor_nbytes(cached_mask) + key_bytes + 8
            candidate = CachedHiddenState(cached_hidden, cached_mask, int(token_count), nbytes)
            previous = unique.get(key)
            if previous is not None and (
                not torch.equal(previous.hidden, candidate.hidden)
                or not torch.equal(previous.mask, candidate.mask)
                or previous.token_count != candidate.token_count
            ):
                raise ValueError("the same prompt maps to conflicting persistent hidden states")
            unique[key] = candidate

        required_entries = len(unique)
        required_bytes = sum(entry.nbytes for entry in unique.values())
        if require_all and (
            required_entries > self.hidden_cache_size
            or required_bytes > self.hidden_cache_max_bytes
        ):
            gib = 1024**3
            raise RuntimeError(
                "DistilBERT hidden-cache capacity is insufficient for the precomputed prompts: "
                f"required {required_entries} entries/{required_bytes / gib:.2f} GiB, "
                f"configured {self.hidden_cache_size} entries/{self.hidden_cache_max_bytes / gib:.2f} GiB. "
                "Increase --text_hidden_cache_size and --text_hidden_cache_max_bytes."
            )
        if replace:
            self.clear_hidden_cache()
        installed = 0
        for key, entry in unique.items():
            if self._hidden_cache_put(key, entry.hidden, entry.mask, entry.token_count) is not None:
                installed += 1
        if require_all and installed != required_entries:  # pragma: no cover - defensive invariant
            raise RuntimeError("failed to install every precomputed DistilBERT hidden state")
        return {
            "required_entries": required_entries,
            "required_bytes": required_bytes,
            "installed_entries": installed,
            "installed_bytes": self.hidden_cache_current_bytes,
        }

    def encode_hidden_states(self, prompts: list[str]) -> list[CachedHiddenState]:
        """Encode and cache frozen-backbone states, never the trainable projection."""

        if not prompts:
            raise ValueError("FrozenDistilBERTEncoder requires at least one prompt")
        device = next(self.parameters()).device
        keys = [(self.model_path, self.max_length, prompt) for prompt in prompts]
        resolved: list[CachedHiddenState | None] = [None] * len(prompts)
        missing_rows: list[int] = []
        missing_lookup: dict[tuple[str, int, str], int] = {}
        for row, key in enumerate(keys):
            cached = self._hidden_cache_get(key)
            if cached is not None:
                resolved[row] = cached
            elif key not in missing_lookup:
                missing_lookup[key] = row
                missing_rows.append(row)

        if missing_rows:
            tokenized = [self._tokenize_one(prompts[row]) for row in missing_rows]
            tensors = self._padded_token_tensors([item[0] for item in tokenized], device)
            self.backbone.eval()
            backbone_autocast = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else nullcontext()
            )
            with torch.no_grad(), backbone_autocast:
                output: Any = self.backbone(**tensors)
            fresh_hidden = output.last_hidden_state
            fresh_mask = tensors["attention_mask"].bool()
            for batch_row, source_row in enumerate(missing_rows):
                encoded, token_count = tokenized[batch_row]
                length = len(encoded["input_ids"])
                key = keys[source_row]
                entry = self._hidden_cache_put(
                    key,
                    fresh_hidden[batch_row, :length],
                    fresh_mask[batch_row, :length],
                    token_count,
                )
                if entry is None:
                    hidden = fresh_hidden[batch_row, :length].detach().to("cpu", torch.bfloat16).contiguous()
                    mask = fresh_mask[batch_row, :length].detach().to("cpu", torch.bool).contiguous().clone()
                    key_bytes = len(key[0].encode("utf-8")) + len(key[2].encode("utf-8")) + 16
                    entry = CachedHiddenState(
                        hidden,
                        mask,
                        token_count,
                        self._tensor_nbytes(hidden) + self._tensor_nbytes(mask) + key_bytes + 8,
                    )
                for row, candidate in enumerate(keys):
                    if candidate == key:
                        resolved[row] = entry
        states = [entry for entry in resolved if entry is not None]
        if len(states) != len(prompts):  # pragma: no cover - defensive invariant
            raise RuntimeError("failed to encode every prompt hidden state")
        return states

    def _padded_token_tensors(
        self,
        rows: list[dict[str, list[int]]],
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        max_tokens = max(len(row["input_ids"]) for row in rows)
        input_names = set().union(*(row.keys() for row in rows))
        tensors: dict[str, torch.Tensor] = {}
        for name in input_names:
            fill = int(self.tokenizer.pad_token_id or 0) if name == "input_ids" else 0
            values = torch.full((len(rows), max_tokens), fill, dtype=torch.long, device=device)
            for row_index, encoded in enumerate(rows):
                sequence = encoded.get(name)
                if sequence is not None:
                    values[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
            tensors[name] = values
        if "attention_mask" not in tensors:
            tensors["attention_mask"] = tensors["input_ids"].ne(int(self.tokenizer.pad_token_id or 0)).long()
        return tensors

    @staticmethod
    def _autocast_enabled(device: torch.device) -> bool:
        try:
            return bool(torch.is_autocast_enabled(device.type))
        except TypeError:  # pragma: no cover - compatibility with older torch
            return bool(torch.is_autocast_enabled())

    def _tokenize_one(self, prompt: str) -> tuple[dict[str, list[int]], int]:
        key = (self.model_path, self.max_length, prompt)
        cached = self._token_cache.get(key)
        if cached is not None:
            self._token_cache.move_to_end(key)
            return cached
        full = self.tokenizer(prompt, padding=False, truncation=False, add_special_tokens=True)
        encoded = self.tokenizer(
            prompt,
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        item = ({name: list(value) for name, value in encoded.items()}, len(full["input_ids"]))
        if self.token_cache_size:
            self._token_cache[key] = item
            self._token_cache.move_to_end(key)
            while len(self._token_cache) > self.token_cache_size:
                self._token_cache.popitem(last=False)
        return item

    def forward(
        self,
        prompts: list[str],
        audit: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not prompts:
            raise ValueError("FrozenDistilBERTEncoder requires at least one prompt")
        device = next(self.parameters()).device
        resolved_states = self.encode_hidden_states(prompts)
        self.last_prompt_audit = (
            [
                {
                    "token_count": entry.token_count,
                    "token_limit": self.max_length,
                    "truncated": entry.token_count > self.max_length,
                }
                for entry in resolved_states
            ]
            if audit
            else None
        )
        max_tokens = max(entry.hidden.size(0) for entry in resolved_states)
        batch_dtype = resolved_states[0].hidden.dtype
        hidden = torch.zeros(
            len(prompts),
            max_tokens,
            self.hidden_size,
            dtype=batch_dtype,
            device=device,
        )
        mask = torch.zeros(len(prompts), max_tokens, dtype=torch.bool, device=device)
        for row, entry in enumerate(resolved_states):
            row_hidden, row_mask = entry.hidden, entry.mask
            length = row_hidden.size(0)
            hidden[row, :length] = row_hidden.to(device=device, dtype=batch_dtype, non_blocking=False)
            mask[row, :length] = row_mask.to(device=device, dtype=torch.bool, non_blocking=False)
        projection_input = hidden
        if not self._autocast_enabled(device) and projection_input.dtype != self.proj.weight.dtype:
            projection_input = projection_input.to(self.proj.weight.dtype)
        pooled = (projection_input * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        return self.proj(projection_input), self.proj(pooled), mask


class GatedSemanticFusionV2(nn.Module):
    """TimeCMA-style token similarity retrieval followed by a sample scalar gate."""

    def __init__(self, d_model: int, dropout: float = 0.1, initial_gate: float = 0.4) -> None:
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.text_out = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_model), nn.Dropout(dropout))
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(d_model * 3),
            nn.Linear(d_model * 3, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        final = self.gate_mlp[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(initial_gate / (1.0 - initial_gate)))

    def forward(
        self,
        variable_tokens: torch.Tensor,
        router_tokens: torch.Tensor,
        variable_mask: torch.Tensor,
        router_mask: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        prompt_audit: list[dict[str, int | bool]] | None = None,
    ) -> SemanticOutput:
        if text_tokens.ndim != 3 or text_mask.shape != text_tokens.shape[:2]:
            raise ValueError("text_tokens/text_mask must be [B,T,D] and [B,T]")
        if not text_mask.any(dim=1).all():
            raise ValueError("every prompt must contain at least one valid text token")
        queries = torch.cat((variable_tokens, router_tokens), dim=1)
        query_mask = torch.cat((variable_mask, router_mask), dim=1)
        projected_queries = self.q(queries)
        score = projected_queries @ self.k(text_tokens).transpose(-1, -2) / math.sqrt(queries.size(-1))
        score = score.masked_fill(~text_mask.unsqueeze(1), -1e4)
        attention = torch.softmax(score, dim=-1)
        attention = attention * text_mask.unsqueeze(1) * query_mask.unsqueeze(-1)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(1e-8)
        retrieved = attention @ self.v(text_tokens)
        query_weight = query_mask.float() / query_mask.sum(-1, keepdim=True).clamp_min(1)
        graph_context = (queries * query_weight.unsqueeze(-1)).sum(1)
        retrieved_features = self.text_out(retrieved)
        text_context = (retrieved_features * query_weight.unsqueeze(-1)).sum(1)
        gate = torch.sigmoid(self.gate_mlp(torch.cat((graph_context, text_context, graph_context * text_context), dim=-1)))
        fused = queries + gate.unsqueeze(1) * retrieved_features
        variables = variable_tokens.size(1)
        return SemanticOutput(
            variable_tokens=fused[:, :variables] * variable_mask.unsqueeze(-1),
            router_tokens=fused[:, variables:] * router_mask.unsqueeze(-1),
            context=graph_context + gate * text_context,
            gate=gate,
            cross_attention=attention,
            align_graph=(projected_queries * query_weight.unsqueeze(-1)).sum(1),
            align_text=text_context,
            prompt_audit=prompt_audit,
        )
