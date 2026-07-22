from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


def tokenizer_prompt_audit(tokenizer, prompts: List[str], max_length: int) -> List[Dict[str, int | bool]]:
    """Count prompts with the already-initialized tokenizer before truncation."""

    encoded = tokenizer(prompts, padding=False, truncation=False, add_special_tokens=True)
    return [
        {"token_count": len(input_ids), "token_limit": int(max_length), "truncated": len(input_ids) > max_length}
        for input_ids in encoded["input_ids"]
    ]


class HFTextEncoder(nn.Module):
    """Frozen HuggingFace encoder for report prompts.

    Supports BERT-like and GPT-like models. It returns token states, pooled states,
    and the attention mask expected by the graph-text residual runtime.
    """

    def __init__(self, model_name: str, d_model: int, freeze: bool = True, max_length: int = 192):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("Please install transformers to use HFTextEncoder.") from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.sep_token or self.tokenizer.cls_token
        self.backbone = AutoModel.from_pretrained(model_name)
        self.hidden_size = int(getattr(self.backbone.config, "hidden_size", getattr(self.backbone.config, "n_embd", 768)))
        self.proj = nn.Linear(self.hidden_size, d_model)
        self.max_length = max_length
        self.last_prompt_audit: Optional[List[Dict[str, int | bool]]] = None
        self.freeze = bool(freeze)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            self.backbone.eval()
        return self

    def forward(self, prompts: List[str], audit: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if self.freeze:
            self.backbone.eval()
        self.last_prompt_audit = tokenizer_prompt_audit(self.tokenizer, prompts, self.max_length) if audit else None
        tok = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        with torch.set_grad_enabled(any(p.requires_grad for p in self.backbone.parameters())):
            out = self.backbone(**tok)
        hidden = out.last_hidden_state
        mask = tok.get("attention_mask", torch.ones(hidden.shape[:2], device=device)).bool()
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return self.proj(hidden), self.proj(pooled), mask


class SimpleTextEncoder(nn.Module):
    """Hash-embedding fallback for smoke tests without local HuggingFace weights."""

    def __init__(self, d_model: int, vocab_size: int = 4096, max_length: int = 192):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.last_prompt_audit: Optional[List[Dict[str, int | bool]]] = None

    def _hash_tokens(self, text: str) -> List[int]:
        toks = text.lower().split()[: self.max_length]
        return [abs(hash(t)) % self.vocab_size for t in toks] or [0]

    def forward(self, prompts: List[str], audit: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        if audit:
            self.last_prompt_audit = [
                {"token_count": len(prompt.lower().split()), "token_limit": self.max_length,
                 "truncated": len(prompt.lower().split()) > self.max_length}
                for prompt in prompts
            ]
        else:
            self.last_prompt_audit = None
        ids = [self._hash_tokens(p) for p in prompts]
        max_len = min(max(len(x) for x in ids), self.max_length)
        arr = torch.zeros(len(ids), max_len, dtype=torch.long, device=device)
        mask = torch.zeros(len(ids), max_len, dtype=torch.bool, device=device)
        for i, row in enumerate(ids):
            row = row[:max_len]
            arr[i, : len(row)] = torch.tensor(row, device=device)
            mask[i, : len(row)] = True
        hidden = self.proj(self.emb(arr))
        pooled = (hidden * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return hidden, pooled, mask
