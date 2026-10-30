"""
Attention mechanisms - Section 3.2 of "Attention Is All You Need".

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V                    (Eq. 1)

    MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i   = Attention(Q W_i^Q, K W_i^K, V W_i^V)

The same MultiHeadAttention module is used in three ways (Section 3.2.3):
  1. encoder self-attention          (Q = K = V = encoder states, padding mask)
  2. masked decoder self-attention   (Q = K = V = decoder states, padding + causal mask)
  3. encoder-decoder cross-attention (Q = decoder states, K = V = encoder output)

Mask convention used throughout the project:
    mask is a bool tensor broadcastable to (batch, heads, len_q, len_k)
    True  -> position may be attended to
    False -> position is blocked (score set to a very large negative number)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        query: (..., len_q, d_k)
        key:   (..., len_k, d_k)
        value: (..., len_k, d_v)
        mask:  bool, broadcastable to (..., len_q, len_k); True = keep.
    Returns:
        output:  (..., len_q, d_v)
        weights: (..., len_q, len_k) - the attention distribution (rows sum to 1)
    """
    d_k = query.size(-1)
    # Q K^T / sqrt(d_k): scaling keeps the dot products from growing with d_k,
    # which would otherwise push softmax into regions with tiny gradients.
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        # Blocked positions get the smallest finite value -> softmax weight ~0.
        # (A finite value avoids NaNs that -inf would cause on fully-masked rows.)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, value), weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with h parallel heads of size d_k = d_v = d_model / h.

    The h per-head projections W_i^Q (d_model x d_k) are stored together as a
    single d_model x (h*d_k) linear layer, which is mathematically identical to
    h separate projections and much faster.
    """

    def __init__(self, d_model: int, num_heads: int, attn_dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model)   # W^Q for all heads
        self.w_k = nn.Linear(d_model, d_model)   # W^K for all heads
        self.w_v = nn.Linear(d_model, d_model)   # W^V for all heads
        self.w_o = nn.Linear(d_model, d_model)   # W^O
        self.dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else None

        # Last attention map, kept for visualisation / testing only.
        self.attn_weights: torch.Tensor | None = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, L, d_model) -> (B, h, L, d_k)
        b, length, _ = x.shape
        return x.view(b, length, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, h, L, d_k) -> (B, L, d_model)  ==  Concat(head_1..head_h)
        b, _, length, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, length, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, len_q, d_model)
            key, value: (B, len_k, d_model)
            mask: bool, broadcastable to (B, h, len_q, len_k)
        Returns:
            (B, len_q, d_model)
        """
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        out, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        self.attn_weights = weights.detach()
        return self.w_o(self._merge_heads(out))
