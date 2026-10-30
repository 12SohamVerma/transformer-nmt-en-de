"""
Building blocks shared by the encoder and the decoder.

  * TokenEmbedding          - Section 3.4 (embeddings multiplied by sqrt(d_model))
  * LayerNorm               - Ba et al. (2016), written out explicitly
  * PositionwiseFeedForward - Section 3.3, FFN(x) = max(0, x W1 + b1) W2 + b2   (Eq. 2)
  * SublayerConnection      - Section 3.1, LayerNorm(x + Dropout(Sublayer(x)))
                              (the paper's original *post*-layer-normalisation)
"""

from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned embedding lookup, scaled by sqrt(d_model) as in Section 3.4."""

    def __init__(self, vocab_size: int, d_model: int, padding_idx: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.scale = math.sqrt(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: (B, L) int64 -> (B, L, d_model)"""
        return self.embedding(tokens) * self.scale


class LayerNorm(nn.Module):
    """
    LayerNorm(x) = gamma * (x - mean) / sqrt(var + eps) + beta
    Statistics are computed over the feature (last) dimension of every token.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class PositionwiseFeedForward(nn.Module):
    """FFN(x) = max(0, x W1 + b1) W2 + b2, applied to every position independently."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(torch.relu(self.linear1(x)))


class SublayerConnection(nn.Module):
    """
    Residual connection + dropout + post-layer-normalisation (Section 3.1, 5.4):

        output = LayerNorm( x + Dropout( Sublayer(x) ) )
    """

    def __init__(self, d_model: int, dropout: float, eps: float = 1e-6):
        super().__init__()
        self.norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: Callable[[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        return self.norm(x + self.dropout(sublayer(x)))
