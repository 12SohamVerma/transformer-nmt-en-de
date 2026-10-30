"""
Encoder - Section 3.1.

"The encoder is composed of a stack of N = 6 identical layers. Each layer has
two sub-layers. The first is a multi-head self-attention mechanism, and the
second is a simple, position-wise fully connected feed-forward network."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection, TokenEmbedding
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, eps: float = 1e-6):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer1 = SublayerConnection(d_model, dropout, eps)
        self.sublayer2 = SublayerConnection(d_model, dropout, eps)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """x: (B, S, d_model); src_mask: (B, 1, 1, S) padding mask."""
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, src_mask))  # self-attention
        x = self.sublayer2(x, self.feed_forward)                            # FFN
        return x


class Encoder(nn.Module):
    """Embedding + positional encoding + stack of N encoder layers."""

    def __init__(
        self,
        embedding: TokenEmbedding,
        num_layers: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float,
        max_len: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.embedding = embedding
        self.positional_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout, eps) for _ in range(num_layers)]
        )

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """src: (B, S) token ids -> memory: (B, S, d_model)"""
        x = self.positional_encoding(self.embedding(src))
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
