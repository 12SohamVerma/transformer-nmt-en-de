"""
Decoder - Section 3.1.

"In addition to the two sub-layers in each encoder layer, the decoder inserts a
third sub-layer, which performs multi-head attention over the output of the
encoder stack. ... We also modify the self-attention sub-layer in the decoder
stack to prevent positions from attending to subsequent positions."
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .layers import PositionwiseFeedForward, SublayerConnection, TokenEmbedding
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float, eps: float = 1e-6):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)    # masked self-attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads)   # encoder-decoder attention
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff)
        self.sublayer1 = SublayerConnection(d_model, dropout, eps)
        self.sublayer2 = SublayerConnection(d_model, dropout, eps)
        self.sublayer3 = SublayerConnection(d_model, dropout, eps)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        x:        (B, T, d_model) decoder states
        memory:   (B, S, d_model) encoder output
        src_mask: (B, 1, 1, S)    padding mask over source keys
        tgt_mask: (B, 1, T, T)    padding & causal (look-ahead) mask
        """
        # 1) Masked multi-head self-attention: Q = K = V = decoder states
        x = self.sublayer1(x, lambda t: self.self_attn(t, t, t, tgt_mask))
        # 2) Cross-attention: Q from the decoder, K = V = encoder memory
        x = self.sublayer2(x, lambda t: self.cross_attn(t, memory, memory, src_mask))
        # 3) Position-wise feed-forward network
        x = self.sublayer3(x, self.feed_forward)
        return x


class Decoder(nn.Module):
    """Embedding + positional encoding + stack of N decoder layers."""

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
            [DecoderLayer(d_model, num_heads, d_ff, dropout, eps) for _ in range(num_layers)]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """tgt: (B, T) shifted-right target ids -> (B, T, d_model)"""
        x = self.positional_encoding(self.embedding(tgt))
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x
