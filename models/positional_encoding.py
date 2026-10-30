"""
Sinusoidal positional encoding - Section 3.5.

    PE(pos, 2i)   = sin( pos / 10000^(2i / d_model) )
    PE(pos, 2i+1) = cos( pos / 10000^(2i / d_model) )

The encoding is added to the (scaled) token embeddings, followed by dropout
(Section 5.4: "we apply dropout to the sums of the embeddings and the
positional encodings in both the encoder and decoder stacks").
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def sinusoidal_table(max_len: int, d_model: int) -> torch.Tensor:
    """Returns a (max_len, d_model) table of fixed (non-learned) encodings."""
    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)          # (L, 1)
    # 1 / 10000^(2i/d_model) computed in log-space for numerical stability
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )                                                                          # (d_model/2,)
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
    pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
    return pe


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 256):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # Buffer: saved in the state_dict, moved with .to(device), but not trained.
        self.register_buffer("pe", sinusoidal_table(max_len, d_model).unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, d_model) scaled embeddings -> (B, L, d_model)"""
        length = x.size(1)
        if length > self.pe.size(1):
            raise ValueError(f"Sequence length {length} exceeds max_len {self.pe.size(1)}")
        return self.dropout(x + self.pe[:, :length])
