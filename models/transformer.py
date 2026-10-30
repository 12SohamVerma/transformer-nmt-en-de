"""
Full Transformer encoder-decoder - Figure 1 of "Attention Is All You Need".

    src ids --Embedding*sqrt(d)+PE--> Encoder x N --> memory
    tgt ids --Embedding*sqrt(d)+PE--> Decoder x N (masked self-attn, cross-attn, FFN)
            --> Linear (d_model -> vocab) --> Softmax --> P(y_t | y_<t, x)

Weight sharing (Section 3.4): "we share the same weight matrix between the two
embedding layers and the pre-softmax linear transformation". This is enabled
by `share_embeddings=True` together with the joint EN+DE BPE vocabulary.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .decoder import Decoder
from .encoder import Encoder
from .layers import TokenEmbedding


# --------------------------------------------------------------------------
# Masks (True = attend, False = blocked)
# --------------------------------------------------------------------------
def make_src_mask(src: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Padding mask for source keys. src: (B, S) -> (B, 1, 1, S)."""
    return (src != pad_id).unsqueeze(1).unsqueeze(2)


def make_causal_mask(size: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Look-ahead mask: position i may attend to positions j <= i only.
    Returns a (1, 1, size, size) lower-triangular bool matrix.
    """
    return torch.tril(torch.ones(size, size, dtype=torch.bool, device=device)).unsqueeze(0).unsqueeze(0)


def make_tgt_mask(tgt: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Padding mask AND causal mask for decoder self-attention. tgt: (B, T) -> (B, 1, T, T)."""
    pad_mask = (tgt != pad_id).unsqueeze(1).unsqueeze(2)          # (B, 1, 1, T)
    return pad_mask & make_causal_mask(tgt.size(1), tgt.device)   # (B, 1, T, T)


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class Generator(nn.Module):
    """Final linear projection + softmax: d_model -> vocabulary distribution."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns unnormalised logits (B, T, V); CrossEntropyLoss applies log-softmax."""
        return self.proj(x)

    def probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """softmax(x W) - the token probability distribution."""
        return torch.softmax(self.proj(x), dim=-1)

    def log_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log_softmax(self.proj(x), dim=-1)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
        share_embeddings: bool = True,
        layer_norm_eps: float = 1e-6,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model

        src_embedding = TokenEmbedding(src_vocab_size, d_model, pad_id)
        if share_embeddings:
            if src_vocab_size != tgt_vocab_size:
                raise ValueError("share_embeddings requires a joint (shared) vocabulary")
            tgt_embedding = src_embedding
        else:
            tgt_embedding = TokenEmbedding(tgt_vocab_size, d_model, pad_id)

        self.encoder = Encoder(src_embedding, num_encoder_layers, d_model, num_heads, d_ff,
                               dropout, max_len, layer_norm_eps)
        self.decoder = Decoder(tgt_embedding, num_decoder_layers, d_model, num_heads, d_ff,
                               dropout, max_len, layer_norm_eps)
        self.generator = Generator(d_model, tgt_vocab_size)

        self._reset_parameters()
        if share_embeddings:
            # Tie the pre-softmax projection to the (shared) embedding matrix.
            self.generator.proj.weight = tgt_embedding.embedding.weight

    @classmethod
    def from_config(cls, cfg, vocab_size: int, pad_id: int = 0) -> "Transformer":
        return cls(
            src_vocab_size=vocab_size,
            tgt_vocab_size=vocab_size,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_encoder_layers=cfg.num_encoder_layers,
            num_decoder_layers=cfg.num_decoder_layers,
            d_ff=cfg.d_ff,
            dropout=cfg.dropout,
            max_len=cfg.max_len,
            pad_id=pad_id,
            share_embeddings=cfg.share_embeddings,
            layer_norm_eps=cfg.layer_norm_eps,
        )

    def _reset_parameters(self) -> None:
        for name, p in self.named_parameters():
            if "embedding" in name:
                # N(0, d_model^-0.5): after the sqrt(d_model) scaling the embeddings
                # have unit variance, and the tied output layer starts with small logits.
                nn.init.normal_(p, mean=0.0, std=self.d_model ** -0.5)
                with torch.no_grad():
                    p[self.pad_id].zero_()
            elif p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif name.endswith("bias"):
                nn.init.zeros_(p)

    # ---- the three stages ------------------------------------------------
    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(src, src_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, src_mask: torch.Tensor,
               tgt_mask: torch.Tensor) -> torch.Tensor:
        return self.decoder(tgt, memory, src_mask, tgt_mask)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass.
            src:    (B, S) source ids
            tgt_in: (B, T) target ids shifted right (starts with <s>)
        Returns logits (B, T, V) predicting tgt_out = target shifted left.
        """
        src_mask = make_src_mask(src, self.pad_id)
        tgt_mask = make_tgt_mask(tgt_in, self.pad_id)
        memory = self.encode(src, src_mask)
        hidden = self.decode(tgt_in, memory, src_mask, tgt_mask)
        return self.generator(hidden)


def count_parameters(model: nn.Module) -> int:
    """Trainable parameters (shared/tied tensors counted once)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
