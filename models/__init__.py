"""From-scratch Transformer (Vaswani et al., 2017) - no nn.Transformer* modules are used."""

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .decoder import Decoder, DecoderLayer
from .decoding import beam_search, greedy_decode
from .encoder import Encoder, EncoderLayer
from .layers import LayerNorm, PositionwiseFeedForward, SublayerConnection, TokenEmbedding
from .positional_encoding import PositionalEncoding, sinusoidal_table
from .transformer import (
    Generator,
    Transformer,
    count_parameters,
    make_causal_mask,
    make_src_mask,
    make_tgt_mask,
)

__all__ = [
    "MultiHeadAttention", "scaled_dot_product_attention", "Decoder", "DecoderLayer",
    "beam_search", "greedy_decode", "Encoder", "EncoderLayer", "LayerNorm",
    "PositionwiseFeedForward", "SublayerConnection", "TokenEmbedding", "PositionalEncoding",
    "sinusoidal_table", "Generator", "Transformer", "count_parameters", "make_causal_mask",
    "make_src_mask", "make_tgt_mask",
]
