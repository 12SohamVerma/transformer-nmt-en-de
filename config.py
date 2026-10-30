"""
Central configuration for the Transformer NMT mini-project.

Two model configurations are provided:

  * "small" - a reduced Transformer for quick experiments on CPU / small GPUs.
  * "base"  - the exact "Transformer (base)" configuration from Table 3 of
              Vaswani et al. (2017): N=6, d_model=512, d_ff=2048, h=8,
              d_k=d_v=64, P_drop=0.1, eps_ls=0.1.

The configurations are never modified silently: the active configuration is
printed at the start of training and stored inside every checkpoint.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
TOKENIZER_DIR = ROOT_DIR / "tokenizer"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
RESULTS_DIR = ROOT_DIR / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
METRICS_DIR = RESULTS_DIR / "metrics"

SPM_PREFIX = TOKENIZER_DIR / "spm_bpe"          # -> spm_bpe.model / spm_bpe.vocab
SPM_MODEL_PATH = TOKENIZER_DIR / "spm_bpe.model"

SRC_LANG = "en"
TGT_LANG = "de"
SPLITS = ("train", "valid", "test")

# --------------------------------------------------------------------------
# Tokenizer / vocabulary
# --------------------------------------------------------------------------
# A single joint (shared) English+German BPE vocabulary, as in the paper
# ("shared source-target vocabulary", Section 5.1).
VOCAB_SIZE = 8000
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN = "<pad>", "<unk>", "<s>", "</s>"


# --------------------------------------------------------------------------
# Model / training configuration
# --------------------------------------------------------------------------
@dataclass
class ModelConfig:
    name: str
    num_encoder_layers: int      # N (encoder)
    num_decoder_layers: int      # N (decoder)
    d_model: int                 # model / embedding dimension
    num_heads: int               # h
    d_ff: int                    # inner dimension of the position-wise FFN
    dropout: float               # P_drop (residual + embedding dropout)
    max_len: int = 256           # max positions for the sinusoidal table
    share_embeddings: bool = True  # tie src emb, tgt emb and pre-softmax W
    layer_norm_eps: float = 1e-6

    @property
    def d_k(self) -> int:
        return self.d_model // self.num_heads

    @property
    def d_v(self) -> int:
        return self.d_model // self.num_heads

    def to_dict(self) -> dict:
        d = asdict(self)
        d["d_k"], d["d_v"] = self.d_k, self.d_v
        return d


@dataclass
class TrainConfig:
    batch_size: int              # sentence pairs per batch
    epochs: int
    warmup_steps: int            # Noam schedule warm-up
    lr_factor: float = 1.0       # multiplier on the Noam formula (1.0 = paper)
    label_smoothing: float = 0.1  # eps_ls
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-9
    grad_clip: float = 1.0       # max global grad-norm
    max_train_tokens: int = 100  # drop training pairs longer than this (BPE tokens)
    seed: int = 42
    patience: int = 0            # early stopping on val loss (0 = disabled)

    def to_dict(self) -> dict:
        return asdict(self)


MODEL_CONFIGS = {
    # Reduced model that trains in reasonable time on a laptop CPU.
    "small": ModelConfig(
        name="small",
        num_encoder_layers=3,
        num_decoder_layers=3,
        d_model=256,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
    ),
    # Transformer (base) - Vaswani et al. 2017, Table 3.
    "base": ModelConfig(
        name="base",
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_model=512,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
    ),
}

TRAIN_CONFIGS = {
    "small": TrainConfig(batch_size=128, epochs=15, warmup_steps=1500),
    # Paper: warmup_steps=4000, ~25k tokens/batch on 8 GPUs, 100k steps.
    "base": TrainConfig(batch_size=128, epochs=25, warmup_steps=4000),
}


def get_config(name: str) -> tuple[ModelConfig, TrainConfig]:
    if name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown config '{name}'. Choose from {list(MODEL_CONFIGS)}")
    return MODEL_CONFIGS[name], TRAIN_CONFIGS[name]


def format_config(model_cfg: ModelConfig, train_cfg: TrainConfig | None = None) -> str:
    """Pretty string used in training logs so the active config is always visible."""
    payload = {"model": model_cfg.to_dict()}
    if train_cfg is not None:
        payload["training"] = train_cfg.to_dict()
    return json.dumps(payload, indent=2)
