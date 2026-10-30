"""
Numerical encoding, padding and batching for the Multi30k corpus.

  source ids : BPE(english)          + [</s>]
  target ids : [<s>] + BPE(german)   + [</s>]

During training the decoder uses teacher forcing:
  decoder input  tgt_in  = target[:, :-1]   (<s> y1 y2 ... yn)
  decoder labels tgt_out = target[:, 1:]    (y1 y2 ... yn </s>)
"""

from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

import sentencepiece as spm
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

import config as C


_WS = re.compile(r"\s+")


def clean(text: str) -> str:
    """Unicode NFC normalisation, remove control chars, collapse whitespace. Case/punctuation kept."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in " \t")
    return _WS.sub(" ", text).strip()


def load_tokenizer(path: Path = C.SPM_MODEL_PATH) -> spm.SentencePieceProcessor:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"SentencePiece model not found at {path}. Run `python prepare_data.py` first."
        )
    return spm.SentencePieceProcessor(model_file=str(path))


def read_split(split: str) -> tuple[list[str], list[str]]:
    if split not in C.SPLITS:
        raise ValueError(f"Unknown split {split!r}")
    src_path = C.PROCESSED_DIR / f"{split}.{C.SRC_LANG}"
    tgt_path = C.PROCESSED_DIR / f"{split}.{C.TGT_LANG}"
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(f"Processed split '{split}' not found. Run `python prepare_data.py` first.")
    src = src_path.read_text(encoding="utf-8").splitlines()
    tgt = tgt_path.read_text(encoding="utf-8").splitlines()
    assert len(src) == len(tgt), f"{split}: misaligned files"
    return src, tgt


def encode_source(sp: spm.SentencePieceProcessor, text: str) -> list[int]:
    return sp.encode(text) + [C.EOS_ID]


class TranslationDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, sp, max_tokens: int | None = None):
        self.src_text, self.tgt_text, self.src, self.tgt = [], [], [], []
        for s_txt, t_txt, s_ids, t_ids in zip(src_lines, tgt_lines, sp.encode(src_lines), sp.encode(tgt_lines)):
            s_ids = s_ids + [C.EOS_ID]
            t_ids = [C.BOS_ID] + t_ids + [C.EOS_ID]
            if max_tokens is not None and (len(s_ids) > max_tokens or len(t_ids) > max_tokens):
                continue
            self.src_text.append(s_txt)
            self.tgt_text.append(t_txt)
            self.src.append(torch.tensor(s_ids, dtype=torch.long))
            self.tgt.append(torch.tensor(t_ids, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, i: int):
        return self.src[i], self.tgt[i], i


def pad_batch(seqs: list[torch.Tensor], pad_id: int = C.PAD_ID) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=pad_id)


def collate(batch):
    src, tgt, idx = zip(*batch)
    return pad_batch(list(src)), pad_batch(list(tgt)), list(idx)


class BucketBatchSampler(Sampler):
    """
    Groups sentences of similar length into the same batch (less padding = faster),
    while still shuffling the order of batches every epoch.
    """

    def __init__(self, lengths: list[int], batch_size: int, shuffle: bool, seed: int = 0):
        self.lengths, self.batch_size, self.shuffle, self.seed = lengths, batch_size, shuffle, seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        idx = list(range(len(self.lengths)))
        rng = random.Random(self.seed + self.epoch)
        if self.shuffle:
            rng.shuffle(idx)
        pool = self.batch_size * 50                       # sort inside large chunks
        batches = []
        for i in range(0, len(idx), pool):
            chunk = sorted(idx[i:i + pool], key=lambda j: self.lengths[j])
            batches += [chunk[k:k + self.batch_size] for k in range(0, len(chunk), self.batch_size)]
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        return iter(self._batches())

    def __len__(self) -> int:
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size


def make_loader(
    split: str,
    sp: spm.SentencePieceProcessor,
    batch_size: int,
    shuffle: bool,
    max_tokens: int | None = None,
    subset: int | None = None,
    seed: int = 0,
) -> DataLoader:
    src, tgt = read_split(split)
    if subset:
        src, tgt = src[:subset], tgt[:subset]
    ds = TranslationDataset(src, tgt, sp, max_tokens)
    lengths = [len(s) + len(t) for s, t in zip(ds.src, ds.tgt)]
    sampler = BucketBatchSampler(lengths, batch_size, shuffle, seed)
    return DataLoader(ds, batch_sampler=sampler, collate_fn=collate)
