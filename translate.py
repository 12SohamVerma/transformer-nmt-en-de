"""
English -> German translation with the locally trained Transformer checkpoint.

Examples:
  python translate.py "A man is riding a bicycle down the street."
  python translate.py --greedy "Two dogs are playing in the snow."
  python translate.py                       # interactive mode
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch

import config as C
from data_utils import clean, encode_source, load_tokenizer
from models import Transformer, beam_search, count_parameters, greedy_decode


class CheckpointError(FileNotFoundError):
    pass


@dataclass
class Translation:
    source: str
    translation: str
    tokens: list[str]
    seconds: float


def default_checkpoint() -> Path:
    """Prefer the 'small' run, otherwise any *_best.pt checkpoint."""
    preferred = C.CHECKPOINT_DIR / "small_best.pt"
    if preferred.exists():
        return preferred
    found = sorted(C.CHECKPOINT_DIR.glob("*_best.pt"))
    if not found:
        raise CheckpointError(
            f"No trained checkpoint found in {C.CHECKPOINT_DIR}. Train one with `python train.py --config small`."
        )
    return found[0]


class Translator:
    def __init__(self, checkpoint: Path | str | None = None, device: str = "cpu"):
        path = Path(checkpoint) if checkpoint else default_checkpoint()
        if not path.exists():
            raise CheckpointError(f"Checkpoint not found: {path}. Train one with `python train.py`.")
        self.sp = load_tokenizer()               # raises FileNotFoundError with instructions
        self.device = torch.device(device)
        ckpt = torch.load(path, map_location=self.device, weights_only=False)

        if ckpt["vocab_size"] != self.sp.get_piece_size():
            raise ValueError(f"Tokenizer vocabulary ({self.sp.get_piece_size()}) does not match the checkpoint "
                             f"({ckpt['vocab_size']}). Re-run prepare_data.py with the same --vocab-size.")
        cfg = {k: v for k, v in ckpt["model_config"].items() if k not in ("d_k", "d_v")}
        self.model_config = C.ModelConfig(**cfg)
        self.model = Transformer.from_config(self.model_config, ckpt["vocab_size"], C.PAD_ID).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.checkpoint_path = path
        self.checkpoint_info = {k: ckpt.get(k) for k in ("epoch", "step", "val_loss", "val_nll", "train_config")}
        self.num_parameters = count_parameters(self.model)

    def _src_tensor(self, texts: list[str]) -> torch.Tensor:
        max_len = self.model_config.max_len - 1
        seqs = [torch.tensor(encode_source(self.sp, clean(t))[-max_len:]) for t in texts]
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True, padding_value=C.PAD_ID).to(self.device)

    def _max_out(self, src: torch.Tensor) -> int:
        return min(src.size(1) + 50, self.model_config.max_len - 1)

    def translate(self, text: str, beam_size: int = 4, alpha: float = 0.6) -> Translation:
        text = clean(text)
        if not text:
            raise ValueError("Input text is empty.")
        t0 = time.perf_counter()
        src = self._src_tensor([text])
        if beam_size > 1:
            ids = beam_search(self.model, src, C.BOS_ID, C.EOS_ID, beam_size, alpha, self._max_out(src))
        else:
            ids = greedy_decode(self.model, src, C.BOS_ID, C.EOS_ID, self._max_out(src))[0]
        return Translation(text, self.sp.decode(ids), [self.sp.id_to_piece(i) for i in ids],
                           time.perf_counter() - t0)

    def translate_batch_greedy(self, texts: list[str]) -> list[str]:
        src = self._src_tensor(texts)
        return [self.sp.decode(ids) for ids in greedy_decode(self.model, src, C.BOS_ID, C.EOS_ID, self._max_out(src))]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("text", nargs="*", help="English sentence(s); omit for interactive mode")
    p.add_argument("--checkpoint", help="path to a *_best.pt checkpoint")
    p.add_argument("--beam", type=int, default=4, help="beam size (paper: 4)")
    p.add_argument("--alpha", type=float, default=0.6, help="length penalty (paper: 0.6)")
    p.add_argument("--greedy", action="store_true", help="greedy decoding instead of beam search")
    args = p.parse_args()

    try:
        translator = Translator(args.checkpoint)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    beam = 1 if args.greedy else args.beam
    print(f"Loaded {translator.checkpoint_path.name} ({translator.model_config.name} config, "
          f"{translator.num_parameters:,} params) | decoding: {'greedy' if beam == 1 else f'beam={beam}'}")

    def run(sentence: str) -> None:
        out = translator.translate(sentence, beam, args.alpha)
        print(f"EN: {out.source}\nDE: {out.translation}   ({out.seconds * 1000:.0f} ms)\n")

    if args.text:
        run(" ".join(args.text))
    else:
        print("Interactive mode - type an English sentence (empty line to quit).")
        while (line := input("> ").strip()):
            run(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
