"""
Phase 3 - Download, verify, clean and tokenise the Multi30k EN-DE corpus.

Steps
  1. Download the official Multi30k Task-1 files (Elliott et al., 2016)
     from github.com/multi30k/dataset (fallback: Hugging Face mirror bentrevett/multi30k).
  2. Verify split names, line counts, EN/DE alignment and non-empty lines.
  3. Clean text (Unicode NFC, whitespace normalisation).
  4. Train a joint EN+DE SentencePiece BPE model on the TRAINING split only.
  5. Write processed files + dataset statistics.

The test split (test_2016_flickr) is written to disk for evaluate.py only; it is
never read by the tokenizer training, train.py or validation.

Usage:  python prepare_data.py [--vocab-size 8000] [--force]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json

import requests
import sentencepiece as spm

import config as C
from data_utils import clean

GITHUB_BASE = "https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/"
GITHUB_FILES = {  # our split name -> official file stem
    "train": "train",
    "valid": "val",
    "test": "test_2016_flickr",
}
HF_BASE = "https://huggingface.co/datasets/bentrevett/multi30k/resolve/main/"
HF_FILES = {"train": "train.jsonl", "valid": "val.jsonl", "test": "test.jsonl"}
EXPECTED_SIZES = {"train": 29000, "valid": 1014, "test": 1000}


def _get(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def download_github() -> dict[str, dict[str, list[str]]]:
    data: dict[str, dict[str, list[str]]] = {}
    for split, stem in GITHUB_FILES.items():
        data[split] = {}
        for lang in (C.SRC_LANG, C.TGT_LANG):
            url = f"{GITHUB_BASE}{stem}.{lang}.gz"
            raw_path = C.RAW_DIR / f"{stem}.{lang}.gz"
            if not raw_path.exists():
                print(f"  downloading {url}")
                raw_path.write_bytes(_get(url))
            with gzip.open(raw_path, "rt", encoding="utf-8") as f:
                data[split][lang] = f.read().rstrip("\n").split("\n")
    return data


def download_hf() -> dict[str, dict[str, list[str]]]:
    data: dict[str, dict[str, list[str]]] = {}
    for split, fname in HF_FILES.items():
        raw_path = C.RAW_DIR / f"hf_{fname}"
        if not raw_path.exists():
            print(f"  downloading {HF_BASE}{fname}")
            raw_path.write_bytes(_get(HF_BASE + fname))
        rows = [json.loads(line) for line in io.StringIO(raw_path.read_text(encoding="utf-8")) if line.strip()]
        missing = {C.SRC_LANG, C.TGT_LANG} - set(rows[0])
        if missing:
            raise ValueError(f"HF schema mismatch in {fname}: missing columns {missing}; got {list(rows[0])}")
        data[split] = {lang: [r[lang] for r in rows] for lang in (C.SRC_LANG, C.TGT_LANG)}
    return data


def verify(data: dict[str, dict[str, list[str]]]) -> None:
    """Checks split names, EN/DE alignment, expected sizes and empty lines."""
    if set(data) != set(C.SPLITS):
        raise ValueError(f"Unexpected split names {sorted(data)}; expected {C.SPLITS}")
    for split, langs in data.items():
        n_src, n_tgt = len(langs[C.SRC_LANG]), len(langs[C.TGT_LANG])
        if n_src != n_tgt:
            raise ValueError(f"{split}: EN has {n_src} lines but DE has {n_tgt}")
        if n_src != EXPECTED_SIZES[split]:
            print(f"  WARNING: {split} has {n_src} pairs (expected {EXPECTED_SIZES[split]})")
        empty = sum(1 for e, d in zip(langs[C.SRC_LANG], langs[C.TGT_LANG]) if not e.strip() or not d.strip())
        print(f"  [ok] {split:5s}: {n_src:6d} aligned pairs, {empty} empty")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab-size", type=int, default=C.VOCAB_SIZE)
    parser.add_argument("--force", action="store_true", help="re-download and re-train tokenizer")
    args = parser.parse_args()

    for d in (C.RAW_DIR, C.PROCESSED_DIR, C.TOKENIZER_DIR):
        d.mkdir(parents=True, exist_ok=True)
    if args.force:
        for f in C.RAW_DIR.glob("*"):
            f.unlink()

    print("[1/4] Downloading Multi30k (Task 1, EN-DE)")
    try:
        data, source = download_github(), "github.com/multi30k/dataset (official)"
    except Exception as exc:  # network / format problem -> mirror
        print(f"  GitHub download failed ({exc}); falling back to Hugging Face mirror")
        data, source = download_hf(), "huggingface.co/datasets/bentrevett/multi30k"

    print("[2/4] Verifying splits and schema")
    verify(data)

    print("[3/4] Cleaning text and writing processed splits")
    stats: dict = {"source": source, "splits": {}}
    for split in C.SPLITS:
        pairs = [(clean(e), clean(d)) for e, d in zip(data[split][C.SRC_LANG], data[split][C.TGT_LANG])]
        dropped = sum(1 for e, d in pairs if not e or not d)
        pairs = [(e, d) for e, d in pairs if e and d]
        for i, lang in enumerate((C.SRC_LANG, C.TGT_LANG)):
            (C.PROCESSED_DIR / f"{split}.{lang}").write_text("\n".join(p[i] for p in pairs) + "\n", encoding="utf-8")
        words = [(len(e.split()), len(d.split())) for e, d in pairs]
        stats["splits"][split] = {
            "pairs": len(pairs),
            "dropped_empty": dropped,
            "avg_words_en": round(sum(w[0] for w in words) / len(words), 2),
            "avg_words_de": round(sum(w[1] for w in words) / len(words), 2),
            "max_words_en": max(w[0] for w in words),
            "max_words_de": max(w[1] for w in words),
        }
        print(f"  {split:5s}: {len(pairs)} pairs  e.g. EN: {pairs[0][0]!r}\n{'':14s}DE: {pairs[0][1]!r}")

    train_src = set((C.PROCESSED_DIR / "train.en").read_text(encoding="utf-8").splitlines())
    test_src = (C.PROCESSED_DIR / "test.en").read_text(encoding="utf-8").splitlines()
    stats["test_sentences_also_in_train_en"] = sum(1 for s in test_src if s in train_src)

    if C.SPM_MODEL_PATH.exists() and not args.force:
        # Never silently replace the tokenizer: trained checkpoints depend on its exact vocabulary.
        print(f"[4/4] Keeping existing tokenizer {C.SPM_MODEL_PATH.name} (use --force to retrain)")
    else:
        print(f"[4/4] Training joint SentencePiece BPE (vocab={args.vocab_size}) on TRAIN split only")
        spm.SentencePieceTrainer.train(
            input=[str(C.PROCESSED_DIR / "train.en"), str(C.PROCESSED_DIR / "train.de")],
            model_prefix=str(C.SPM_PREFIX),
            vocab_size=args.vocab_size,
            model_type="bpe",
            character_coverage=1.0,
            pad_id=C.PAD_ID, unk_id=C.UNK_ID, bos_id=C.BOS_ID, eos_id=C.EOS_ID,
            pad_piece=C.PAD_TOKEN, unk_piece=C.UNK_TOKEN, bos_piece=C.BOS_TOKEN, eos_piece=C.EOS_TOKEN,
            input_sentence_size=1_000_000, shuffle_input_sentence=True, seed_sentencepiece_size=1_000_000,
            minloglevel=2,
        )
    sp = spm.SentencePieceProcessor(model_file=str(C.SPM_MODEL_PATH))
    assert [sp.id_to_piece(i) for i in range(4)] == [C.PAD_TOKEN, C.UNK_TOKEN, C.BOS_TOKEN, C.EOS_TOKEN]

    for split in C.SPLITS:
        for lang in (C.SRC_LANG, C.TGT_LANG):
            lines = (C.PROCESSED_DIR / f"{split}.{lang}").read_text(encoding="utf-8").splitlines()
            lens = [len(ids) for ids in sp.encode(lines)]
            unk = sum(ids.count(C.UNK_ID) for ids in sp.encode(lines))
            stats["splits"][split][f"avg_bpe_tokens_{lang}"] = round(sum(lens) / len(lens), 2)
            stats["splits"][split][f"max_bpe_tokens_{lang}"] = max(lens)
            stats["splits"][split][f"unk_tokens_{lang}"] = unk
    stats["vocab_size"] = sp.get_piece_size()
    stats["special_tokens"] = {C.PAD_TOKEN: C.PAD_ID, C.UNK_TOKEN: C.UNK_ID, C.BOS_TOKEN: C.BOS_ID, C.EOS_TOKEN: C.EOS_ID}
    example = (C.PROCESSED_DIR / "train.en").read_text(encoding="utf-8").splitlines()[0]
    stats["tokenization_example"] = {"text": example, "pieces": sp.encode(example, out_type=str),
                                     "ids": sp.encode(example)}

    (C.PROCESSED_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"\nTokenizer saved to {C.SPM_MODEL_PATH}")


if __name__ == "__main__":
    main()
