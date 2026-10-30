"""
Phase 5 - Evaluate a trained checkpoint on the held-out Multi30k test set (test_2016_flickr).

Reports
  * corpus BLEU (SacreBLEU, detokenised output vs raw references) - greedy and beam search
  * training loss (from the training history), validation loss, test loss (+ NLL / perplexity)
  * inference time and number of trainable parameters
Writes
  results/metrics/<tag>_metrics.json
  results/predictions/<tag>_test_predictions.csv     (every test sentence)
  results/predictions/<tag>_sample_translations.md   (readable samples)
  results/graphs/<tag>_loss_curve.png, <tag>_cross_attention.png

NOTE: these are Multi30k results. They are NOT comparable to the WMT 2014 numbers
reported in the paper (different data size, domain, and test set).

Usage:  python evaluate.py [--checkpoint checkpoints/small_best.pt] [--beam 4] [--alpha 0.6]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import sacrebleu  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import config as C  # noqa: E402
from data_utils import make_loader, read_split  # noqa: E402
from plotting import plot_loss_curves  # noqa: E402
from train import run_epoch  # noqa: E402
from translate import Translator, default_checkpoint  # noqa: E402

PAPER_REFERENCE = {
    "note": "Reported in Vaswani et al. (2017), Table 2, on WMT 2014 EN-DE (newstest2014, ~4.5M training "
            "pairs, 8x P100 GPUs). Shown for context only - NOT reproduced and NOT comparable to Multi30k.",
    "transformer_base_bleu_wmt14_en_de": 27.3,
    "transformer_big_bleu_wmt14_en_de": 28.4,
}


def plot_cross_attention(translator: Translator, sentence: str, path: Path) -> str:
    """Heatmap of the last decoder layer's encoder-decoder attention (averaged over heads)."""
    model, sp = translator.model, translator.sp
    out = translator.translate(sentence, beam_size=4)
    src = translator._src_tensor([sentence])
    ids = sp.encode(out.translation)
    tgt_in = torch.tensor([[C.BOS_ID] + ids])
    with torch.no_grad():
        model(src, tgt_in)
    attn = model.decoder.layers[-1].cross_attn.attn_weights[0].mean(0).numpy()   # (T, S)
    src_tokens = [sp.id_to_piece(i) for i in src[0].tolist()]
    tgt_tokens = ["<s>"] + [sp.id_to_piece(i) for i in ids]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(src_tokens)), max(4, 0.42 * len(tgt_tokens))), dpi=150)
    im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(src_tokens)), src_tokens, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tgt_tokens)), tgt_tokens, fontsize=8)
    ax.set_xlabel("Source (English) BPE tokens = keys")
    ax.set_ylabel("Decoder input tokens = queries")
    ax.set_title("Encoder–decoder attention, last decoder layer (mean over heads)", fontsize=10, loc="left")
    fig.colorbar(im, ax=ax, fraction=0.03, label="attention weight")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return out.translation


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint")
    p.add_argument("--beam", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--limit", type=int, help="evaluate only the first N test sentences (debug)")
    args = p.parse_args()

    ckpt_path = (Path(args.checkpoint) if args.checkpoint else default_checkpoint()).resolve()
    tag = ckpt_path.stem.replace("_best", "")
    translator = Translator(ckpt_path)
    model, sp = translator.model, translator.sp
    train_cfg = translator.checkpoint_info["train_config"]
    print(f"Checkpoint: {ckpt_path}  (epoch {translator.checkpoint_info['epoch']}, "
          f"{translator.num_parameters:,} params, config '{translator.model_config.name}')")

    # ---------------- losses ----------------
    criterion = nn.CrossEntropyLoss(ignore_index=C.PAD_ID, label_smoothing=train_cfg["label_smoothing"])
    device = translator.device
    val_loss, val_nll, _ = run_epoch(model, make_loader("valid", sp, args.batch_size, False), criterion, device)
    test_loss, test_nll, _ = run_epoch(model, make_loader("test", sp, args.batch_size, False), criterion, device)
    print(f"Validation loss {val_loss:.4f} (NLL {val_nll:.4f}, PPL {math.exp(val_nll):.2f})")
    print(f"Test loss       {test_loss:.4f} (NLL {test_nll:.4f}, PPL {math.exp(test_nll):.2f})")

    history_path = C.METRICS_DIR / f"{tag}_history.json"
    history = json.loads(history_path.read_text()) if history_path.exists() else None
    train_loss_at_best = None
    if history:
        h = history["history"]
        best_i = h["epoch"].index(translator.checkpoint_info["epoch"])
        train_loss_at_best = h["train_loss"][best_i]
        plot_loss_curves(h, C.GRAPHS_DIR / f"{tag}_loss_curve.png",
                         f"Transformer ({translator.model_config.name}) on Multi30k EN→DE: "
                         "training vs validation loss")

    # ---------------- translations ----------------
    src_texts, refs = read_split("test")
    if args.limit:
        src_texts, refs = src_texts[:args.limit], refs[:args.limit]

    t0 = time.perf_counter()
    greedy = []
    for i in range(0, len(src_texts), args.batch_size):
        greedy += translator.translate_batch_greedy(src_texts[i:i + args.batch_size])
    greedy_time = time.perf_counter() - t0
    print(f"Greedy decoding: {len(src_texts)} sentences in {greedy_time:.1f}s")

    t0 = time.perf_counter()
    beam = []
    for i, s in enumerate(src_texts):
        beam.append(translator.translate(s, args.beam, args.alpha).translation)
        if (i + 1) % 200 == 0:
            print(f"  beam search {i + 1}/{len(src_texts)}", flush=True)
    beam_time = time.perf_counter() - t0
    print(f"Beam search (k={args.beam}, alpha={args.alpha}): {beam_time:.1f}s")

    # single-sentence latency (what a user of the demo experiences)
    latencies = [translator.translate(s, 1).seconds for s in src_texts[:50]]

    bleu_metric = sacrebleu.metrics.BLEU()
    bleu_greedy = bleu_metric.corpus_score(greedy, [refs])
    bleu_beam = bleu_metric.corpus_score(beam, [refs])
    chrf_beam = sacrebleu.metrics.CHRF().corpus_score(beam, [refs])
    signature = str(bleu_metric.get_signature())
    print(f"BLEU greedy: {bleu_greedy.score:.2f}   BLEU beam: {bleu_beam.score:.2f}   chrF beam: {chrf_beam.score:.2f}")
    print(f"SacreBLEU signature: {signature}")

    metrics = {
        "dataset": "Multi30k Task 1 EN->DE, test_2016_flickr (1000 sentences)",
        "checkpoint": str(ckpt_path.relative_to(C.ROOT_DIR)),
        "model_config": translator.model_config.to_dict(),
        "train_config": train_cfg,
        "best_epoch": translator.checkpoint_info["epoch"],
        "num_trainable_parameters": translator.num_parameters,
        "evaluated_sentences": len(src_texts),
        "losses": {
            "note": "loss = label-smoothed cross-entropy per target token (eps_ls=0.1); "
                    "nll = un-smoothed cross-entropy; ppl = exp(nll)",
            "train_loss_at_best_epoch": train_loss_at_best,
            "val_loss": val_loss, "val_nll": val_nll, "val_ppl": math.exp(val_nll),
            "test_loss": test_loss, "test_nll": test_nll, "test_ppl": math.exp(test_nll),
        },
        "bleu": {
            "greedy": round(bleu_greedy.score, 2),
            f"beam{args.beam}_alpha{args.alpha}": round(bleu_beam.score, 2),
            "beam_detail": str(bleu_beam),
            "chrf_beam": round(chrf_beam.score, 2),
            "sacrebleu_signature": signature,
        },
        "inference_time": {
            "device": str(device),
            "greedy_total_sec": round(greedy_time, 2),
            "greedy_ms_per_sentence_batched": round(1000 * greedy_time / len(src_texts), 2),
            "beam_total_sec": round(beam_time, 2),
            "beam_ms_per_sentence": round(1000 * beam_time / len(src_texts), 2),
            "single_sentence_greedy_latency_ms_mean": round(1000 * sum(latencies) / len(latencies), 2),
        },
        "paper_reference_not_comparable": PAPER_REFERENCE,
    }
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (C.METRICS_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False),
                                                         encoding="utf-8")

    C.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.PREDICTIONS_DIR / f"{tag}_test_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "english_source", "german_reference", "model_greedy", "model_beam", "sentence_bleu_beam"])
        for i, (s, r, g, b) in enumerate(zip(src_texts, refs, greedy, beam)):
            w.writerow([i, s, r, g, b, round(sacrebleu.sentence_bleu(b, [r]).score, 2)])

    lines = [f"# Sample test-set translations ({tag} model, beam={args.beam})\n",
             "First 25 sentences of Multi30k test_2016_flickr - not cherry-picked.\n"]
    for i in range(min(25, len(src_texts))):
        lines += [f"**{i + 1}.** EN: {src_texts[i]}  ", f"REF (DE): {refs[i]}  ",
                  f"MODEL (DE): {beam[i]}\n"]
    (C.PREDICTIONS_DIR / f"{tag}_sample_translations.md").write_text("\n".join(lines), encoding="utf-8")

    attn_example = "A man in a blue shirt is standing on a ladder."
    plot_cross_attention(translator, attn_example, C.GRAPHS_DIR / f"{tag}_cross_attention.png")

    print(json.dumps({k: metrics[k] for k in ("bleu", "losses", "inference_time")}, indent=2))
    print("\nFirst 5 test translations:")
    for i in range(min(5, len(src_texts))):
        print(f"EN : {src_texts[i]}\nREF: {refs[i]}\nHYP: {beam[i]}\n")


if __name__ == "__main__":
    main()
