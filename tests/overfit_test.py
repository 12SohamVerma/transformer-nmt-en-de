"""
Overfit sanity check: a correct Transformer must be able to memorise a tiny
subset of real Multi30k training pairs. If the loss does not go to ~0 and the
memorised sentences cannot be reproduced, something in the architecture,
masking or training loop is broken.

Run:  python tests/overfit_test.py
Writes results/metrics/overfit_test.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as C  # noqa: E402
from data_utils import collate, load_tokenizer, read_split, TranslationDataset  # noqa: E402
from models import Transformer, greedy_decode  # noqa: E402
from train import noam_lr  # noqa: E402

N_PAIRS, STEPS = 32, 500


def main() -> int:
    torch.manual_seed(0)
    sp = load_tokenizer()
    src, tgt = read_split("train")
    ds = TranslationDataset(src[:N_PAIRS], tgt[:N_PAIRS], sp)
    src_b, tgt_b, _ = collate([ds[i] for i in range(len(ds))])

    model = Transformer(sp.get_piece_size(), sp.get_piece_size(), d_model=128, num_heads=4,
                        num_encoder_layers=2, num_decoder_layers=2, d_ff=512, dropout=0.0, pad_id=C.PAD_ID)
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda k: noam_lr(k + 1, 128, 100, factor=0.5))
    crit = nn.CrossEntropyLoss(ignore_index=C.PAD_ID)   # no smoothing so loss can reach ~0

    t0, losses = time.time(), []
    for step in range(1, STEPS + 1):
        model.train()
        logits = model(src_b, tgt_b[:, :-1])
        loss = crit(logits.reshape(-1, logits.size(-1)), tgt_b[:, 1:].reshape(-1))
        assert torch.isfinite(loss), "non-finite loss"
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    hyps = [sp.decode(h) for h in greedy_decode(model, src_b, C.BOS_ID, C.EOS_ID)]
    exact = sum(h == r for h, r in zip(hyps, ds.tgt_text))
    for i in range(3):
        print(f"\nEN : {ds.src_text[i]}\nREF: {ds.tgt_text[i]}\nHYP: {hyps[i]}")

    passed = losses[-1] < 0.05 and exact >= 0.9 * len(ds)
    result = {"pairs": len(ds), "steps": STEPS, "initial_loss": losses[0], "final_loss": losses[-1],
              "exact_match_reproductions": exact, "seconds": round(time.time() - t0, 1), "passed": passed}
    print("\n" + json.dumps(result, indent=2))
    C.METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (C.METRICS_DIR / "overfit_test.json").write_text(json.dumps(result, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
