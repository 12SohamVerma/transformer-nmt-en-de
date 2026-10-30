"""
Inference-time (auto-regressive) decoding.

At inference time no target sentence is available, so the decoder generates one
token at a time, feeding its own previous outputs back in (Section 3: "the model
is auto-regressive, consuming the previously generated symbols as additional
input when generating the next").

  * greedy_decode - pick argmax softmax probability at each step (batched)
  * beam_search   - keep the `beam_size` best partial hypotheses; final score is
                    normalised with the GNMT length penalty. The paper uses
                    beam size 4 and length penalty alpha = 0.6 (Section 6.1).
"""

from __future__ import annotations

import torch

from .transformer import Transformer, make_src_mask, make_tgt_mask


@torch.no_grad()
def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int | None = None,
) -> list[list[int]]:
    """src: (B, S). Returns a list of B token-id lists (without <s>/</s>)."""
    model.eval()
    pad_id = model.pad_id
    src_mask = make_src_mask(src, pad_id)
    memory = model.encode(src, src_mask)
    batch = src.size(0)
    max_len = max_len or (src.size(1) + 50)   # "maximum output length to input length + 50"

    ys = torch.full((batch, 1), bos_id, dtype=torch.long, device=src.device)
    finished = torch.zeros(batch, dtype=torch.bool, device=src.device)
    for step in range(1, max_len + 1):
        out = model.decode(ys, memory, src_mask, make_tgt_mask(ys, pad_id))
        log_probs = model.generator.log_probabilities(out[:, -1])
        if step == 1:
            log_probs[:, eos_id] = float("-inf")       # minimum output length of one token
        next_token = log_probs.argmax(dim=-1)
        next_token = next_token.masked_fill(finished, pad_id)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        finished |= next_token == eos_id
        if finished.all():
            break

    results = []
    for row in ys[:, 1:].tolist():
        tokens = []
        for t in row:
            if t in (eos_id, pad_id):
                break
            tokens.append(t)
        results.append(tokens)
    return results


def _length_penalty(length: int, alpha: float) -> float:
    return ((5.0 + length) / 6.0) ** alpha


@torch.no_grad()
def beam_search(
    model: Transformer,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_len: int | None = None,
) -> list[int]:
    """Beam search for a single sentence. src: (1, S). Returns token ids (no <s>/</s>)."""
    model.eval()
    pad_id = model.pad_id
    src_mask = make_src_mask(src, pad_id)
    memory = model.encode(src, src_mask)
    max_len = max_len or (src.size(1) + 50)

    beams = torch.full((1, 1), bos_id, dtype=torch.long, device=src.device)  # (k, t)
    beam_scores = torch.zeros(1, device=src.device)                           # sum log-probs
    finished: list[tuple[float, list[int]]] = []

    for step in range(1, max_len + 1):
        k = beams.size(0)
        out = model.decode(beams, memory.expand(k, -1, -1), src_mask.expand(k, -1, -1, -1),
                           make_tgt_mask(beams, pad_id))
        log_probs = model.generator.log_probabilities(out[:, -1])             # (k, V)
        if step == 1:
            # Forbid an empty translation: otherwise "</s>" as the first token can win
            # after length normalisation on hard inputs (same rule as fairseq's min_len=1).
            log_probs[:, eos_id] = float("-inf")
        vocab = log_probs.size(-1)
        candidates = (beam_scores.unsqueeze(1) + log_probs).view(-1)
        top_scores, top_idx = candidates.topk(min(2 * beam_size, candidates.numel()))

        new_beams, new_scores = [], []
        for score, idx in zip(top_scores.tolist(), top_idx.tolist()):
            beam_idx, token = divmod(idx, vocab)
            seq = beams[beam_idx].tolist()[1:] + [token]
            if token == eos_id:
                finished.append((score / _length_penalty(step, alpha), seq[:-1]))
            else:
                new_beams.append(beams[beam_idx].tolist() + [token])
                new_scores.append(score)
            if len(new_beams) == beam_size:
                break

        # Stop once we have enough finished hypotheses and no live beam can beat them.
        if len(finished) >= beam_size or not new_beams:
            break
        best_finished = max((s for s, _ in finished), default=float("-inf"))
        best_alive_bound = new_scores[0] / _length_penalty(max_len, alpha)
        if best_finished > best_alive_bound:
            break

        beams = torch.tensor(new_beams, dtype=torch.long, device=src.device)
        beam_scores = torch.tensor(new_scores, device=src.device)

    if not finished:  # nothing emitted </s>: fall back to the best live beam
        finished.append((beam_scores[0].item(), beams[0].tolist()[1:]))
    return max(finished, key=lambda x: x[0])[1]
