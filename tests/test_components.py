"""
Unit tests for every Transformer component.

Run:  python -m unittest discover -s tests -v
"""

from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import MODEL_CONFIGS  # noqa: E402
from models import (  # noqa: E402
    Decoder,
    Encoder,
    LayerNorm,
    MultiHeadAttention,
    PositionalEncoding,
    PositionwiseFeedForward,
    TokenEmbedding,
    Transformer,
    beam_search,
    count_parameters,
    greedy_decode,
    make_causal_mask,
    make_src_mask,
    make_tgt_mask,
    scaled_dot_product_attention,
    sinusoidal_table,
)

PAD, BOS, EOS = 0, 2, 3
B, S, T, D, H, V = 2, 7, 5, 32, 4, 50


def tiny_model(**kw) -> Transformer:
    args = dict(src_vocab_size=V, tgt_vocab_size=V, d_model=D, num_heads=H, num_encoder_layers=2,
                num_decoder_layers=2, d_ff=64, dropout=0.1, max_len=64, pad_id=PAD)
    args.update(kw)
    return Transformer(**args)


class TestPositionalEncoding(unittest.TestCase):
    def test_shape_and_formula(self):
        pe = sinusoidal_table(50, D)
        self.assertEqual(pe.shape, (50, D))
        pos, i = 7, 3
        self.assertAlmostEqual(pe[pos, 2 * i].item(), math.sin(pos / 10000 ** (2 * i / D)), places=5)
        self.assertAlmostEqual(pe[pos, 2 * i + 1].item(), math.cos(pos / 10000 ** (2 * i / D)), places=5)
        self.assertTrue(torch.all(pe[0, 0::2] == 0) and torch.all(pe[0, 1::2] == 1))

    def test_module_adds_encoding(self):
        module = PositionalEncoding(D, dropout=0.0, max_len=50).eval()
        x = torch.zeros(B, 10, D)
        out = module(x)
        self.assertEqual(out.shape, (B, 10, D))
        self.assertTrue(torch.allclose(out[0], sinusoidal_table(50, D)[:10]))


class TestAttention(unittest.TestCase):
    def test_scaled_dot_product_shapes_and_rows_sum_to_one(self):
        q, k, v = torch.randn(B, H, T, 8), torch.randn(B, H, S, 8), torch.randn(B, H, S, 16)
        out, w = scaled_dot_product_attention(q, k, v)
        self.assertEqual(out.shape, (B, H, T, 16))
        self.assertEqual(w.shape, (B, H, T, S))
        self.assertTrue(torch.allclose(w.sum(-1), torch.ones(B, H, T), atol=1e-6))

    def test_matches_manual_formula(self):
        q, k, v = torch.randn(1, 3, 4), torch.randn(1, 5, 4), torch.randn(1, 5, 6)
        out, _ = scaled_dot_product_attention(q, k, v)
        expected = torch.softmax(q @ k.transpose(-2, -1) / 2.0, dim=-1) @ v   # sqrt(4) = 2
        self.assertTrue(torch.allclose(out, expected, atol=1e-6))

    def test_multi_head_shapes(self):
        mha = MultiHeadAttention(D, H)
        x, mem = torch.randn(B, T, D), torch.randn(B, S, D)
        self.assertEqual(mha(x, x, x).shape, (B, T, D))            # self-attention
        self.assertEqual(mha(x, mem, mem).shape, (B, T, D))        # cross-attention
        self.assertEqual(mha.attn_weights.shape, (B, H, T, S))
        self.assertEqual(mha.d_k, D // H)

    def test_multi_head_equals_per_head_computation(self):
        """Batched implementation == explicit Concat(head_1..head_h) W^O."""
        torch.manual_seed(0)
        mha = MultiHeadAttention(D, H).eval()
        x = torch.randn(1, T, D)
        dk = D // H
        heads = []
        for i in range(H):
            sl = slice(i * dk, (i + 1) * dk)
            qi = x @ mha.w_q.weight[sl].T + mha.w_q.bias[sl]
            ki = x @ mha.w_k.weight[sl].T + mha.w_k.bias[sl]
            vi = x @ mha.w_v.weight[sl].T + mha.w_v.bias[sl]
            heads.append(torch.softmax(qi @ ki.transpose(-2, -1) / math.sqrt(dk), -1) @ vi)
        expected = mha.w_o(torch.cat(heads, dim=-1))
        self.assertTrue(torch.allclose(mha(x, x, x), expected, atol=1e-5))

    def test_requires_divisible_heads(self):
        with self.assertRaises(ValueError):
            MultiHeadAttention(30, 4)


class TestMasks(unittest.TestCase):
    def test_padding_mask(self):
        src = torch.tensor([[5, 6, 7, PAD, PAD], [5, 6, 7, 8, 9]])
        mask = make_src_mask(src, PAD)
        self.assertEqual(mask.shape, (2, 1, 1, 5))
        self.assertEqual(mask[0, 0, 0].tolist(), [True, True, True, False, False])
        self.assertTrue(mask[1].all())

    def test_padded_keys_get_zero_attention(self):
        src = torch.tensor([[5, 6, 7, PAD, PAD]])
        mask = make_src_mask(src, PAD)
        q, k, v = torch.randn(1, 1, 3, 8), torch.randn(1, 1, 5, 8), torch.randn(1, 1, 5, 8)
        _, w = scaled_dot_product_attention(q, k, v, mask)
        self.assertTrue(torch.all(w[..., 3:] == 0))

    def test_causal_mask_is_lower_triangular(self):
        m = make_causal_mask(4)[0, 0]
        expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        self.assertTrue(torch.equal(m, expected))

    def test_tgt_mask_combines_padding_and_causal(self):
        tgt = torch.tensor([[BOS, 5, 6, PAD]])
        m = make_tgt_mask(tgt, PAD)[0, 0]
        self.assertEqual(m.shape, (4, 4))
        self.assertFalse(m[:, 3].any())           # padded key never attended
        self.assertFalse(m[0, 1])                 # no look-ahead
        self.assertTrue(m[2, :3].all())

    def test_causal_mask_blocks_future_information(self):
        """Changing a future target token must not change earlier decoder outputs."""
        torch.manual_seed(0)
        model = tiny_model().eval()
        src = torch.randint(4, V, (1, S))
        tgt = torch.randint(4, V, (1, T))
        tgt2 = tgt.clone()
        tgt2[0, -1] = (tgt[0, -1] + 1 - 4) % (V - 4) + 4    # change only the last token
        out1, out2 = model(src, tgt), model(src, tgt2)
        self.assertTrue(torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5))
        self.assertFalse(torch.allclose(out1[:, -1], out2[:, -1], atol=1e-5))

    def test_source_padding_does_not_change_output(self):
        """Appending PAD tokens to the source must not change the translation logits."""
        torch.manual_seed(0)
        model = tiny_model().eval()
        src = torch.randint(4, V, (1, S))
        src_padded = torch.cat([src, torch.full((1, 3), PAD)], dim=1)
        tgt = torch.randint(4, V, (1, T))
        self.assertTrue(torch.allclose(model(src, tgt), model(src_padded, tgt), atol=1e-5))


class TestLayers(unittest.TestCase):
    def test_layer_norm_matches_pytorch(self):
        x = torch.randn(B, T, D)
        ours, ref = LayerNorm(D, eps=1e-5), nn.LayerNorm(D, eps=1e-5)
        self.assertTrue(torch.allclose(ours(x), ref(x), atol=1e-5))

    def test_ffn_shape(self):
        self.assertEqual(PositionwiseFeedForward(D, 64)(torch.randn(B, T, D)).shape, (B, T, D))

    def test_embedding_scaled(self):
        emb = TokenEmbedding(V, D, PAD)
        ids = torch.tensor([[5]])
        self.assertTrue(torch.allclose(emb(ids), emb.embedding.weight[5] * math.sqrt(D)))


class TestEncoderDecoder(unittest.TestCase):
    def test_encoder_forward(self):
        enc = Encoder(TokenEmbedding(V, D, PAD), 2, D, H, 64, 0.1, 64)
        src = torch.randint(4, V, (B, S))
        self.assertEqual(enc(src, make_src_mask(src, PAD)).shape, (B, S, D))

    def test_decoder_forward(self):
        dec = Decoder(TokenEmbedding(V, D, PAD), 2, D, H, 64, 0.1, 64)
        src, tgt = torch.randint(4, V, (B, S)), torch.randint(4, V, (B, T))
        out = dec(tgt, torch.randn(B, S, D), make_src_mask(src, PAD), make_tgt_mask(tgt, PAD))
        self.assertEqual(out.shape, (B, T, D))

    def test_full_model_output_vocab_dim_and_softmax(self):
        model = tiny_model()
        src, tgt = torch.randint(4, V, (B, S)), torch.randint(4, V, (B, T))
        logits = model(src, tgt)
        self.assertEqual(logits.shape, (B, T, V))
        probs = torch.softmax(logits, -1)
        self.assertTrue(torch.allclose(probs.sum(-1), torch.ones(B, T), atol=1e-5))

    def test_weight_tying(self):
        model = tiny_model()
        self.assertIs(model.encoder.embedding, model.decoder.embedding)
        self.assertIs(model.generator.proj.weight, model.encoder.embedding.embedding.weight)

    def test_base_config_matches_paper(self):
        cfg = MODEL_CONFIGS["base"]
        self.assertEqual((cfg.num_encoder_layers, cfg.num_decoder_layers, cfg.d_model, cfg.num_heads,
                          cfg.d_ff, cfg.d_k, cfg.d_v, cfg.dropout), (6, 6, 512, 8, 2048, 64, 64, 0.1))
        # With the paper's 37k shared vocabulary the base model has ~65M parameters (Table 3).
        n = count_parameters(Transformer.from_config(cfg, vocab_size=37000))
        self.assertTrue(60e6 < n < 66e6, n)

    def test_no_builtin_transformer_modules(self):
        model = Transformer.from_config(MODEL_CONFIGS["base"], vocab_size=100)
        forbidden = (nn.Transformer, nn.TransformerEncoder, nn.TransformerDecoder,
                     nn.TransformerEncoderLayer, nn.TransformerDecoderLayer, nn.MultiheadAttention)
        self.assertFalse(any(isinstance(m, forbidden) for m in model.modules()))


class TestTrainingStep(unittest.TestCase):
    def test_loss_and_backward_are_finite(self):
        torch.manual_seed(0)
        model = tiny_model()
        crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
        src = torch.randint(4, V, (B, S))
        tgt = torch.cat([torch.full((B, 1), BOS), torch.randint(4, V, (B, T)), torch.full((B, 1), EOS)], 1)
        logits = model(src, tgt[:, :-1])
        loss = crit(logits.reshape(-1, V), tgt[:, 1:].reshape(-1))
        self.assertTrue(torch.isfinite(loss))
        # untrained model: loss should be close to log(V)
        self.assertLess(abs(loss.item() - math.log(V)), 1.5)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_pad_targets_are_ignored(self):
        crit = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
        logits = torch.randn(1, 3, V)
        a = crit(logits[:, :2].reshape(-1, V), torch.tensor([5, 6]))
        b = crit(logits.reshape(-1, V), torch.tensor([5, 6, PAD]))
        self.assertTrue(torch.allclose(a, b))

    def test_checkpoint_save_load_roundtrip(self):
        torch.manual_seed(0)
        model = tiny_model().eval()
        src, tgt = torch.randint(4, V, (1, S)), torch.randint(4, V, (1, T))
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "ckpt.pt"
            torch.save({"model_state": model.state_dict()}, path)
            model2 = tiny_model().eval()
            model2.load_state_dict(torch.load(path, map_location="cpu")["model_state"])
        self.assertTrue(torch.allclose(model(src, tgt), model2(src, tgt)))
        self.assertIs(model2.generator.proj.weight, model2.encoder.embedding.embedding.weight)


class TestDecoding(unittest.TestCase):
    def test_greedy_and_beam_run(self):
        torch.manual_seed(0)
        model = tiny_model().eval()
        src = torch.randint(4, V, (B, S))
        outs = greedy_decode(model, src, BOS, EOS, max_len=10)
        self.assertEqual(len(outs), B)
        self.assertTrue(all(len(o) <= 10 for o in outs))
        beam = beam_search(model, src[:1], BOS, EOS, beam_size=3, max_len=10)
        self.assertTrue(all(isinstance(t, int) for t in beam))

    def test_decoders_never_return_empty_output(self):
        """Even a model that strongly prefers </s> first must emit at least one token."""
        torch.manual_seed(0)
        model = tiny_model().eval()
        proj = model.generator.proj
        orig_forward = proj.forward
        eos_boost = torch.nn.functional.one_hot(torch.tensor(EOS), V).float() * 100.0
        proj.forward = lambda x: orig_forward(x) + eos_boost   # </s> overwhelmingly likely
        src = torch.randint(4, V, (2, S))
        self.assertTrue(all(len(o) >= 1 for o in greedy_decode(model, src, BOS, EOS, max_len=8)))
        self.assertGreaterEqual(len(beam_search(model, src[:1], BOS, EOS, beam_size=4, max_len=8)), 1)

    def test_beam_size_one_equals_greedy(self):
        torch.manual_seed(1)
        model = tiny_model().eval()
        src = torch.randint(4, V, (1, S))
        g = greedy_decode(model, src, BOS, EOS, max_len=12)[0]
        b = beam_search(model, src, BOS, EOS, beam_size=1, alpha=0.0, max_len=12)
        self.assertEqual(g, b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
