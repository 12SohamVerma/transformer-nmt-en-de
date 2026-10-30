"""
Streamlit demo for the from-scratch Transformer (English -> German, Multi30k).

Run:  python -m streamlit run app_detailed.py   (detailed version with test-set browser and per-epoch log)
Uses only the locally trained checkpoint and our own implementation - no external APIs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402
import torch  # noqa: E402

import config as C  # noqa: E402

st.set_page_config(page_title="Transformer NMT · EN→DE", page_icon="🔤", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1200px;}
      .translation-box {background: rgba(42,120,214,0.08); border-left: 4px solid #2a78d6;
                        padding: 1rem 1.2rem; border-radius: 6px; font-size: 1.25rem;}
      .muted {color: #6b6a64; font-size: 0.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [
    "A man is riding a bicycle down the street.",
    "Two dogs are playing in the snow.",
    "A little girl in a pink dress is climbing a tree.",
    "A group of people are standing in front of a building.",
    "A woman is reading a book on a bench in the park.",
    "Three children are playing soccer on a green field.",
]


# --------------------------------------------------------------------------
# Loading (cached, with explicit error messages)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Transformer checkpoint…")
def load_translator(path: str):
    from translate import Translator
    return Translator(path)


def load_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


checkpoints = sorted(C.CHECKPOINT_DIR.glob("*_best.pt"))
translator, load_error = None, None

with st.sidebar:
    st.header("⚙️ Settings")
    if checkpoints:
        names = [p.name for p in checkpoints]
        default = names.index("small_best.pt") if "small_best.pt" in names else 0
        ckpt_name = st.selectbox("Checkpoint", names, index=default)
        try:
            translator = load_translator(str(C.CHECKPOINT_DIR / ckpt_name))
        except Exception as exc:  # tokenizer missing, corrupt file, vocab mismatch ...
            load_error = str(exc)
    else:
        load_error = (f"No checkpoint found in `{C.CHECKPOINT_DIR.name}/`. "
                      "Run `python prepare_data.py` and `python train.py --config small` first.")
    if not C.SPM_MODEL_PATH.exists():
        load_error = "SentencePiece tokenizer missing (`tokenizer/spm_bpe.model`). Run `python prepare_data.py`."

    decoding = st.radio("Decoding", ["Beam search", "Greedy"], help="The paper uses beam size 4, α = 0.6")
    beam_size = st.slider("Beam size", 2, 8, 4, disabled=decoding == "Greedy")
    alpha = st.slider("Length penalty α", 0.0, 1.5, 0.6, 0.1, disabled=decoding == "Greedy")
    show_attention = st.checkbox("Show cross-attention heatmap", value=True)
    st.divider()
    if translator:
        st.success(f"Model loaded: **{translator.model_config.name}** config")
        st.caption(f"{translator.num_parameters:,} trainable parameters · "
                   f"best epoch {translator.checkpoint_info['epoch']} · device: CPU")
    elif load_error:
        st.error(load_error)

tag = translator.checkpoint_path.stem.replace("_best", "") if translator else "small"
metrics = load_json(C.METRICS_DIR / f"{tag}_metrics.json")
history = load_json(C.METRICS_DIR / f"{tag}_history.json")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Transformer Neural Machine Translation: English → German")
st.markdown(
    "A **from-scratch PyTorch implementation** of the Transformer encoder–decoder from "
    "*Attention Is All You Need* (Vaswani et al., 2017), trained on the **Multi30k** "
    "English–German corpus. No `nn.Transformer`, no pretrained models, no external translation APIs."
)

tab_translate, tab_results, tab_arch, tab_paper = st.tabs(
    ["🔤 Translate", "📊 Evaluation results", "🏗️ Architecture", "📄 Paper & dataset"]
)


def attention_figure(text: str, translation: str):
    sp, model = translator.sp, translator.model
    src = translator._src_tensor([text])
    ids = sp.encode(translation)
    with torch.no_grad():
        model(src, torch.tensor([[C.BOS_ID] + ids]))
    attn = model.decoder.layers[-1].cross_attn.attn_weights[0].mean(0).numpy()
    src_tok = [sp.id_to_piece(i) for i in src[0].tolist()]
    tgt_tok = ["<s>"] + [sp.id_to_piece(i) for i in ids]
    fig, ax = plt.subplots(figsize=(max(5, 0.45 * len(src_tok)), max(3, 0.36 * len(tgt_tok))), dpi=120)
    im = ax.imshow(attn, cmap="Blues", aspect="auto", vmin=0)
    ax.set_xticks(range(len(src_tok)), src_tok, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tgt_tok)), tgt_tok, fontsize=8)
    ax.set_xlabel("English source tokens (keys)")
    ax.set_ylabel("German decoder tokens (queries)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Translate
# --------------------------------------------------------------------------
with tab_translate:
    if "src_text" not in st.session_state:
        st.session_state.src_text = EXAMPLES[0]

    st.markdown("**Try an example:**")
    cols = st.columns(3)
    for i, ex in enumerate(EXAMPLES):
        if cols[i % 3].button(ex, key=f"ex{i}", width="stretch"):
            st.session_state.src_text = ex

    text = st.text_area("English input", key="src_text", height=100, max_chars=500)
    go = st.button("Translate", type="primary", disabled=translator is None)

    if translator is None:
        st.warning("Translation is unavailable until a trained checkpoint and tokenizer are present. "
                   + (load_error or ""))
    elif go:
        if not text.strip():
            st.error("Please enter an English sentence.")
        else:
            k = 1 if decoding == "Greedy" else beam_size
            with st.spinner("Translating…"):
                out = translator.translate(text, k, alpha)
            st.markdown("**German translation (model output)**")
            st.markdown(f"<div class='translation-box'>{out.translation or '<i>(empty output)</i>'}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<p class='muted'>{'Greedy' if k == 1 else f'Beam search (k={k}, α={alpha})'} · "
                        f"{len(out.tokens)} BPE tokens · {out.seconds * 1000:.0f} ms on CPU</p>",
                        unsafe_allow_html=True)
            with st.expander("Generated BPE tokens"):
                st.code(" ".join(out.tokens))
            if show_attention and out.translation:
                st.markdown("**Encoder–decoder (cross) attention** – last decoder layer, averaged over heads. "
                            "Each row shows which English tokens the decoder attended to when producing "
                            "the next German token.")
                st.pyplot(attention_figure(out.source, out.translation), width="content")

    st.info("The model was trained only on Multi30k (short image captions, ~29k sentence pairs). "
            "It translates simple descriptive sentences well, but out-of-domain text "
            "(news, technical, conversational) will be translated poorly.", icon="ℹ️")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
with tab_results:
    n_eval = metrics["evaluated_sentences"] if metrics else 1000
    st.subheader(f"Evaluation on Multi30k test set (test_2016_flickr, {n_eval} sentences)")
    if metrics is None:
        st.warning(f"No evaluation metrics found for '{tag}'. Run `python evaluate.py`.")
    else:
        bleu, losses, inf = metrics["bleu"], metrics["losses"], metrics["inference_time"]
        beam_key = next(k for k in bleu if k.startswith("beam") and k != "beam_detail")
        c = st.columns(4)
        c[0].metric("BLEU (beam search)", f"{bleu[beam_key]:.2f}")
        c[1].metric("BLEU (greedy)", f"{bleu['greedy']:.2f}")
        c[2].metric("chrF (beam)", f"{bleu['chrf_beam']:.2f}")
        c[3].metric("Trainable parameters", f"{metrics['num_trainable_parameters'] / 1e6:.2f} M")
        c = st.columns(4)
        if losses.get("train_loss_at_best_epoch") is not None:
            c[0].metric("Training loss", f"{losses['train_loss_at_best_epoch']:.3f}")
        c[1].metric("Validation loss", f"{losses['val_loss']:.3f}")
        c[2].metric("Test loss", f"{losses['test_loss']:.3f}", help=f"Test perplexity {losses['test_ppl']:.2f}")
        c[3].metric("Latency / sentence (greedy, CPU)", f"{inf['single_sentence_greedy_latency_ms_mean']:.0f} ms")
        st.caption(f"Losses are label-smoothed cross-entropy per target token (ε_ls = 0.1). "
                   f"SacreBLEU signature: `{bleu['sacrebleu_signature']}`")
        st.warning("These numbers are for **Multi30k**, not WMT 2014. The paper's 27.3 (base) / 28.4 (big) BLEU "
                   "on WMT14 newstest2014 was **not** reproduced here and is not directly comparable.", icon="⚠️")

    loss_png = C.GRAPHS_DIR / f"{tag}_loss_curve.png"
    attn_png = C.GRAPHS_DIR / f"{tag}_cross_attention.png"
    g1, g2 = st.columns(2)
    if loss_png.exists():
        g1.image(str(loss_png), caption="Training vs validation loss", width="stretch")
    if attn_png.exists():
        g2.image(str(attn_png), caption="Cross-attention of the trained model", width="stretch")

    if history:
        with st.expander("Per-epoch training log"):
            h = history["history"]
            st.dataframe(pd.DataFrame({
                "epoch": h["epoch"], "train loss": h["train_loss"], "val loss": h["val_loss"],
                "val perplexity": h["val_ppl"], "learning rate": h["lr"],
                "epoch time (min)": [t / 60 for t in h["epoch_time_sec"]],
            }).round(4), hide_index=True, width="stretch")
            st.caption(f"Total training time: {history['total_train_time_sec'] / 3600:.2f} h on {history['device']}")

    pred_csv = C.PREDICTIONS_DIR / f"{tag}_test_predictions.csv"
    if pred_csv.exists():
        st.subheader("Sample predictions (test set)")
        df = pd.read_csv(pred_csv)
        q = st.text_input("Filter by English word", "")
        if q:
            df = df[df["english_source"].str.contains(q, case=False, na=False)]
        text_col = st.column_config.TextColumn
        st.dataframe(
            df[["english_source", "model_beam", "german_reference", "sentence_bleu_beam"]],
            column_config={
                "english_source": text_col("English input", width="medium"),
                "model_beam": text_col("Model output (DE)", width="medium"),
                "german_reference": text_col("Ground truth (DE)", width="medium"),
                "sentence_bleu_beam": st.column_config.NumberColumn("sentence BLEU", width="small", format="%.1f"),
            },
            hide_index=True, width="stretch", height=420)

# --------------------------------------------------------------------------
# Architecture
# --------------------------------------------------------------------------
with tab_arch:
    st.subheader("Encoder–decoder data flow")
    st.code(
        """English tokens                         German tokens (shifted right)
     │                                         │
Embedding × √d_model                    Embedding × √d_model
     + Positional Encoding                     + Positional Encoding
     │                                         │
┌────▼──────── Encoder × N ─────┐   ┌──────────▼──── Decoder × N ──────────┐
│ Multi-Head Self-Attention     │   │ Masked Multi-Head Self-Attention     │
│ Add & LayerNorm               │   │ Add & LayerNorm                      │
│ Position-wise FFN (ReLU)      │──►│ Cross-Attention (Q=dec, K=V=enc)     │
│ Add & LayerNorm               │   │ Add & LayerNorm                      │
└───────────────────────────────┘   │ Position-wise FFN (ReLU)             │
                                    │ Add & LayerNorm                      │
                                    └──────────┬───────────────────────────┘
                                         Linear → Softmax
                                               │
                                   P(next German token)""",
        language=None,
    )
    st.subheader("Key equations")
    st.latex(r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V")
    st.latex(r"\mathrm{MultiHead}(Q,K,V)=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O")
    st.latex(r"PE_{(pos,2i)}=\sin\!\left(pos/10000^{2i/d_{model}}\right),\;"
             r"PE_{(pos,2i+1)}=\cos\!\left(pos/10000^{2i/d_{model}}\right)")
    st.latex(r"\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2")
    st.latex(r"\text{sublayer output}=\mathrm{LayerNorm}(x+\mathrm{Dropout}(\mathrm{Sublayer}(x)))")
    st.latex(r"lrate=d_{model}^{-0.5}\cdot\min(step^{-0.5},\;step\cdot warmup^{-1.5})")
    st.subheader("Model configuration")
    rows = []
    for name in ("small", "base"):
        m, t = C.get_config(name)
        rows.append({"config": name, "N (enc/dec)": f"{m.num_encoder_layers}/{m.num_decoder_layers}",
                     "d_model": m.d_model, "heads h": m.num_heads, "d_k = d_v": m.d_k, "d_ff": m.d_ff,
                     "dropout": m.dropout, "label smoothing": t.label_smoothing, "warmup": t.warmup_steps})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("`base` = Transformer (base) from Table 3 of the paper. The demo model uses the "
               "configuration stored in the loaded checkpoint.")
    if translator:
        st.json(translator.model_config.to_dict(), expanded=False)
    st.subheader("Component → source file")
    st.dataframe(pd.DataFrame([
        ("Input embedding (× √d_model)", "models/layers.py · TokenEmbedding"),
        ("Sinusoidal positional encoding", "models/positional_encoding.py"),
        ("Scaled dot-product attention", "models/attention.py · scaled_dot_product_attention"),
        ("Multi-head (self / masked / cross) attention", "models/attention.py · MultiHeadAttention"),
        ("Position-wise feed-forward", "models/layers.py · PositionwiseFeedForward"),
        ("Residual + LayerNorm (post-LN)", "models/layers.py · SublayerConnection, LayerNorm"),
        ("Encoder layer / stack", "models/encoder.py"),
        ("Decoder layer / stack", "models/decoder.py"),
        ("Padding & causal masks", "models/transformer.py · make_src_mask, make_tgt_mask"),
        ("Linear projection + softmax", "models/transformer.py · Generator"),
        ("Greedy & beam search", "models/decoding.py"),
        ("Noam schedule, label smoothing", "train.py"),
    ], columns=["Component", "Implementation"]), hide_index=True, width="stretch")

# --------------------------------------------------------------------------
# Paper
# --------------------------------------------------------------------------
with tab_paper:
    st.subheader("Research paper")
    st.markdown(
        """
**Attention Is All You Need**. Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones,
Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin. *Advances in Neural Information Processing Systems 30
(NeurIPS 2017)*, pp. 5998–6008. arXiv:1706.03762.

The paper replaces recurrence and convolution entirely with **attention**, so all positions of a
sentence are processed in parallel. On WMT 2014 the authors report 27.3 BLEU (base) and 28.4 BLEU (big)
for English→German.
        """
    )
    st.subheader("Dataset")
    st.markdown(
        """
**Multi30k** (Elliott, Frank, Sima'an & Specia, 2016): English image descriptions from Flickr30k with
professional German translations. Splits used: **train 29,000**, **validation 1,014**,
**test 1,000** (test_2016_flickr). The test set is used only for final evaluation.

Tokenisation: joint English+German **SentencePiece BPE**, 8,000 sub-word units, special tokens
`<pad>`=0, `<unk>`=1, `<s>`=2, `</s>`=3.
        """
    )
    st.subheader("How this project differs from the paper's experiments")
    st.table(pd.DataFrame([
        ("Training data", "WMT14 EN–DE, ~4.5M pairs", "Multi30k, 29k pairs"),
        ("Vocabulary", "~37k shared BPE", "8k shared SentencePiece BPE"),
        ("Hardware", "8 × NVIDIA P100 GPUs", "Laptop CPU (AMD Ryzen 5 5500U)"),
        ("Model trained", "base (65M) and big (213M)", "small (7.6M); base implemented, not trained"),
        ("Training length", "100k steps (base)", "15 epochs = 3,405 steps (≈75 min CPU)"),
        ("Checkpoint averaging", "last 5 checkpoints", "single best checkpoint"),
    ], columns=["Aspect", "Paper (Vaswani et al.)", "This project"]))
