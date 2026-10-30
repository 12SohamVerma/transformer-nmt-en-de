"""
Streamlit UI for the from-scratch Transformer NMT model (English -> German, Multi30k).

Run:  streamlit run app.py          (or: python -m streamlit run app.py)

Translation uses the project's own Translator (translate.py): it loads
checkpoints/small_best.pt and the SentencePiece tokenizer, and calls the greedy and
beam-search decoders in models/decoding.py. All metrics are read from results/ -
nothing shown here is hard-coded. No external translation service is used.
"""

from __future__ import annotations

import html
import json

import streamlit as st

import config as C

CHECKPOINT = C.CHECKPOINT_DIR / "small_best.pt"
BEAM_SIZE, ALPHA = 4, 0.6
GREEDY, BEAM = "Greedy Search", f"Beam Search (beam size {BEAM_SIZE}, α = {ALPHA})"
EXAMPLES = [
    "A man is riding a bicycle down the street.",
    "Two dogs are playing in the snow.",
    "A little girl in a pink dress is climbing a tree.",
    "A woman is reading a book on a bench in the park.",
    "Three children are playing soccer on a green field.",
    "A group of people are standing in front of a building.",
]

st.set_page_config(page_title="Transformer NMT · English → German", page_icon="🔤", layout="wide")

CSS = """
<style>
:root {
  --blue: #2563eb; --purple: #7c3aed; --ink: #0f172a; --muted: #64748b;
  --line: #e2e8f0; --card: #ffffff; --soft: #f1f5ff;
}
.block-container {max-width: 1120px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1, h2, h3, p, li, label {font-family: Inter, "Segoe UI", -apple-system, Roboto, sans-serif;}

.hero {background: var(--card); border: 1px solid var(--line); border-radius: 18px;
       padding: 2rem 2.2rem; position: relative; overflow: hidden;}
.hero::before {content: ""; position: absolute; inset: 0 0 auto 0; height: 5px;
               background: linear-gradient(90deg, var(--blue), var(--purple));}
.hero .eyebrow {color: var(--purple); font-weight: 600; font-size: .8rem;
                letter-spacing: .06em; text-transform: uppercase; margin: 0 0 .5rem;}
.hero h1 {font-size: 2.3rem; line-height: 1.15; margin: 0; padding: 0; font-weight: 750;
          background: linear-gradient(90deg, var(--blue), var(--purple));
          -webkit-background-clip: text; background-clip: text; color: transparent;}
.hero .sub {color: var(--muted); font-size: 1.1rem; margin: .5rem 0 1rem;}
.chips span {display: inline-block; background: var(--soft); color: #3730a3; border-radius: 999px;
             padding: .25rem .75rem; font-size: .82rem; margin: 0 .4rem .4rem 0; font-weight: 500;}

.section-title {font-size: 1.35rem; font-weight: 700; color: var(--ink); margin: 2.4rem 0 .2rem;}
.section-sub {color: var(--muted); font-size: .95rem; margin: 0 0 1rem;}

.grid {display: grid; gap: .8rem;}
.grid.cols-3 {grid-template-columns: repeat(3, minmax(0, 1fr));}
.grid.cols-5 {grid-template-columns: repeat(5, minmax(0, 1fr));}
@media (max-width: 900px) {.grid.cols-5 {grid-template-columns: repeat(3, minmax(0, 1fr));}}
@media (max-width: 640px) {.grid.cols-3, .grid.cols-5 {grid-template-columns: repeat(2, minmax(0, 1fr));}}
.card {background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 1rem 1.1rem;}
.card .label {color: var(--muted); font-size: .78rem; font-weight: 600; text-transform: uppercase;
              letter-spacing: .04em; margin-bottom: .3rem;}
.card .value {color: var(--ink); font-size: 1.05rem; font-weight: 600; line-height: 1.35;}
.card .hint {color: var(--muted); font-size: .8rem; margin-top: .25rem;}
.metric .value {font-size: 1.7rem; font-weight: 700;}
.metric.highlight {border: 1.5px solid #c7d2fe; background: linear-gradient(180deg, #eef2ff, #ffffff);}
.metric.highlight .value {color: #4338ca;}

.output {background: var(--card); border: 1px solid var(--line); border-left: 5px solid var(--purple);
         border-radius: 14px; padding: 1.3rem 1.4rem; min-height: 190px;}
.output .label {color: var(--purple); font-size: .78rem; font-weight: 700; text-transform: uppercase;
                letter-spacing: .05em;}
.output .text {color: var(--ink); font-size: 1.45rem; line-height: 1.45; margin: .6rem 0 .9rem; font-weight: 500;}
.output .source {color: var(--muted); font-size: .9rem;}
.output .meta span {display: inline-block; background: var(--soft); color: #3730a3; border-radius: 8px;
                    padding: .15rem .55rem; font-size: .8rem; margin: .5rem .35rem 0 0;}
.output.empty {border-left-color: var(--line); display: flex; align-items: center; justify-content: center;
               color: var(--muted); text-align: center;}

.status {display: inline-block; border-radius: 999px; padding: .2rem .7rem; font-size: .8rem; font-weight: 600;}
.status.ok {background: #ecfdf5; color: #047857;}
.status.err {background: #fef2f2; color: #b91c1c;}

.flow {display: flex; align-items: stretch; gap: .35rem; margin: .4rem 0 1.2rem;}
.flow .step {flex: 1 1 0; min-width: 0; background: var(--card); border: 1px solid var(--line); border-radius: 12px;
             padding: .7rem .5rem; font-weight: 600; color: var(--ink); font-size: .88rem; text-align: center;
             display: flex; flex-direction: column; justify-content: center;}
.flow .step small {display: block; color: var(--muted); font-weight: 400; font-size: .75rem; margin-top: .1rem;}
.flow .step.io {background: linear-gradient(90deg, var(--blue), var(--purple)); color: #fff; border: none;}
.flow .step.io small {color: #e0e7ff;}
.flow .arrow {color: #a5b4fc; font-size: 1.2rem; font-weight: 700; align-self: center;}
@media (max-width: 760px) {
  .flow {flex-direction: column;}
  .flow .arrow {transform: rotate(90deg);}
}

.note {background: #fffbeb; border: 1px solid #fde68a; color: #78350f; border-radius: 12px;
       padding: .75rem 1rem; font-size: .88rem; margin-top: .9rem;}
.footer {color: var(--muted); font-size: .82rem; text-align: center; margin-top: 3rem;
         border-top: 1px solid var(--line); padding-top: 1rem;}

div[data-testid="stTextArea"] textarea {font-size: 1.05rem; line-height: 1.5; border-radius: 12px;}
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(90deg, var(--blue), var(--purple)); border: none; font-weight: 600;}
button[data-testid="stBaseButton-secondary"] {border-radius: 12px; font-size: .85rem; height: 100%;}
button[data-testid="stBaseButton-secondary"] p {white-space: normal; text-align: left; overflow: visible;}
</style>
"""


def render(markup: str) -> None:
    """Render an HTML snippet (lines are stripped so Markdown never treats them as code blocks)."""
    st.markdown("\n".join(line.strip() for line in markup.strip().splitlines() if line.strip()),
                unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text))


@st.cache_resource(show_spinner="Loading the trained Transformer checkpoint…")
def load_translator():
    from translate import Translator   # project's own inference code (tokenizer + decoders)
    return Translator(CHECKPOINT)


def get_translator():
    try:
        return load_translator(), None
    except Exception as exc:          # missing checkpoint / tokenizer, vocabulary mismatch, ...
        return None, str(exc)


def load_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def use_example(sentence: str) -> None:
    st.session_state.src_text = sentence


render(CSS)
translator, load_error = get_translator()
metrics = load_json(C.METRICS_DIR / "small_metrics.json")
history = load_json(C.METRICS_DIR / "small_history.json")

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
render("""
<div class="hero">
  <p class="eyebrow">Computational Linguistics &amp; NLP Lab · Experiments 9 &amp; 10</p>
  <h1>Transformer-Based Neural Machine Translation</h1>
  <p class="sub">Implementation of <em>Attention Is All You Need</em> (2017)</p>
  <div class="chips">
    <span>English → German</span><span>Built from scratch in PyTorch</span>
    <span>Runs locally · no external APIs</span>
  </div>
</div>
""")

# --------------------------------------------------------------------------
# A. Project overview
# --------------------------------------------------------------------------
render('<div class="section-title">Project Overview</div>')
overview = [
    ("Research paper", "Attention Is All You Need (2017)"),
    ("Authors", "Vaswani et al."),
    ("Framework", "PyTorch"),
    ("Application", "English-to-German Machine Translation"),
    ("Dataset", "Multi30k"),
    ("Architecture", "Transformer Encoder–Decoder"),
]
render('<div class="grid cols-3">' + "".join(
    f'<div class="card"><div class="label">{esc(k)}</div><div class="value">{esc(v)}</div></div>'
    for k, v in overview) + "</div>")

# --------------------------------------------------------------------------
# B. Translation interface
# --------------------------------------------------------------------------
status = ('<span class="status ok">● Model loaded · checkpoints/small_best.pt</span>' if translator
          else '<span class="status err">● Model unavailable</span>')
render(f'<div class="section-title">Translate English → German</div>'
       f'<p class="section-sub">Type an English sentence or pick an example. {status}</p>')

if "src_text" not in st.session_state:
    st.session_state.src_text = EXAMPLES[0]

left, right = st.columns([1, 1], gap="large")
with left:
    st.text_area("English sentence", key="src_text", height=150, max_chars=500,
                 placeholder="e.g. A man is riding a bicycle down the street.")
    decoding = st.radio("Decoding method", [GREEDY, BEAM], index=1, horizontal=True,
                        help="Greedy picks the most likely word at each step. Beam search keeps the 4 best "
                             "partial translations and uses length penalty α = 0.6 (the paper's settings).")
    clicked = st.button("Translate", type="primary", width="stretch")
    st.caption("Try an example:")
    ex_cols = st.columns(2)
    for i, sentence in enumerate(EXAMPLES):
        ex_cols[i % 2].button(sentence, key=f"example_{i}", on_click=use_example, args=(sentence,),
                              width="stretch")

    if clicked:
        text = st.session_state.src_text.strip()
        if not text:
            st.warning("Please enter an English sentence to translate.", icon="✏️")
        elif translator is None:
            st.error("The trained model could not be loaded, so translation is unavailable right now. "
                     "Make sure `checkpoints/small_best.pt` and `tokenizer/spm_bpe.model` exist "
                     f"(run `python prepare_data.py` and `python train.py`).\n\nDetails: {load_error}",
                     icon="⚠️")
        else:
            beam = BEAM_SIZE if decoding == BEAM else 1           # beam size 1 = greedy decoding
            try:
                with st.spinner("Translating with the Transformer…"):
                    result = translator.translate(text, beam_size=beam, alpha=ALPHA)
                st.session_state.result = {"source": result.source, "translation": result.translation,
                                           "method": decoding, "ms": result.seconds * 1000,
                                           "tokens": len(result.tokens)}
            except Exception as exc:
                st.error(f"Something went wrong during translation: {exc}", icon="⚠️")

with right:
    res = st.session_state.get("result")
    if res:
        method = "Greedy search" if res["method"] == GREEDY else f"Beam search (k = {BEAM_SIZE}, α = {ALPHA})"
        render(f"""
        <div class="output">
          <div class="label">German translation</div>
          <div class="text">{esc(res['translation']) or '<em>(empty output)</em>'}</div>
          <div class="source"><strong>Input:</strong> {esc(res['source'])}</div>
          <div class="meta"><span>{esc(method)}</span><span>{res['ms']:.0f} ms on CPU</span>
          <span>{res['tokens']} sub-word tokens</span></div>
        </div>""")
    else:
        render('<div class="output empty">Your German translation will appear here.</div>')
    st.caption("The model was trained on Multi30k image captions, so it works best on short, "
               "descriptive sentences about people, animals and everyday scenes.")

# --------------------------------------------------------------------------
# C. Model performance dashboard
# --------------------------------------------------------------------------
render('<div class="section-title">Model Performance</div>'
       '<p class="section-sub">Multi30k test results · test_2016_flickr (1,000 held-out sentence pairs) · '
       'best checkpoint (epoch 15)</p>')

if metrics is None:
    st.info("Evaluation results not found. Run `python evaluate.py` to generate "
            "`results/metrics/small_metrics.json`.")
else:
    bleu, losses, timing = metrics["bleu"], metrics["losses"], metrics["inference_time"]
    beam_bleu = next(v for k, v in bleu.items() if k.startswith("beam") and k != "beam_detail")
    train_min = history["total_train_time_sec"] / 60 if history else None
    cards = [
        ("BLEU · Beam Search", f"{beam_bleu:.2f}", "SacreBLEU, beam 4, α 0.6", True),
        ("BLEU · Greedy", f"{bleu['greedy']:.2f}", "SacreBLEU", False),
        ("chrF", f"{bleu['chrf_beam']:.2f}", "character n-gram F-score", False),
        ("Perplexity", f"{losses['test_ppl']:.2f}", "test set, per sub-word token", False),
        ("Training Loss", f"{losses['train_loss_at_best_epoch']:.3f}", "epoch 15", False),
        ("Validation Loss", f"{losses['val_loss']:.3f}", "1,014 sentences", False),
        ("Test Loss", f"{losses['test_loss']:.3f}", "1,000 sentences", False),
        ("Parameters", f"{metrics['num_trainable_parameters'] / 1e6:.2f}M", "trainable", False),
        ("Training Time", f"{train_min:.0f} min" if train_min else "–", "15 epochs on CPU", False),
        ("Inference Latency", f"≈{timing['single_sentence_greedy_latency_ms_mean']:.0f} ms",
         f"per sentence, greedy, CPU (beam ≈{timing['beam_ms_per_sentence']:.0f} ms)", False),
    ]
    render('<div class="grid cols-5">' + "".join(
        f'<div class="card metric{" highlight" if hl else ""}"><div class="label">{esc(label)}</div>'
        f'<div class="value">{esc(value)}</div><div class="hint">{esc(hint)}</div></div>'
        for label, value, hint, hl in cards) + "</div>")
    render("""
    <div class="note">Losses are label-smoothed cross-entropy (ε = 0.1). Perplexity = exp(un-smoothed test
    cross-entropy). These are <strong>Multi30k</strong> results and are not comparable to the paper's
    WMT 2014 scores (27.3 / 28.4 BLEU), which were not reproduced here.</div>""")

    loss_png = C.GRAPHS_DIR / "small_loss_curve.png"
    if loss_png.exists():
        with st.expander("Training vs validation loss curve"):
            st.image(str(loss_png), width="stretch")

# --------------------------------------------------------------------------
# D. Architecture overview
# --------------------------------------------------------------------------
render('<div class="section-title">Architecture Overview</div>'
       '<p class="section-sub">How a sentence flows through the model</p>')
steps = [("English Input", "source sentence", True), ("Tokenization", "SentencePiece BPE", False),
         ("Positional Encoding", "embedding × √d + sin/cos", False), ("Encoder", "self-attention + FFN", False),
         ("Decoder", "masked + cross-attention", False), ("Linear + Softmax", "word probabilities", False),
         ("German Output", "translated sentence", True)]
render('<div class="flow">' + '<span class="arrow">→</span>'.join(
    f'<div class="step{" io" if io else ""}">{esc(name)}<small>{esc(desc)}</small></div>'
    for name, desc, io in steps) + "</div>")

cfg = translator.model_config if translator else C.MODEL_CONFIGS["small"]   # config stored in the checkpoint
arch = [("Encoder layers", cfg.num_encoder_layers), ("Decoder layers", cfg.num_decoder_layers),
        ("Embedding dimension", cfg.d_model), ("Attention heads", cfg.num_heads),
        ("Feed-forward dimension", cfg.d_ff), ("Dropout", cfg.dropout)]
render('<div class="grid cols-3">' + "".join(
    f'<div class="card"><div class="label">{esc(k)}</div><div class="value">{esc(v)}</div></div>'
    for k, v in arch) + "</div>")
base = C.MODEL_CONFIGS["base"]
st.caption(f"This is the **Small configuration** of the original Transformer, sized for CPU training. "
           f"The paper's base model uses {base.num_encoder_layers} + {base.num_decoder_layers} layers, "
           f"d_model = {base.d_model}, {base.num_heads} heads and d_ff = {base.d_ff}. That configuration is "
           f"also implemented in this project (`python train.py --config base`).")

render("""
<div class="footer">Vaswani et al., <em>Attention Is All You Need</em>, NeurIPS 2017 · Multi30k (Elliott et al., 2016)
· All translations are produced locally by the trained checkpoint.</div>
""")
