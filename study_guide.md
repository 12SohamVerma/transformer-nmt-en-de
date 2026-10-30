# Study Guide – Transformer Neural Machine Translation (English → German)

*Complete study material for Experiments 9 & 10: concepts, mathematics, your implementation, your results,
and viva preparation. Every number in this guide comes from your own trained model and code.*

---

## How to use this guide

| If you have… | Read |
|---|---|
| 10 minutes | §1 Big picture, §15 Cheat sheet |
| 1 hour | §1–§9, §15 |
| A full study session | Everything, then answer §17 without looking, then read `viva_questions.md` |

---

## 1. The big picture (memorise this story)

> "Machine translation turns a sentence in one language into another. Older neural models (RNNs and
> LSTMs) read a sentence one word at a time, so they were slow to train and forgot distant words.
> In 2017, Vaswani et al. proposed the **Transformer**, which removes recurrence completely and uses
> only **attention**: every word looks at every other word in one step, so the whole sentence is
> processed in parallel. I implemented this architecture **from scratch in PyTorch**, without
> `nn.Transformer`, trained it on the **Multi30k** English–German dataset, and it reaches
> **37.09 BLEU** on the held-out test set. A Streamlit app demonstrates live translation."

**Five facts to remember:**
1. Paper: *Attention Is All You Need*, Vaswani et al., NeurIPS 2017.
2. Architecture: encoder–decoder, **3 + 3 layers**, d_model **256**, **8** heads, d_ff **1024**, dropout **0.1**.
3. Data: Multi30k with **29,000 / 1,014 / 1,000** train / validation / test sentence pairs.
4. Training: **15 epochs** on a laptop CPU, about **75 minutes**, **7,577,600** parameters.
5. Results: BLEU **37.09** (beam) / **36.25** (greedy), chrF **60.98**, test perplexity **6.00**.

---

## 2. Background you must know

### 2.1 Machine translation as probability
A translation model learns

**P(y | x) = P(y₁ | x) · P(y₂ | y₁, x) · … · P(yₘ | y₁…yₘ₋₁, x)**

where x is the English sentence and y the German one. The model predicts **one word at a time**, each time
conditioned on the source sentence and the words already generated. This is called **auto-regressive**
generation.

### 2.2 Sequence-to-sequence (encoder–decoder)
- The **encoder** reads the source sentence and builds a representation of it.
- The **decoder** generates the target sentence word by word, consulting the encoder's representation.

### 2.3 Why RNNs/LSTMs were replaced
| Problem with RNN/LSTM | How the Transformer fixes it |
|---|---|
| Sequential: step t must wait for step t−1, so no parallelism | All positions are computed at once |
| Long-range dependencies pass through many steps, so information fades | Any two words connect in **one** attention step (path length O(1)) |
| Fixed-size bottleneck vector (before attention was added) | Decoder attends to **all** encoder states |

### 2.4 What "attention" means intuitively
To translate a word, look back at the source words that matter most for it, weighted by relevance.
When the model writes **"Hunde"** it should look mostly at **"dogs"**. Attention computes these
relevance weights automatically, without any alignment supervision.

---

## 3. Data pipeline, step by step with a real example

Example sentence: **"Two dogs are playing in the snow."**

| Step | What happens | Real output from your system |
|---|---|---|
| 1. Cleaning | Unicode NFC, remove control characters, collapse spaces. Case and punctuation are kept. | `Two dogs are playing in the snow.` |
| 2. BPE tokenisation | SentencePiece splits the text into sub-words (`▁` marks the start of a word) | `▁Two ▁dogs ▁are ▁playing ▁in ▁the ▁snow .` |
| 3. Numerical encoding | Each piece becomes an id; `</s>` (id 3) is appended to the source | `[161, 816, 131, 282, 18, 54, 492, 7916, 3]` |
| 4. Padding | Shorter sentences in a batch are filled with `<pad>` (id 0) | `[…, 3, 0, 0]` |
| 5. Model output | Decoder generates German sub-word tokens | `▁Zwei ▁Hunde ▁spielen ▁im ▁Schnee .` |
| 6. Detokenisation | Pieces are joined back into text | **`Zwei Hunde spielen im Schnee.`** |

**Special tokens:** `<pad>` = 0, `<unk>` = 1, `<s>` (BOS) = 2, `</s>` (EOS) = 3.

**Why BPE (sub-words)?** A word-level vocabulary cannot handle unseen words. BPE splits rare words into
known pieces (e.g. *Antriebsradsystem* → several pieces), so there are no out-of-vocabulary words. In
your data the number of `<unk>` tokens is 0 in every split.

**Why a joint (shared) vocabulary?** English and German share many sub-words (names, numbers,
punctuation). One 8,000-token vocabulary lets the encoder embedding, decoder embedding and output layer
share **one** weight matrix (weight tying, §4.9).

**Data hygiene:** the tokenizer is trained on the **training split only**. The test set is used only in
`evaluate.py`, and no test sentence appears in the training data (checked: 0 overlaps).

| Split | Pairs | Avg. words EN / DE | Avg. BPE tokens EN / DE |
|---|---|---|---|
| Train | 29,000 | 11.9 / 11.1 | 14.3 / 14.8 |
| Validation | 1,014 | 12.0 / 11.4 | 14.5 / 15.3 |
| Test (test_2016_flickr) | 1,000 | 11.9 / 10.9 | 14.2 / 14.3 |

---

## 4. The Transformer, component by component

Notation: B = batch size, S = source length, T = target length, d = d_model = 256, h = 8 heads,
d_k = d_v = d/h = 32, V = vocabulary = 8,000.

### 4.1 Input embedding — `models/layers.py → TokenEmbedding`
- A lookup table of shape **8000 × 256**: each token id becomes a 256-dimensional vector.
- The vector is multiplied by **√d_model = √256 = 16** (paper §3.4). This brings it to the same scale as
  the positional encoding, which lies in [−1, 1].
- Shape: (B, S) ids → (B, S, 256).

### 4.2 Positional encoding — `models/positional_encoding.py`
Attention has no notion of order: "dog bites man" and "man bites dog" would look identical. So a
position signal is added:

**PE(pos, 2i) = sin(pos / 10000^(2i/d)),  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))**

Real values from your model (first four dimensions):

| Position | dim 0 | dim 1 | dim 2 | dim 3 |
|---|---|---|---|---|
| 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| 1 | 0.8415 | 0.5403 | 0.8020 | 0.5974 |
| 2 | 0.9093 | −0.4161 | 0.9581 | −0.2863 |

- Low dimensions change quickly with position (short wavelength). High dimensions change very slowly:
  dims 254–255 are ≈ (0.0001, 1.0) for position 1.
- PE(pos + k) is a **linear function** (a rotation) of PE(pos), so relative positions are easy to learn.
- It is fixed (no parameters), and dropout of 0.1 is applied after adding it.

### 4.3 Scaled dot-product attention — `models/attention.py`

**Attention(Q, K, V) = softmax( Q·Kᵀ / √d_k ) · V**

| Symbol | Meaning | Analogy |
|---|---|---|
| Q (query) | what this position is looking for | a search query |
| K (key) | what each position offers | the title of each document |
| V (value) | the content returned | the document text |

**Worked numerical example** (computed with your code, d_k = 2):
- Q = [1, 0]; K = [[1, 0], [0, 1], [1, 1]]; V = [[1, 2], [3, 4], [5, 6]]
- Scores Q·Kᵀ / √2 = [0.7071, 0, 0.7071]
- Softmax → weights **[0.4011, 0.1978, 0.4011]**, which sum to 1
- Output = 0.4011·[1,2] + 0.1978·[3,4] + 0.4011·[5,6] = **[3.0, 4.0]**

The query matches keys 1 and 3 equally well, so it takes mostly their values.

**Why divide by √d_k?** For random vectors the dot product has variance d_k. Large scores make softmax
almost one-hot, and its gradient becomes ≈ 0, so the model stops learning. Dividing by √d_k keeps the
variance near 1.

### 4.4 Masking — `models/transformer.py → make_src_mask, make_tgt_mask`
Blocked positions get score = −∞ (in code, the most negative float), so softmax gives them weight 0.

Same example with key 3 masked: weights become **[0.6698, 0.3302, 0.0]** and the output becomes
**[1.66, 2.66]**. Key 3 is now completely ignored.

**1. Padding mask:** do not attend to `<pad>` tokens. Used in all three attention types.

**2. Causal (look-ahead) mask:** in decoder self-attention, position i may only see positions ≤ i.
For 4 target tokens the mask is lower-triangular (1 = allowed):

```
          <s>  Zwei Hunde spielen
<s>        1    0    0     0
Zwei       1    1    0     0
Hunde      1    1    1     0
spielen    1    1    1     1
```

Without it, the decoder could "cheat" during training by reading the next word it is supposed to predict.

**Proof it works (unit tests):** changing a *future* target token leaves all earlier outputs unchanged,
and adding `<pad>` to the source leaves the logits unchanged.

### 4.5 Multi-head attention — `models/attention.py → MultiHeadAttention`

**MultiHead(Q, K, V) = Concat(head₁, …, head₈) · W^O,  headᵢ = Attention(Q·Wᵢ^Q, K·Wᵢ^K, V·Wᵢ^V)**

- 8 heads, each working in a 32-dimensional subspace (256 / 8).
- Different heads learn different relations: word alignment, neighbouring words, subject–verb links.
- In the code, the 8 projections are one `Linear(256, 256)` reshaped to (B, 8, L, 32), so all heads run
  in one matrix multiplication. A unit test proves this equals the explicit per-head loop.
- Parameters per attention block: 4 × (256×256 + 256) = **263,168**.

**The same module is used three ways:**

| Where | Q from | K, V from | Mask |
|---|---|---|---|
| Encoder self-attention | encoder | encoder | padding |
| Decoder masked self-attention | decoder | decoder | padding + causal |
| Cross-attention (encoder–decoder) | decoder | encoder output ("memory") | source padding |

### 4.6 Position-wise feed-forward network — `models/layers.py`
**FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂**, with 256 → 1024 → 256 and ReLU.
- Applied to each position independently, with the same weights at every position.
- Attention *mixes* information between words; the FFN *processes* each word's mixed information.
- Parameters: 256×1024 + 1024 + 1024×256 + 256 = **525,568**.

### 4.7 Residual connection + layer normalisation — `SublayerConnection`, `LayerNorm`
**output = LayerNorm( x + Dropout( Sublayer(x) ) )**, the paper's **post-LN** form.
- **Residual (x + …):** gives gradients a shortcut, so deep stacks still train, and each layer only learns
  a *correction*.
- **LayerNorm:** normalises each token's 256 features to mean 0 and variance 1, then applies a learned
  scale γ and shift β. Unlike BatchNorm it does not depend on batch size or padding.
- Your `LayerNorm` is written by hand, and a unit test shows it equals PyTorch's `nn.LayerNorm`.

### 4.8 Encoder and decoder layers
```
EncoderLayer:  x → [Self-Attention → Add&Norm] → [FFN → Add&Norm]
DecoderLayer:  y → [Masked Self-Attention → Add&Norm] → [Cross-Attention(memory) → Add&Norm] → [FFN → Add&Norm]
```
Each is stacked **3 times** in your model (6 in the paper's base model).

### 4.9 Output layer: linear + softmax — `models/transformer.py → Generator`
- Linear 256 → 8,000 gives one score (logit) per vocabulary token.
- Softmax turns the scores into probabilities: **P(next token)**.
- **Weight tying** (paper §3.4): the source embedding, target embedding and this output matrix are the
  **same 8000 × 256 matrix**. This saves parameters and improves generalisation.

### 4.10 Parameter count, derived exactly

| Part | Calculation | Parameters |
|---|---|---|
| Shared embedding / output | 8,000 × 256 | 2,048,000 |
| 1 encoder layer | 263,168 (MHA) + 525,568 (FFN) + 2×512 (LN) | 789,760 |
| 3 encoder layers | 3 × 789,760 | 2,369,280 |
| 1 decoder layer | 2×263,168 + 525,568 + 3×512 | 1,053,440 |
| 3 decoder layers | 3 × 1,053,440 | 3,160,320 |
| **Total** | | **7,577,600** ✔ |

(The paper's base model has about 65M parameters with a 37k vocabulary. Your base configuration
implementation gives about 63M with 37k and 48.2M with your 8k vocabulary.)

---

## 5. Training: how the model learns

### 5.1 Teacher forcing
During training the decoder gets the **correct** previous words, not its own guesses:
```
decoder input  (tgt_in):  <s>   Zwei  Hunde  spielen  im      Schnee  .
labels         (tgt_out): Zwei  Hunde spielen im      Schnee  .       </s>
```
All 7 predictions are trained **in parallel** in one forward pass. The causal mask ensures each position
sees only earlier words. In code: `tgt_in = tgt[:, :-1]`, `tgt_out = tgt[:, 1:]`.

### 5.2 Loss: cross-entropy with label smoothing (ε = 0.1)
- Normal cross-entropy wants probability 1.0 on the correct word.
- Label smoothing wants **0.9000125** on the correct word and **0.0000125** (= 0.1 / 8000) on every other
  word.
- Effect: the model is less over-confident, which hurts perplexity slightly but improves BLEU (paper §5.4).
- Padding positions are ignored (`ignore_index = PAD`).

### 5.3 Optimiser: Adam with β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹ (as in the paper)

### 5.4 Learning-rate schedule (the "Noam" schedule, paper Eq. 3)
**lr = d_model^−0.5 · min(step^−0.5, step · warmup^−1.5)**, with d_model = 256 and warmup = 1,500.

| Step | Learning rate | Phase |
|---|---|---|
| 1 | 0.000001 | warm-up starts tiny |
| 500 | 0.000538 | rising linearly |
| 1,500 | **0.001614 (peak)** | end of warm-up (epoch ≈ 6.6) |
| 2,000 | 0.001398 | decaying as 1/√step |
| 3,405 (last) | 0.001071 | end of training |

**Why warm-up?** At the start the weights are random. Large updates would destabilise a post-LN
Transformer, so the learning rate is increased gradually first.

### 5.5 Regularisation and stability
- **Dropout 0.1** on every sub-layer output and on embedding + positional encoding.
- **Gradient clipping** at norm 1.0 (an addition of ours for stability; not in the paper).
- **Best checkpoint** = lowest validation loss (epoch 15).
- **Seed 42**, so runs are reproducible.

### 5.6 Your training curve (real numbers)

| Epoch | Train loss | Val loss | Val perplexity |
|---|---|---|---|
| 1 | 6.872 | 5.720 | 178.8 |
| 3 | 4.521 | 4.162 | 29.0 |
| 5 | 3.655 | 3.585 | 14.3 |
| 7 | 3.228 | 3.291 | 10.1 |
| 10 | 2.704 | 3.011 | 7.2 |
| 12 | 2.486 | 2.914 | 6.5 |
| 15 | **2.256** | **2.881** | **6.17** |

**Explain the curve:**
- Early on, train loss is *higher* than validation loss. The train loss is averaged over an epoch in which
  the model is still improving, and dropout is active only during training.
- From epoch 6, train loss is lower than validation loss, which is normal.
- After epoch 12, validation loss flattens (2.91 → 2.88) while train loss keeps falling. This is the
  **beginning of over-fitting**, a 7.6M-parameter model on a small dataset.

### 5.7 Sanity check: the overfit test
A small model trained on 32 sentence pairs went from loss 9.44 to 0.00005 and reproduced **32/32**
translations exactly. This proves the architecture, masks and training loop can learn. If something were
broken, it could not memorise.

---

## 6. Inference: how a translation is generated

No target sentence is available at inference time, so the decoder works **auto-regressively**:
```
step 1: <s>                          → "Zwei"
step 2: <s> Zwei                     → "Hunde"
step 3: <s> Zwei Hunde               → "spielen"
…                                    → "</s>"  (stop)
```

### 6.1 Greedy search
Pick the single most probable token at each step. It is fast, but an early mistake cannot be undone.

### 6.2 Beam search (k = 4, α = 0.6, the paper's settings)
- Keep the **4 best partial translations** at every step, and extend each one.
- Compare finished hypotheses with a **length penalty**, because longer sentences naturally have lower
  total log-probability:

  **score = log P(Y) / lp(Y),  lp(Y) = ((5 + |Y|) / 6)^α**

  Values: lp(1) = 1.00, lp(5) = 1.36, lp(10) = 1.73, lp(15) = 2.06.
- Result on the test set: **37.09 BLEU (beam) vs 36.25 (greedy)**, a gain of +0.84.
- Maximum output length = input length + 50 (from the paper).

### 6.3 A bug found and fixed (a good viva story)
On 3 difficult test sentences, beam search first produced **empty translations**. Emitting `</s>`
immediately gave the best length-normalised score. **Fix:** forbid `</s>` as the first token (minimum
length 1, the same rule used in fairseq). A regression test was added, and BLEU rose from 36.99 to
**37.09**. This shows the evaluation was examined carefully, not just run.

---

## 7. Evaluation metrics, explained

### 7.1 BLEU (Papineni et al., 2002)
**BLEU = BP × exp( ¼ · Σ log pₙ ),  n = 1…4**
- pₙ = modified n-gram precision: the fraction of the output's n-grams that also appear in the reference.
- BP = brevity penalty: punishes outputs shorter than the reference (BP = e^(1 − ref/hyp) if shorter).

**Real example from your test set:**
- EN: *A man in an orange hat starring at something.*
- Reference: *Ein Mann mit einem orangefarbenen Hut, der etwas anstarrt.*
- Model: *Ein Mann mit einem orangefarbenen Hut starrt auf etwas.*
- 1–4-gram precisions: **80.0 / 55.6 / 50.0 / 42.9 %**; length 10 vs 11, so **BP = 0.905**
- Sentence BLEU = **50.27**. The translation is correct but phrased differently from the reference,
  and BLEU only rewards exact n-gram overlap.

**Why SacreBLEU?** It scores detokenised text against the original references with a standard
tokenisation. Results are reproducible and comparable, and are reported with a signature:
`nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`.

### 7.2 chrF = 60.98
An F-score over **character** n-grams. It gives partial credit for nearly-right words
(*Fußtritt* vs *Tritt*), which suits German compounds.

### 7.3 Loss and perplexity
- **Loss** (2.881 val, 2.858 test) = label-smoothed cross-entropy per BPE token, the training objective.
- **Perplexity = exp(un-smoothed cross-entropy) = 6.00 on the test set.** Roughly, the model is as
  uncertain as choosing among about 6 equally likely sub-words at each step.
- ⚠️ Trap question: exp(2.858) = 17.4 is **not** the perplexity, because 2.858 includes label smoothing.

### 7.4 BLEU by sentence length (beam)

| Source length | Sentences | BLEU |
|---|---|---|
| ≤ 10 words | 412 | 41.1 |
| 11–15 | 443 | 38.0 |
| 16–20 | 108 | 34.8 |
| > 20 | 37 | 19.6 |

Long sentences are harder: they are rare in training, and errors accumulate during generation.

---

## 8. Your results, and how to defend them

| Metric | Value |
|---|---|
| BLEU (beam 4, α 0.6) | **37.09** |
| BLEU (greedy) | 36.25 |
| chrF | 60.98 |
| Training / validation / test loss | 2.256 / 2.881 / 2.858 |
| Test perplexity | 6.00 |
| Parameters | 7.58M |
| Training time | ~75 min (15 epochs, CPU) |
| Latency | ≈ 119 ms per sentence (greedy, CPU), ≈ 208 ms (beam) |

**"Your 37 BLEU is higher than the paper's 27.3. Did you beat the paper?"**
No. The two numbers measure different tasks:

| | Paper | This project |
|---|---|---|
| Data | WMT 2014, ~4.5M sentence pairs, news text | Multi30k, 29k pairs, short image captions |
| Test set | newstest2014 | test_2016_flickr |
| Model | base 65M / big 213M | small 7.6M |
| Hardware | 8 × P100 GPUs | laptop CPU |

Multi30k sentences are short, simple and repetitive, so BLEU is naturally much higher. The results are
**not comparable**, and the paper's results were **not reproduced** here.

**Strengths:** frequent caption patterns (people, colours, clothing, actions) are translated reliably, e.g.
*"Two dogs are playing in the snow." → "Zwei Hunde spielen im Schnee."*

**Typical errors:** rare words (*Boston → "Bo"*), omissions in long sentences, invented words for unseen
concepts (*home plate*).

---

## 9. Map of your code (what to open in the viva)

| File | What it contains | Key function or class |
|---|---|---|
| `config.py` | small/base configs, paths, special tokens | `MODEL_CONFIGS`, `TRAIN_CONFIGS` |
| `prepare_data.py` | download, verify, clean, train BPE | `verify()`, SentencePiece training |
| `data_utils.py` | encoding, padding, length-bucketed batches | `TranslationDataset`, `BucketBatchSampler` |
| `models/attention.py` | attention formula, multi-head | `scaled_dot_product_attention`, `MultiHeadAttention` |
| `models/positional_encoding.py` | sin/cos table | `sinusoidal_table` |
| `models/layers.py` | embedding, LayerNorm, FFN, residual | `SublayerConnection` |
| `models/encoder.py` / `decoder.py` | layers and stacks | `EncoderLayer`, `DecoderLayer` |
| `models/transformer.py` | full model, masks, output layer | `Transformer.forward`, `make_tgt_mask` |
| `models/decoding.py` | greedy and beam search | `greedy_decode`, `beam_search` |
| `train.py` | Adam, Noam schedule, label smoothing, checkpointing | `noam_lr`, `run_epoch` |
| `evaluate.py` | BLEU, losses, latency, graphs, predictions | `main()` |
| `translate.py` | inference wrapper used by the app | `Translator.translate` |
| `app.py` | Streamlit UI | — |
| `tests/` | 28 unit tests, overfit test, UI tests | — |

**Commands:**
```bash
python prepare_data.py                      # data + tokenizer
python train.py --config small              # train (prints the active configuration)
python evaluate.py                          # BLEU, losses, graphs
python translate.py "A dog runs in the park."
python -m unittest discover -s tests -v     # tests
streamlit run app.py                        # demo (or: python -m streamlit run app.py)
```

---

## 10. Complexity and comparison

| | RNN | LSTM | Transformer |
|---|---|---|---|
| Processing | sequential | sequential | parallel |
| Cost per layer | O(n·d²) | O(n·d²) | **O(n²·d)** |
| Sequential steps | O(n) | O(n) | **O(1)** |
| Path between distant words | O(n) | O(n) | **O(1)** |
| Long-range memory | poor (vanishing gradient) | better (gates) | excellent |
| Word order | built in | built in | needs positional encoding |

Self-attention is cheaper than an RNN when n < d, which is true for sentences (n ≈ 15, d = 256).

---

## 11. Advantages and limitations

**Advantages:** parallel training; direct long-range connections; interpretable attention maps; one
reusable attention block; scales to huge models (BERT, GPT, T5 descend from this paper).

**Limitations of the architecture:** O(n²) memory and time in sentence length; relies on positional
encoding; generation is still sequential; post-LN needs warm-up; needs a lot of data.

**Limitations of this project:** only the small configuration was trained, on a CPU; Multi30k is small and
narrow-domain (image captions); single run with a single seed; no checkpoint averaging; no key/value
caching in decoding.

---

## 12. Paper vs your implementation

| Paper choice | In your project |
|---|---|
| Encoder–decoder, N = 6 | ✔ implemented; trained with N = 3 (small) |
| Scaled dot-product, 8 heads | ✔ |
| Sinusoidal positions | ✔ |
| Post-LN residual blocks | ✔ |
| Shared embeddings + output weights | ✔ (joint BPE vocabulary) |
| Adam (0.9, 0.98, 1e-9) + warm-up schedule | ✔ warm-up 1,500 (paper 4,000 for base) |
| Dropout 0.1, label smoothing 0.1 | ✔ |
| Beam 4, α 0.6, max length input + 50 | ✔ |
| Checkpoint averaging | ✘ (single best checkpoint) |
| WMT14 data, 8 GPUs, 100k steps | ✘ Multi30k, CPU, 3,405 steps |

---

## 13. Glossary

| Term | Meaning |
|---|---|
| Token / sub-word | a unit of text produced by BPE (e.g. `▁Hunde`) |
| BPE | byte-pair encoding: builds a vocabulary of frequent character sequences |
| Embedding | a learned vector representing a token |
| d_model | the size of every token vector inside the model (256) |
| Head | one of 8 parallel attention computations |
| Memory | the encoder output that the decoder attends to |
| Logits | raw scores before softmax |
| Teacher forcing | feeding the true previous words during training |
| Auto-regressive | generating one token at a time from previous outputs |
| Label smoothing | softening the one-hot training target |
| Warm-up | gradually increasing the learning rate at the start |
| Perplexity | exp(cross-entropy); an "effective number of choices" |
| BLEU | n-gram precision metric for translation (0–100) |
| chrF | character n-gram F-score |
| Checkpoint | a saved copy of the model weights |
| Overfitting | doing well on training data but worse on new data |

---

## 14. Viva demo script (about 3 minutes)

1. **Open the app** (`streamlit run app.py`). "This is my from-scratch Transformer, trained on Multi30k."
2. **Translate** *"Two dogs are playing in the snow."* → *"Zwei Hunde spielen im Schnee."* Point to the
   inference time and token count.
3. **Switch Greedy ↔ Beam.** "Beam search keeps four candidates and uses length penalty 0.6, as in the
   paper."
4. **Architecture section.** Walk the flow: tokenisation → positional encoding → encoder → decoder →
   linear + softmax.
5. **Performance section.** BLEU 37.09, test perplexity 6.0. "Multi30k results, not comparable with the
   paper's WMT14 scores."
6. **Attention heatmap.** "The diagonal pattern is a word alignment the model learned by itself: *einer*
   attends to *ladder*."
7. **Open `models/attention.py`.** Show the 4 lines implementing softmax(QKᵀ/√d_k)V and the mask.

---

## 15. One-page cheat sheet

```
Attention(Q,K,V)   = softmax(QKᵀ/√d_k + M) V             d_k = 256/8 = 32
MultiHead          = Concat(head_1..head_8) W^O
PE(pos,2i)         = sin(pos/10000^(2i/d)),  PE(pos,2i+1) = cos(…)
FFN(x)             = max(0, xW1+b1)W2+b2                 256 → 1024 → 256
Sublayer           = LayerNorm(x + Dropout(Sublayer(x)))  (post-LN)
Embedding          × √d_model = ×16 ; weights tied with output layer
lr                 = d^-0.5 · min(step^-0.5, step·warmup^-1.5)   peak 0.00161 at step 1500
Label smoothing    gold 0.9, rest 0.1/8000
Beam               k = 4, lp = ((5+|Y|)/6)^0.6, max len = src + 50
BLEU               = BP · exp(¼ Σ log p_n)
Perplexity         = exp(un-smoothed CE) = 6.00 (test)

Config: N=3+3, d=256, h=8, d_ff=1024, p=0.1, vocab 8k, 7,577,600 params
Data:   Multi30k 29,000 / 1,014 / 1,000
Train:  15 epochs, 3,405 steps, batch 128, ~75 min CPU
Result: BLEU 37.09 beam / 36.25 greedy, chrF 60.98, loss 2.256/2.881/2.858
```

---

## 16. Tricky questions examiners like (with answers)

1. **Why does the encoder also need a mask?** Batches are padded. Without the padding mask, real words
   would attend to meaningless `<pad>` tokens.
2. **What happens without positional encoding?** The model becomes order-blind: it treats the sentence as
   a bag of words.
3. **Why multiply embeddings by √d_model?** With the small initial embedding values (std d^−0.5), the
   scaling brings them to unit size, comparable to the positional encoding.
4. **Why 8 heads and not 1?** Several heads attend to different relations at the same cost; one head
   averages everything together (paper Table 3, row A).
5. **Can the decoder run in parallel?** During **training**, yes (teacher forcing + causal mask). During
   **inference**, no: it generates token by token.
6. **Difference between self- and cross-attention?** Where K and V come from: the same sequence vs the
   encoder output.
7. **Why did you train the small model?** No GPU. The base model takes about 5 s per batch on this CPU
   (8+ hours). Base is implemented and tested but not trained.
8. **What is post-LN vs pre-LN?** Post-LN (the paper) normalises after the residual addition. Pre-LN
   normalises the sub-layer input; it is more stable for deep models but is not the original design.
9. **Why is label smoothing good if it hurts perplexity?** It prevents over-confidence and improves
   generalisation and BLEU.
10. **Why is BLEU not a perfect metric?** It only counts exact n-gram overlap with one reference. A
    correct paraphrase (e.g. *Mann* for *Typ*) is penalised.
11. **How do you know there is no data leakage?** The tokenizer was trained on train only, the test set
    is used only in `evaluate.py`, and there is zero overlap between test and train sentences.
12. **What is weight tying and why use it?** The same matrix serves as input embedding and output
    projection. It saves 2 × 8000 × 256 parameters and couples input and output word meaning.

---

## 17. Self-test (answer without looking, then check above)

1. Write the attention formula and explain each symbol.
2. What is d_k in your model, and why?
3. Draw the causal mask for 3 target tokens.
4. Name the three uses of multi-head attention.
5. What are the shapes after embedding a batch of 128 sentences of length 20?
   *(Answer: 128 × 20 × 256)*
6. Compute the learning rate at step 1,500 for your model. *(0.00161)*
7. What is the difference between training loss and perplexity in your results?
8. Why is beam search better than greedy here, and by how much?
9. Explain why your BLEU is not comparable with the paper's.
10. Derive your parameter count (§4.10).

*For 30 more questions with model answers, see `viva_questions.md`; for the formal write-up, see `report.md`.*
