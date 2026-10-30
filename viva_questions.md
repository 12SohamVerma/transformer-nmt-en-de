# Viva Questions & Answers – Transformer NMT Mini-Project

Experiments 9 & 10 · Computational Linguistics and NLP Lab
Paper: *Attention Is All You Need* (Vaswani et al., 2017)

---

## A. Transformer architecture

**1. What problem does the Transformer solve compared to RNN-based seq2seq models?**
RNNs process tokens one after another, so training cannot be parallelised across time steps, and information between distant words must pass through many recurrent steps. The Transformer removes recurrence and relies only on attention. Every token attends to every other token in one step (path length O(1)), and all positions are computed in parallel.

**2. Describe the overall Transformer architecture.**
It is an encoder–decoder model. The encoder is a stack of N identical layers, each with multi-head self-attention followed by a position-wise feed-forward network. The decoder is a stack of N layers, each with masked self-attention, encoder–decoder (cross) attention, and a feed-forward network. Every sub-layer is wrapped as `LayerNorm(x + Dropout(Sublayer(x)))`. Token embeddings plus positional encodings form the input, and a final linear layer followed by softmax produces next-token probabilities.

**3. What is the "base" configuration in the paper?**
N = 6 encoder and 6 decoder layers, d_model = 512, h = 8 heads, d_k = d_v = 64, d_ff = 2048, dropout 0.1, label smoothing 0.1, about 65M parameters with a 37k vocabulary. In our code this is `MODEL_CONFIGS["base"]` in `config.py`, and a unit test checks it against the paper.

**4. Why did you train a "small" configuration?**
The lab machine has no CUDA GPU. The base model (48M parameters with our 8k vocabulary) needs about 5 s per batch on the laptop CPU, which would mean 8+ hours for 25 epochs. The small model (3+3 layers, d_model = 256, 8 heads, d_ff = 1024, 7.6M parameters) uses exactly the same code and only different hyperparameters. The base configuration is implemented, tested, and selectable with `--config base`.

**5. What are residual connections and why are they used?**
The input of a sub-layer is added to its output: `x + Sublayer(x)`. This gives gradients a direct path through deep stacks (avoiding vanishing gradients). It also means each layer only has to learn a "correction" to its input.

**6. What is Layer Normalization? Why not Batch Normalization?**
LayerNorm normalises each token's feature vector to zero mean and unit variance, then applies a learned scale γ and shift β. It does not depend on the batch size or on sequence padding, and it behaves the same at training and inference time. BatchNorm statistics are unreliable for variable-length sequences. Our implementation is `models/layers.py → LayerNorm`, and a test checks it against `nn.LayerNorm`.

**7. What is "post-LN", and what did the original paper use?**
The original paper uses post-LN: `LayerNorm(x + Dropout(Sublayer(x)))`, which normalises after the residual addition. Many later models use pre-LN (`x + Sublayer(LayerNorm(x))`) because it trains more stably without warmup. We implemented post-LN to follow the paper (`SublayerConnection`).

**8. What does the position-wise feed-forward network do?**
`FFN(x) = max(0, xW₁ + b₁)W₂ + b₂` is applied to each position separately with the same weights. It expands to d_ff (2048 in base), applies ReLU, and projects back to d_model. It adds non-linearity and per-token processing capacity after attention has mixed information across tokens.

## B. Attention mechanism

**9. Write the scaled dot-product attention formula and explain each term.**
`Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`. Q (queries) says what each position is looking for, K (keys) says what each position offers, and V (values) is the content returned. QKᵀ gives similarity scores, softmax turns each row into weights that sum to 1, and those weights average the values.

**10. Why divide by √d_k?**
For random vectors with unit-variance components, the dot product of two d_k-dimensional vectors has variance d_k. Large scores push softmax into saturated regions where gradients are almost zero. Dividing by √d_k keeps the variance at about 1.

**11. What is multi-head attention and why is it better than single-head?**
The model projects Q, K, and V into h lower-dimensional subspaces (d_k = d_model/h), runs attention in each subspace in parallel, concatenates the results, and projects them with W^O. Different heads can learn different relations, such as syntax, word alignment, or neighbouring words. A single head averages them all together. The total computation is similar to one full-dimension head.

**12. How is multi-head attention implemented efficiently in your code?**
One `nn.Linear(d_model, d_model)` holds the projections for all heads. Its output is reshaped from (B, L, d_model) to (B, h, L, d_k) so that all heads are computed with a single batched matrix multiplication. A unit test confirms the result equals the explicit per-head `Concat(head₁…head_h)W^O`.

**13. What is the role of softmax in the Transformer?**
It appears in two places. (1) Inside attention, softmax turns similarity scores into a probability distribution over positions (the attention weights). (2) At the output, softmax over the vocabulary logits gives P(next token | previous tokens, source), which is used for the loss and for decoding.

**14. What is the computational complexity of self-attention?**
O(n² · d) per layer, where n is the sequence length: every token attends to every other token. This is fast for sentence-length inputs (n < d) but becomes a bottleneck for very long sequences.

## C. Encoder vs decoder; self- vs cross-attention

**15. What is the difference between the encoder and the decoder?**
The encoder reads the whole source sentence bidirectionally and produces contextual representations ("memory"). The decoder generates the target auto-regressively, one token at a time. It has masked self-attention so that it cannot see future target tokens, plus cross-attention that reads the encoder memory.

**16. Self-attention vs cross-attention?**
In self-attention, Q, K, and V all come from the same sequence (source with source, or target prefix with target prefix). In cross-attention (encoder–decoder attention), Q comes from the decoder while K and V come from the encoder output. This lets every German token look at the relevant English words, and it plays the role of the alignment model in classic NMT.

**17. What is teacher forcing?**
During training the decoder input is the ground-truth target shifted right (`<s> y₁ … y_{n-1}`), not the model's own predictions, and the labels are `y₁ … y_n </s>`. This allows all positions to be trained in parallel in one forward pass. In `train.py`, `tgt_in = tgt[:, :-1]` and `tgt_out = tgt[:, 1:]`.

**18. How does inference differ from training?**
At inference no target is available, so decoding is auto-regressive: start with `<s>`, predict a token, append it, and repeat until `</s>` or a maximum length. We implement greedy decoding (argmax at each step) and beam search (k = 4, length penalty α = 0.6, as in the paper) in `models/decoding.py`.

## D. Positional encoding

**19. Why does the Transformer need positional encoding?**
Attention is permutation-invariant: without position information, "dog bites man" and "man bites dog" would produce the same set of representations. Positional encodings add word-order information to the embeddings.

**20. Explain the sinusoidal positional encoding formula.**
`PE(pos, 2i) = sin(pos / 10000^{2i/d_model})` and `PE(pos, 2i+1) = cos(pos / 10000^{2i/d_model})`. Each pair of dimensions is a sinusoid with a different wavelength, ranging geometrically from 2π to 10000·2π. For any fixed offset k, PE(pos+k) is a linear function (a rotation) of PE(pos), so relative positions are easy to learn. The encoding is fixed rather than learned, and it can in principle extend to lengths not seen in training.

**21. Why are embeddings multiplied by √d_model?**
Our embeddings are initialised with standard deviation d_model^{-0.5} and share weights with the output layer. Scaling by √d_model brings them to unit variance, the same scale as the positional encoding (values in [-1, 1]), so position information neither dominates nor disappears.

## E. Masking

**22. What masks are used and why?**
(1) A **padding mask** sets attention to `<pad>` positions to zero, so batching sentences of different lengths does not change the result. It is used in encoder self-attention, decoder self-attention, and cross-attention. (2) A **causal (look-ahead) mask** is a lower-triangular matrix in decoder self-attention, so position i only sees positions ≤ i. Training then matches inference, where future tokens are unknown.

**23. How is masking implemented numerically?**
Before the softmax, blocked scores are replaced by a very large negative number (`torch.finfo.min`), so exp(score) ≈ 0 and those positions get zero weight. We use a finite value rather than −∞ so that a fully masked row cannot produce NaN.

**24. How did you verify the masks are correct?**
Unit tests in `tests/test_components.py`: (a) changing a future target token leaves all earlier decoder outputs unchanged, which shows the causal mask works; (b) appending `<pad>` tokens to the source leaves the logits unchanged, which shows the padding mask works; (c) padded keys receive exactly zero attention weight.

## F. Training, BLEU, dataset

**25. Explain the learning-rate schedule.**
`lrate = d_model^{-0.5} · min(step^{-0.5}, step · warmup^{-1.5})` (the "Noam" schedule). The learning rate increases linearly for `warmup` steps, then decays as 1/√step. Warmup is important for post-LN Transformers because large early updates destabilise training. We use Adam with β₁ = 0.9, β₂ = 0.98, ε = 10⁻⁹, as in the paper.

**26. What is label smoothing and why is it used?**
Instead of a one-hot target, the model is trained toward a distribution that puts 1−ε on the correct token and spreads ε = 0.1 over the vocabulary. This stops the model from becoming over-confident. The paper notes it hurts perplexity but improves BLEU. We use `nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=PAD)`.

**27. What is BLEU and how did you compute it?**
BLEU = BP · exp(Σ ¼ log pₙ), where pₙ is the modified n-gram precision for n = 1..4 and BP is a brevity penalty for outputs shorter than the reference. It ranges from 0 to 100 (higher is better). We use **SacreBLEU** on detokenised output against the raw references, so the score is reproducible and its signature is reported. We do not use our own BPE tokens for scoring.

**28. Describe the dataset and preprocessing.**
Multi30k: English Flickr30k image captions with German translations. Train has 29,000 pairs, validation 1,014, and test 1,000 (test_2016_flickr). The pipeline downloads and verifies the splits (line counts, alignment), cleans the text (Unicode NFC, whitespace), and trains a joint 8k SentencePiece BPE model **on the training split only**. It then encodes the text (source + `</s>`, `<s>` + target + `</s>`), pads, and batches sentences of similar length together. The test set is used only in `evaluate.py`.

**29. Compare RNN, LSTM, and Transformer.**
| | RNN | LSTM | Transformer |
|---|---|---|---|
| Processing | sequential | sequential | parallel over positions |
| Long-range dependencies | poor (vanishing gradients) | better (gates, cell state) | direct, O(1) path length |
| Training speed | slow | slow | fast on GPUs |
| Per-layer complexity | O(n·d²) | O(n·d²) | O(n²·d) |
| Word order | built in | built in | needs positional encoding |

**30. What are the limitations of the original Transformer?**
(1) The O(n²) memory and time of attention makes very long documents expensive. (2) There is no inherent notion of order, so it relies on positional encodings, and sinusoidal ones extrapolate poorly in practice. (3) It needs large data and compute; on small corpora like Multi30k it overfits without regularisation. (4) Post-LN needs careful warmup and is unstable when deep. (5) Auto-regressive decoding is still sequential at inference time. (6) A fixed context length means sentences longer than `max_len` must be truncated.
