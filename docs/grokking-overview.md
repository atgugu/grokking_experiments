# The Grokking Phenomenon

An introduction to grokking, the modular addition task, and the trigonometric algorithm — aimed at researchers who haven't yet read the primary papers.

## What is Grokking?

Grokking is the phenomenon where a neural network first *memorizes* its training data — achieving perfect training accuracy — then, long after training loss has plateaued, suddenly *generalizes* to held-out test data. The term was coined by Power et al. (2022), who observed it on small algorithmic datasets like modular arithmetic and permutation composition.

This is surprising because conventional wisdom says generalization either happens alongside training loss improvement or not at all. In grokking, the network spends thousands of epochs at ~100% train accuracy and near-random test accuracy before a sharp phase transition pushes test accuracy to near-perfect. The gap between memorization and generalization can be 10-30x the memorization time.

## The Modular Addition Task

This project trains on **modular addition**: given two integers `a` and `b`, predict `(a + b) mod p` where `p = 113`.

**Why a prime?** Primes have the property that the integers modulo p form a *field* — every non-zero element has a multiplicative inverse. This means Fourier analysis over Z/pZ is clean: the p-th roots of unity are all distinct, and the Discrete Fourier Transform is a unitary transformation. Non-primes would introduce aliasing between frequency components.

**Why 113 specifically?** Large enough that the task is non-trivial (12,769 total input pairs) but small enough for exhaustive evaluation. All p^2 = 12,769 pairs can be enumerated to compute the full logit table for analysis.

**Input format.** Each input is a 3-token sequence `[a, b, =]` where `a` and `b` are integers in `{0, 1, ..., 112}` and `=` is a special token (index 113). The vocabulary has 114 tokens. The model predicts one of 113 output classes at the `=` position.

**Data split.** 30% of all (a, b) pairs are used for training (~3,831 pairs), 70% for testing (~8,938 pairs). This is an unusually small training set relative to the input space, which is what makes grokking possible — the network can memorize 3,831 facts before being forced to find a pattern.

## Model Architecture

The model is a **1-layer transformer** with the following specifications:

| Component | Value | Notes |
|-----------|-------|-------|
| Layers | 1 | Single transformer block |
| d_model | 128 | Residual stream dimension |
| Attention heads | 4 | d_head = 32 |
| d_mlp | 512 | MLP hidden dimension |
| Activation | ReLU | Not GELU — deliberate choice |
| LayerNorm | None | Omitted entirely |
| Embedding | Separate W_E and W_U | Not tied |
| Init | Xavier uniform | All weights |

**Why these choices?** The architecture is deliberately minimal. A single layer forces the entire algorithm into one attention + MLP pass. No LayerNorm simplifies the computational graph for mechanistic analysis (LayerNorm introduces nonlinear normalization that complicates Fourier interpretation). Separate embedding and unembedding matrices allow W_E and W_U to specialize for encoding and decoding respectively.

**Forward pass:**
1. Token embedding: `h = W_E[tokens] + W_pos[positions]` — shape `(batch, 3, 128)`
2. Self-attention: all 3 positions attend to each other (4 heads, residual connection)
3. MLP: `h = h + W_out @ ReLU(W_in @ h)` — 512-dimensional hidden layer
4. Extract position 2 (the `=` token): `h_eq = h[:, 2, :]` — shape `(batch, 128)`
5. Unembedding: `logits = W_U @ h_eq` — shape `(batch, 113)`

## The Trigonometric Algorithm

The central finding of Nanda et al. (2023) is that after grokking, the network has learned a **trigonometric algorithm** for modular addition. Here is what the network computes, described at a high level:

### Step 1: Fourier Embedding

The token embedding W_E maps each integer `a` to a vector whose components encode `cos(2*pi*k*a/p)` and `sin(2*pi*k*a/p)` for a small set of "key frequencies" k. Geometrically, the integers 0 through 112 are arranged as evenly-spaced points on circles — one circle per key frequency.

### Step 2: Attention Routing

The attention mechanism copies information from positions 0 (the `a` token) and 1 (the `b` token) to position 2 (the `=` token). After attention, the residual stream at position 2 contains Fourier representations of both `a` and `b`.

### Step 3: MLP Computation

The MLP neurons exploit a trigonometric identity. For a given frequency k:

    cos(2*pi*k*(a+b)/p) = cos(2*pi*k*a/p) * cos(2*pi*k*b/p) - sin(2*pi*k*a/p) * sin(2*pi*k*b/p)

Individual neurons compute terms like `ReLU(cos(2*pi*k*a/p) + cos(2*pi*k*b/p))`, which (combined with other neurons handling the negative terms) reconstruct the sum-of-angles. Each neuron specializes in a single frequency k.

### Step 4: Unembedding

The unembedding matrix W_U converts the Fourier representation of `(a+b) mod p` back to logits over the 113 output classes. Since the output classes are themselves integers mod p, this is essentially an inverse Fourier transform restricted to the key frequencies.

### Key Frequencies

Not all 113 possible frequencies matter. After grokking, only **3-5 key frequencies** carry significant energy (typically around {14, 35, 41, 42, 52} for p=113). These are the frequencies the network uses; all others are noise. The Gini coefficient measures how concentrated the energy is in these few frequencies.

### Fourier Decomposition

The logit table L[a, b, c] can be decomposed via 2D DFT into frequency components L_hat[k1, k2]. For the trained network:

- **Restricted logits** L_restricted: keep only DC (k=0) and key frequency components → these alone achieve low loss
- **Excluded logits** L_excluded = L - L_restricted: everything else → this is noise that doesn't help prediction

The restricted logits suffice for accurate prediction because the trigonometric algorithm only uses the key frequencies.

## Three Phases of Training

Training proceeds through three distinct phases:

### Phase 1: Memorization (epochs ~0–300)

- **Train accuracy** rises quickly to ~100%
- **Test accuracy** stays near random (~1/113 ≈ 0.9%)
- The network memorizes all 3,831 training pairs as a lookup table
- Fourier spectrum is diffuse — energy spread across all frequencies
- Weight norms grow as the network stores individual facts

### Phase 2: Circuit Formation (epochs ~300–3,000)

- **Train accuracy** remains at 100%
- **Test accuracy** remains near random, but begins to climb very slowly
- Under the surface, key frequency energy grows monotonically
- Gini coefficient rises steadily (frequency spectrum becomes sparser)
- Weight norm begins to *decrease* — weight decay is compressing the representation
- Restricted loss decreases (useful algorithm forming) while excluded loss may increase (noise being pushed out)

> **Note:** This is the critical phase. Standard metrics (train/test accuracy) show nothing interesting, but Fourier-based progress measures reveal steady algorithmic development. This is the key insight of Nanda et al.

### Phase 3: Generalization (epochs ~3,000–5,000)

- **Test accuracy** undergoes a sharp phase transition, jumping from near-random to ~95%+
- Gini coefficient reaches ~0.9 (highly sparse spectrum)
- Weight norm has decreased substantially
- The trigonometric algorithm has become strong enough to outperform memorization on test data
- After this point, both train and test accuracy converge to ~100%

## The Role of Weight Decay

Weight decay (wd=1.0 in this project) is **essential** for grokking. It provides the regularization pressure that forces the network to transition from memorization to a generalizable algorithm.

**Why it works:** A memorized lookup table requires large weights to store ~3,831 independent facts. The trigonometric algorithm requires only 3-5 frequency components — a far more compact representation. Weight decay penalizes the L2 norm of all parameters, continuously pushing the network toward smaller-weight solutions. Over time, this pressure tips the balance: the compact trigonometric circuit becomes lower-cost than the memorized table.

**Without weight decay** (wd=0), the network memorizes and never generalizes. The memorized solution has no reason to simplify. With **too much weight decay**, the network may not have enough capacity to memorize in the first place, preventing the initial phase that provides the training signal.

The value wd=1.0 is deliberately high — it creates strong regularization pressure that accelerates the transition from memorization to generalization. Different weight decay values change the timing but not the qualitative behavior (as long as it's large enough to force simplification).

## References

- Power, A., Burda, Y., Edwards, H., Babuschkin, I., & Misra, V. (2022). Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)
- Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). Progress Measures for Grokking via Mechanistic Interpretability. ICLR 2023. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)
- Zhong, Z., Liu, Z., Tegmark, M., & Andreas, J. (2023). The Clock and the Pizza. NeurIPS 2023. [arXiv:2306.17844](https://arxiv.org/abs/2306.17844)
- Michaud, E.J., Liu, Z., & Tegmark, M. (2023). The Quantization Model of Neural Scaling. [arXiv:2303.13506](https://arxiv.org/abs/2303.13506)

See [references.md](./references.md) for the full annotated bibliography.
