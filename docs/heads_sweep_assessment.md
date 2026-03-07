# Critical Assessment: Is the n_heads Sweep Finding Publishable?

## Our Experimental Result

We swept n_heads ∈ {1, 2, 4, 8, 16} at fixed p=113, d_model=128, wd=1.0, lr=1e-3.
All 5 runs grokked. De-mirrored top-10 frequency count = 5 universally. We claimed
this refutes "Nanda's 1-head-per-frequency story" and proves task-determines-frequencies.

## Problem 1: The Hypothesis We Tested Is a Strawman

**Nanda never claims 1 head = 1 frequency.** The actual paper (arxiv 2301.05217) says:

> "Two of the attention heads approximately compute degree-2 polynomials of sines
> and cosines of a particular frequency (and the other two are used to increase the
> magnitude of the input embeddings)."

So in Nanda's own 4-head model: 2 heads do trig, 2 heads amplify embeddings. There
is no "1 head per frequency" claim anywhere in the paper. The paper also **does not
explain why 5 frequencies emerge** — it's purely empirical. The "1 head = 1 freq"
story is a folk simplification that circulates informally but doesn't appear in the
literature. We tested and "refuted" something nobody claimed.

## Problem 2: Existing Theory Already Predicts This

**McCracken et al. (2025) — "Uncovering a Universal Abstract Algorithm for Modular
Addition in Neural Networks" (arxiv 2505.18266):**
- Proves the universal algorithm uses O(log n) frequencies (for p=113, log₂(113) ≈ 7)
- Already sweeps 1-4 layer MLPs AND 1-4 layer transformers, varying width up to 2048
- Shows depth and trainable embeddings affect frequency count, but the algorithm is universal
- Our finding (≈5 frequencies for all head counts) is a special case of their result

**Gromov (2023) — "Grokking modular arithmetic" (arxiv 2301.02679):**
- Shows the solution is determined by the group structure of ℤ/pℤ
- The Fourier basis frequencies are properties of the algebraic task, not the architecture

**Li et al. (2024) — "Fourier Circuits in Neural Networks" (arxiv 2402.09469):**
- Shows margin maximization determines which frequencies emerge
- Each neuron aligns with a specific frequency; this is independent of head count

**Mohamadi et al. (2024) — "Why Do You Grok?" (arxiv 2407.12332):**
- Theoretical analysis showing generalization is driven by regularization, not architecture

## Problem 3: Our "Always 5" Metric Has Issues

Our main claim rests on "de-mirrored top-10 frequency count = 5 universally."

- Frequencies in ℤ/pℤ come in conjugate pairs (f, p−f). So top-10 raw frequencies
  → at most 5 unique pairs. If the spectrum has ANY concentration (which it does —
  Gini > 0.94 for all runs), you'll always get exactly 5 from top-10. This is close
  to a mathematical tautology, not an empirical finding.

- The 90% energy threshold (a more honest metric) gives K = {8, 4, 6, 4, 5} —
  variable, NOT constant. n_heads=1 needs K=8 frequencies for 90% energy, while
  n_heads=2 and n_heads=8 need only K=4. This variation is actually more interesting
  than the "always 5" headline, but we didn't analyze it.

## Problem 4: The Interesting Signal We Missed

There ARE potentially interesting observations we didn't pursue:

1. **n_heads=1 grokked fastest (epoch 3200)** while n_heads=4 was slowest (8350).
   Why? The mechanistic story of how a single-head model implements 5+ frequencies
   via superposition would be genuinely novel. But we only have the summary metric.

2. **K(90%) varies from 4 to 8 across head counts** — this suggests the spectral
   concentration depends on architecture even if the "winning" frequencies don't.
   This connects to the representation efficiency question.

3. **What do the attention patterns look like for n_heads=1 vs 16?** We generated
   no mechanistic analysis at the circuit level.

## Verdict: Not Publishable As-Is

**The finding is not novel enough for publication because:**
- The hypothesis tested is a strawman (Nanda doesn't claim 1:1 head:freq)
- The result is already predicted by existing theory (O(log n) frequencies, task-determined)
- McCracken et al. (2025) already did more comprehensive architecture sweeps
- Our key metric ("always 5") has a measurement circularity issue
- We lack the mechanistic depth that would make this a real contribution

**What would make it publishable (if extended significantly):**
- Deep mechanistic analysis of HOW n_heads=1 implements multi-frequency via superposition
- Joint (n_heads × weight_decay × p) sweep to disentangle all three factors
- Connection to the O(log p) theory — does frequency count scale with p given fixed heads?
- Circuit-level attention pattern analysis showing how the algorithm adapts to head count
- Comparison of which specific frequencies are selected (do they differ by head count?)

## Key References

- Nanda et al. (2023) "Progress measures for grokking" — arxiv 2301.05217
- Gromov (2023) "Grokking modular arithmetic" — arxiv 2301.02679
- Li et al. (2024) "Fourier Circuits in Neural Networks" — arxiv 2402.09469
- Mohamadi et al. (2024) "Why Do You Grok?" — arxiv 2407.12332
- McCracken et al. (2025) "Universal Abstract Algorithm for Modular Addition" — arxiv 2505.18266
- Rangamani (2025) "Low Rank Sparse Fourier in RNNs" — arxiv 2503.22059
- (2025) "On the Mechanism and Dynamics of Modular Addition" — arxiv 2602.16849
