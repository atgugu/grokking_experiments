# Follow-Up Experiment Ideas

Promising directions extracted from the [heads sweep assessment](heads_sweep_assessment.md).
Each is scoped to be actionable with the existing codebase.

---

## 1. Superposition Mechanics in the Single-Head Model

**Question:** How does a single attention head implement 5+ Fourier frequencies
simultaneously? This must involve superposition — encoding multiple frequencies
in overlapping directions within the residual stream.

**Why it matters:** n_heads=1 grokked *fastest* (epoch 3200 vs 8350 for n_heads=4)
despite having no ability to dedicate separate heads to separate frequencies. The
mechanistic story here is genuinely unexplored. McCracken et al. (2025) sweep
architecture but don't do per-head circuit analysis.

**Protocol:**
- Load the saved n_heads=1 checkpoint (`results/p113_d128_h1_mlp512_L1_wd1.0_s42/`)
- Use `ActivationCache` to extract attention patterns, QK/OV circuits
- Project W_E, W_U, and MLP weights onto the Fourier basis
- Measure interference between frequency components in the single head's OV matrix
- Compare representational geometry (cosine similarity of frequency directions) to the
  n_heads=4 and n_heads=16 models
- Visualize how the single head's attention pattern differs from multi-head models

**Expected output:** Figures showing frequency multiplexing in the OV circuit;
quantitative measure of superposition (e.g., subspace angles between frequency
representations).

---

## 2. Spectral Concentration vs Architecture: Analyzing K(90%)

**Question:** Why does the 90%-energy frequency count K vary from 4 to 8 across
head counts (h=1→8, h=2→4, h=4→6, h=8→4, h=16→5)?

**Why it matters:** While the "winning" frequencies are task-determined, the spectral
*concentration* appears architecture-dependent. This is a cleaner, more defensible
finding than "always 5." It connects to representation efficiency: fewer heads may
force the model to spread energy across more frequencies.

**Protocol:**
- For each saved checkpoint, compute the full 2D Fourier spectrum of the logit table
- Plot energy vs frequency rank (cumulative energy curve) for each n_heads value
- Measure K at thresholds {80%, 85%, 90%, 95%} to see if the pattern is robust
- Extend to a finer grid: n_heads ∈ {1, 2, 3, 4, 6, 8, 12, 16} (needs new training runs)
- Correlate K with Gini coefficient and grokking speed
- Test with multiple seeds to separate signal from noise

**Expected output:** Plot of K(threshold) vs n_heads showing whether spectral
concentration reliably depends on head count.

---

## 3. Joint (n_heads × p) Sweep: Frequency Scaling

**Question:** Does the number of learned frequencies scale as O(log p) regardless
of n_heads, as McCracken et al. predict?

**Why it matters:** McCracken et al. (2025) prove the O(log n) bound but only verify
it by sweeping architecture at fixed p. We could contribute the *complementary*
experiment: sweeping p at multiple head counts to test whether the scaling law
holds across architectures.

**Protocol:**
- Select p ∈ {13, 23, 43, 59, 89, 113} (spanning the grokking phase transition)
- For each p, train with n_heads ∈ {1, 2, 4, 8}
- Use consistent hyperparameters: wd=1.0, lr=1e-3, 40K epochs
- Measure K (de-mirrored, 90% energy threshold) and n_key_freq for each (p, n_heads) pair
- Fit K vs log(p) for each n_heads — check if slope is constant
- Check if the grokking phase transition (between p=43 and p=59) shifts with n_heads

**Expected output:** 2D heatmap of K(p, n_heads); fitted scaling curves; clear
statement on whether architecture affects the O(log p) constant.

**Note:** This is ~24 training runs. With existing infrastructure, doable in a few
hours on a single GPU.

---

## 4. Circuit-Level Attention Analysis Across Head Counts

**Question:** What do the attention patterns actually look like for 1-head vs
16-head models? Do redundant heads specialize, or do they all learn the same pattern?

**Why it matters:** Nanda et al. found 2 trig heads + 2 amplification heads in the
4-head model. With 16 heads, there are 12 "extra" heads. Understanding what they
do (redundancy? different frequencies? noise?) would clarify how transformers
allocate capacity.

**Protocol:**
- Load checkpoints for n_heads ∈ {1, 4, 16}
- Extract attention patterns (using `ActivationCache`) on the full input distribution
- For each head, compute: (a) entropy of attention pattern, (b) projection of OV
  matrix onto Fourier modes, (c) contribution to final logit
- Classify heads as: trig (aligned to specific frequency), amplification (uniform
  attention boosting embeddings), or inactive (near-zero logit contribution)
- For the 16-head model: cluster heads by function, check for redundancy

**Expected output:** Head classification table; attention pattern visualizations;
quantitative breakdown of how many heads serve each role as total head count grows.
