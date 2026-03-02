# Analysis Pipeline

Technical reference for the Fourier analysis, progress measures, and neuron-level analysis implemented in `src/analysis/`.

## Pipeline Overview

```
Train model (scripts/train_single.py)
    |
    v
Load final model + checkpoints (scripts/analyze_run.py)
    |
    v
Compute full logit table: L[a, b, c] for all (a, b) pairs
    |
    v
2D Fourier decomposition of L
    |
    +---> Frequency norms (marginal) ---> Identify key frequencies (top-5)
    |                                          |
    |                                          v
    +---> Restricted logits (DC + key freqs only)
    |         |
    |         +---> Restricted loss (CE on training pairs)
    |
    +---> Excluded logits (everything else)
    |         |
    |         +---> Excluded loss (CE on training pairs)
    |
    +---> Gini coefficient (sparsity of frequency distribution)
    |
    v
Neuron analysis
    |
    +---> Per-neuron activation grid: act[a, b, n] for all (a, b)
    |         |
    |         +---> 2D DFT per neuron ---> Dominant frequency + R^2
    |         |
    |         +---> Frequency clustering (group neurons by dominant freq)
    |
    +---> Neuron-logit map: W_L = W_out @ W_U
    |
    +---> Per-neuron frequency spectrum
    |
    v
Visualization (scripts/generate_figures.py)
    |
    v
20 figures + 1 animated GIF
```

## Fourier Decomposition

All Fourier analysis lives in `src/analysis/fourier.py`.

### DFT Matrix Construction

The unitary DFT matrix for Z/pZ:

    F[k, n] = exp(-2*pi*i*k*n / p) / sqrt(p)

where k is the frequency index and n is the spatial index, both in {0, ..., p-1}. The `1/sqrt(p)` normalization makes F unitary: `F @ F*.T = I`. This is computed by `dft_matrix(p)`.

### 2D DFT of the Logit Table

The model produces a logit table `L[a, b, c]` of shape `(p, p, p)` — the logit for output class `c` given input `(a, b)`. For each output class `c`, we compute the 2D DFT:

    L_hat[k1, k2, c] = (F @ L[:, :, c] @ F.T)

This decomposes the (a, b) dependence into Fourier components. `L_hat[k1, k2, c]` measures how much the logit for class `c` oscillates at frequency `k1` in `a` and frequency `k2` in `b`.

In the code, this is vectorized over all classes simultaneously using `einsum`:

```python
L_hat = np.einsum("ij,jkc,lk->ilc", F, logit_table, F)
```

### Component Norms

`compute_fourier_component_norms(logit_table, p)` returns:

- **`component_norms[k1, k2]`**: Total energy at frequency pair `(k1, k2)`, summed over all output classes. Shape `(p, p)`.
- **`frequency_norms[k]`**: Marginal energy for frequency `k`. Computed as the sum of all components where `k` appears in either dimension: `sum_over_k2(norms[k, k2]) + sum_over_k1(norms[k1, k]) - norms[k, k]` (subtracting the diagonal to avoid double-counting). Shape `(p,)`.
- **`per_class_norms[k1, k2, c]`**: Energy per output class per frequency pair. Shape `(p, p, p)`.

### Key Frequency Identification

`identify_key_frequencies(frequency_norms, n_top=5)` returns the top-5 non-DC frequencies by marginal energy. The DC component (k=0) is excluded because it represents the mean logit, not algorithmic structure.

For p=113, the expected key frequencies are approximately {14, 35, 41, 42, 52}.

### Restricted and Excluded Logits

Given key frequencies K, the Fourier components split into two groups:

- **Restricted**: components where *both* k1 and k2 are in {0} ∪ K
- **Excluded**: everything else

`compute_restricted_logits(logit_table, p, key_freqs)` builds a binary mask over the 2D frequency grid, applies it to the DFT coefficients, then inverse-transforms:

```
mask[k1, k2] = 1 if (k1 in {0} ∪ K) and (k2 in {0} ∪ K), else 0
L_restricted = F_inv @ (L_hat * mask) @ F_inv.T
```

`compute_excluded_logits` simply returns `L - L_restricted`.

> **Note:** The restricted logits alone should achieve low cross-entropy loss on training data after grokking. If they don't, the model hasn't learned a clean Fourier algorithm.

### Embedding Fourier Analysis

`fourier_embed_analysis(W_E, p)` applies 1D DFT along the token dimension of the embedding matrix:

```
E_hat = F @ W_E[:p, :]    # (p, d_model) complex
```

This reveals which Fourier frequencies are encoded in the embedding space. After grokking, the same key frequencies that dominate the logit table should also appear prominently in the embedding spectrum.

### Gini Coefficient

`compute_gini_coefficient(frequency_norms)` measures how concentrated the frequency energy distribution is:

- **Gini = 0**: all frequencies have equal energy (uniform distribution, no structure)
- **Gini = 1**: all energy is in a single frequency (maximally sparse)

Computed via the Lorenz curve:

```
sorted_norms = sort(frequency_norms)
gini = 1 - 2 * sum(cumsum(sorted_norms)) / (n * sum(sorted_norms)) + 1/n
```

During grokking, Gini increases from ~0.3 (random model) to ~0.9 (sparse trig algorithm).

## Progress Measures

Implemented in `src/analysis/progress_measures.py`.

`compute_all_progress_measures(model, train_inputs, train_targets, p)` computes all four Nanda et al. progress measures in a single call:

### 1. Restricted Loss

Cross-entropy loss computed using only the restricted logits (DC + key frequency components) on the training data. Tracks how well the useful part of the Fourier spectrum predicts training labels.

- **Decreases during circuit formation** — the trigonometric algorithm is learning
- Should approach the full model's training loss after grokking

### 2. Excluded Loss

Cross-entropy loss using only the excluded logits (non-key frequency components) on the training data. Tracks how much the non-algorithmic "noise" contributes to prediction.

- **Increases during circuit formation** — weight decay is suppressing noise components
- High excluded loss = the noise frequencies don't help prediction (good)

### 3. Gini Coefficient

Sparsity measure of the marginal frequency energy distribution. See [Fourier Decomposition](#gini-coefficient) above.

### 4. Weight Norm

L2 norm of all model parameters: `sqrt(sum(param^2 for all params))`.

- **Increases during memorization** — storing facts requires large weights
- **Decreases during circuit formation** — weight decay compresses to the compact trig algorithm
- The weight norm decrease is *the* signal that regularization is doing its job

> **Note:** The critical insight is that restricted loss and Gini change monotonically during the circuit formation phase, even though test accuracy shows no improvement. These are the "progress measures" that make grokking predictable.

## Neuron Analysis

Implemented in `src/analysis/neuron_analysis.py`.

The MLP has 512 neurons. After grokking, each neuron specializes in computing a single Fourier frequency. The analysis identifies which frequency each neuron computes and how cleanly.

### Activation Grid

For each neuron `n`, we record its post-ReLU activation for every (a, b) pair, producing a `(p, p)` grid. This is extracted using the `ActivationCache` context manager from `src/models/hooks.py`, which hooks into the MLP's output.

The full activation tensor has shape `(p, p, d_mlp)` = `(113, 113, 512)`.

### Frequency Classification

`classify_neuron_frequencies(neuron_activations, p, threshold=0.5)` processes each neuron:

1. Compute 2D DFT of the neuron's `(p, p)` activation grid
2. Compute marginal frequency energy for each frequency k
3. Identify the dominant (non-DC) frequency: `k_dom = argmax(freq_energy[1:])`
4. Compute R^2: fraction of total energy in the dominant frequency

Returns:
- **`dominant_freq[n]`**: which frequency neuron `n` computes
- **`r_squared[n]`**: how cleanly (R^2 > 0.5 = well-classified)
- **`clusters`**: dict mapping frequency → list of neuron indices (only neurons with R^2 ≥ threshold)

### Neuron-Logit Map

`compute_neuron_logit_map(W_U, W_out)` computes `W_L = W_out @ W_U`, the composite map from MLP neurons directly to output logits. Shape: `(d_mlp, p)` = `(512, 113)`.

Row `n` of W_L shows how neuron `n` contributes to each of the 113 output classes. After grokking, neurons tuned to frequency k produce sinusoidal patterns in their W_L row, oscillating at frequency k over the output classes.

### Per-Neuron Frequency Spectrum

`compute_neuron_frequency_spectrum(neuron_activations, p)` returns a `(d_mlp, p)` array where `spectrum[n, k]` is the marginal energy of frequency k in neuron n's activation grid.

After grokking, each row should have energy concentrated in exactly one column (the neuron's dominant frequency), producing a block-diagonal-like pattern when neurons are sorted by frequency.

## Key Data Shapes

| Variable | Shape | Description |
|----------|-------|-------------|
| `logit_table` | `(113, 113, 113)` | `[a, b, c]` = logit for class c given (a, b) |
| `component_norms` | `(113, 113)` | 2D Fourier energy at each (k1, k2) pair |
| `frequency_norms` | `(113,)` | Marginal frequency energy |
| `key_freqs` | `(5,)` | Indices of top-5 non-DC frequencies |
| `restricted_logits` | `(113, 113, 113)` | Logits from DC + key freq components only |
| `excluded_logits` | `(113, 113, 113)` | Logits from non-key freq components only |
| `W_E` | `(114, 128)` | Token embedding (113 integers + 1 equals token) |
| `W_pos` | `(3, 128)` | Positional embedding (positions 0, 1, 2) |
| `W_U` | `(128, 113)` | Unembedding matrix |
| `neuron_activations` | `(113, 113, 512)` | Post-ReLU MLP activations on full grid |
| `neuron_logit_map` | `(512, 113)` | W_out @ W_U composite weights |
| `neuron_spectrum` | `(512, 113)` | Per-neuron marginal frequency energy |
| `dominant_freq` | `(512,)` | Dominant frequency per neuron |
| `r_squared` | `(512,)` | Fraction of energy in dominant freq per neuron |
| `gini` | scalar | Frequency sparsity in [0, 1] |
| `weight_norm` | scalar | L2 norm of all parameters |
