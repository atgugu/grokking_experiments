# Visualization Guide

Complete reference for every visualization function in this project. Organized by topic, matching the figure generation pipeline in `scripts/generate_figures.py`.

Each section follows the pattern: **What it shows** — **How to read it** — **What to watch for** — **What it means if something looks wrong**.

For background on the Fourier analysis, see [Analysis Pipeline](./analysis-pipeline.md). For the scientific context, see [Grokking Overview](./grokking-overview.md).

---

## Training Dynamics

Three plots that show the high-level training story. Module: `src/viz/training_curves.py`.

### `plot_grokking_curves`

**Source:** `plot_grokking_curves(history: dict) -> plt.Figure`
**Output file:** `training_curves.png`

**What it shows.** Four-panel overview of the grokking phenomenon:
1. **Loss** (log scale): train and test cross-entropy loss over epochs
2. **Accuracy**: train and test classification accuracy over epochs
3. **Weight norm**: L2 norm of all parameters over epochs
4. **Gini coefficient**: Fourier frequency sparsity over epochs

**How to read it.** The four panels together tell the complete grokking story. Look for the characteristic pattern: fast train convergence (panel 1-2), long plateau, then sudden test improvement. The weight norm (panel 3) and Gini (panel 4) provide the mechanistic explanation.

**What to watch for:**
- Train loss should drop to near zero by epoch ~300; test loss should remain high until epoch ~3000–5000
- Train accuracy should reach ~100% early; test accuracy should show a sharp sigmoid-like jump
- Weight norm should peak during memorization then decrease as the trig algorithm takes over
- Gini should rise monotonically from ~0.3 to ~0.9, even during the test-accuracy plateau

**Something looks wrong if:**
- Train loss doesn't converge → learning rate too low or model too small
- Test accuracy never improves → weight decay too low (try wd=1.0) or not enough epochs
- Weight norm keeps growing → weight decay not active or too low
- Gini stays flat → Fourier analysis may have a bug, or model is stuck in memorization

---

### `plot_progress_measures`

**Source:** `plot_progress_measures(history: dict, fourier_snapshots: dict) -> plt.Figure`
**Output file:** `progress_measures.png`

**What it shows.** The four Nanda et al. progress measures tracked during training:
1. **Loss curves**: train/test loss (same as panel 1 of grokking curves, for context)
2. **Gini coefficient**: frequency sparsity over epochs
3. **Weight norm**: parameter L2 norm over epochs
4. **Key frequency energy**: total energy in top-5 frequencies at each Fourier snapshot

**How to read it.** This plot reveals what's happening "under the hood" during the test-accuracy plateau. While accuracy metrics show nothing, these measures show steady progress.

**What to watch for:**
- Key frequency energy should grow *monotonically* even while test accuracy is stuck
- Gini and frequency energy should track each other (both measure Fourier sparsification)
- The epoch where key frequency energy starts rising rapidly should precede the test accuracy jump

**Something looks wrong if:**
- Key frequency energy is flat → model isn't learning Fourier structure (check weight decay)
- Gini and frequency energy diverge → possible bug in frequency identification

---

### `plot_phase_boundaries`

**Source:** `plot_phase_boundaries(history: dict) -> plt.Figure`
**Output file:** `phase_boundaries.png`

**What it shows.** Train and test accuracy curves with vertical lines annotating phase boundaries: memorization onset, memorization complete, grokking onset, grokking complete.

**How to read it.** The shaded regions between phase boundaries highlight the three training phases (memorization, circuit formation, generalization). The gap between "memorization complete" and "grokking onset" is the circuit formation phase where progress measures move but accuracy does not.

**What to watch for:**
- The circuit formation phase should be the longest phase (typically 10-30x the memorization phase)
- Phase boundaries should be well-separated; if memorization and grokking overlap, weight decay may be too high

---

## Fourier Analysis

Five plots examining the frequency-domain structure of the model's computations. Module: `src/viz/fourier_plots.py`.

### `plot_frequency_spectrum`

**Source:** `plot_frequency_spectrum(frequency_norms: np.ndarray, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `frequency_spectrum.png`

**What it shows.** Bar chart of marginal frequency energy for all 113 frequencies. The top-5 key frequencies are highlighted in a contrasting color.

**How to read it.** Each bar represents how much energy the model's logit table has at that frequency. After grokking, 3-5 bars should tower above the rest.

**What to watch for:**
- Key frequency bars should be >10x the noise floor (the median bar height)
- Typically 3-5 clear peaks, not a gradual roll-off
- The DC component (k=0) may or may not be large — it represents the mean logit, which is fine either way

**Something looks wrong if:**
- No clear peaks → model hasn't learned Fourier structure (check if grokking occurred)
- Many peaks of similar height → model may be in a transitional state; train longer
- Peaks don't match expected key frequencies → not necessarily wrong (different seeds can find different frequency sets) but verify with other plots

---

### `plot_fourier_heatmap`

**Source:** `plot_fourier_heatmap(component_norms: np.ndarray, p: int) -> plt.Figure`
**Output file:** `fourier_heatmap.png`

**What it shows.** 2D heatmap of Fourier component energy |L_hat[k1, k2]|^2 summed over output classes. Log scale. Axes are frequency indices k1 and k2.

**How to read it.** This is the full 2D Fourier decomposition. Bright spots indicate frequency pairs with high energy.

**What to watch for:**
- Bright spots should appear at (k1, k2) where *both* k1 and k2 are key frequencies
- A cross pattern at the DC row (k1=0) and DC column (k2=0) is expected — these are the DC-crossed-with-key-frequency terms
- The diagonal may show some structure (k1 = k2 terms)
- Off-key-frequency regions should be dark (near zero energy)

**Something looks wrong if:**
- Energy is diffuse (no bright spots) → model hasn't formed Fourier structure
- Bright spots at unexpected frequency pairs → the model found a different algorithm or is still transitional

---

### `plot_fourier_evolution`

**Source:** `plot_fourier_evolution(fourier_snapshots: dict, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `fourier_evolution.png`

**What it shows.** Line plot of key frequency energies over training epochs. Each key frequency gets its own colored line. A gray dashed line shows the noise floor (median non-key frequency energy).

**How to read it.** This is the dynamic view of Fourier sparsification. You can see exactly when each key frequency emerges and strengthens.

**What to watch for:**
- Key frequencies should rise roughly in tandem (they're part of the same algorithm)
- The noise floor (gray dashed) should decrease over training as weight decay suppresses non-key components
- Key frequency energy should cross above the noise floor well before the test accuracy jump
- The separation between key frequency energy and noise floor should grow over time

**Something looks wrong if:**
- Key frequencies rise at very different rates → possible, but unusual
- Noise floor doesn't decrease → weight decay may not be strong enough
- Key frequencies peak then decrease → model may be unstable; check learning rate

---

### `plot_embedding_fourier`

**Source:** `plot_embedding_fourier(embed_fourier: dict, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `embedding_fourier.png`

**What it shows.** Bar chart of the embedding matrix W_E's frequency content, computed via 1D DFT along the token dimension.

**How to read it.** Similar to the logit-table frequency spectrum, but for the embedding layer specifically. Shows which frequencies are encoded in how the model represents input tokens.

**What to watch for:**
- The same key frequencies that dominate the logit table should appear here
- If W_E has different peaks than the logit table, the embedding may not be the sole source of Fourier structure (attention or MLP may also contribute)

**Something looks wrong if:**
- W_E spectrum is flat → the embedding hasn't specialized; structure may be in later layers
- Peaks don't match logit-table key frequencies → investigate further (not necessarily a bug)

---

### `plot_fourier_spectrum_strip`

**Source:** `plot_fourier_spectrum_strip(fourier_snapshots: dict, key_freqs: np.ndarray, p: int, n_panels: int = 6) -> plt.Figure`
**Output file:** `fourier_spectrum_strip.png`

**What it shows.** Side-by-side bar charts of the frequency spectrum at multiple training epochs (default: 6 snapshots evenly spaced). Each panel is a snapshot of `frequency_norms` at that epoch.

**How to read it.** Reading left to right shows the temporal evolution from diffuse (uniform) to sparse (peaked). This is the filmstrip version of `plot_fourier_evolution`.

**What to watch for:**
- Early panels should look roughly uniform (all bars similar height)
- Middle panels should show emerging peaks
- Late panels should show 3-5 clear peaks with everything else near zero
- The transition from uniform to sparse should be gradual, not abrupt

---

## Attention Patterns

Two plots showing how the transformer routes information. Module: `src/viz/attention_plots.py`.

### `plot_attention_patterns`

**Source:** `plot_attention_patterns(attn_weights: torch.Tensor, token_labels: list[str] | None = None) -> plt.Figure`
**Output file:** `attention_patterns.png`

**What it shows.** Heatmap of average attention weights across the batch. Since `nn.MultiheadAttention` returns head-averaged weights, this is a 3x3 matrix showing how much each position attends to each other position.

**How to read it.** Rows are query positions (where information flows *to*), columns are key positions (where information flows *from*). Entry [i, j] = how much position i attends to position j.

**What to watch for:**
- Strong attention from position 2 (`=`) to positions 0 (`a`) and 1 (`b`). This is the information routing that feeds both operands to the prediction position
- The `=` position needs information from both `a` and `b` to compute the sum — attention should reflect this
- Attention from positions 0 and 1 to themselves or each other is less important

**Something looks wrong if:**
- Position 2 doesn't attend to positions 0 and 1 → the model may be using a non-standard information routing strategy
- All attention weights are roughly uniform (~0.33 each) → attention hasn't specialized

---

### `plot_attention_by_input`

**Source:** `plot_attention_by_input(model, inputs, a_vals, b_vals, p, n_examples=9) -> plt.Figure`
**Output file:** (generated on demand, not in standard pipeline)

**What it shows.** Grid of individual attention patterns for specific (a, b) input examples. Each sub-heatmap is the 3x3 attention matrix for one input.

**How to read it.** Allows checking whether the attention pattern is *input-independent* (same for all inputs) or varies. For the trigonometric algorithm, attention should be largely uniform across inputs since the same routing is needed regardless of specific a and b values.

**What to watch for:**
- Patterns should be very similar across all examples — the algorithm is the same for all inputs
- If patterns vary dramatically by input, the model may be using input-conditional routing

---

## Embedding Geometry

Two plots showing the geometric structure of learned representations. Module: `src/viz/embedding_geometry.py`.

### `plot_embedding_circles`

**Source:** `plot_embedding_circles(W_E: np.ndarray, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `embedding_circles.png`

**What it shows.** Polar star plots — one per key frequency. For each frequency k, projects the p=113 token embeddings onto their Fourier-k components and plots them in the complex plane.

**How to read it.** If the embedding encodes frequency k cleanly, the 113 tokens should form a regular polygon (circle) in the Fourier-k projection. The angular position of token `a` should be `2*pi*k*a/p`.

**What to watch for:**
- Clean circles with evenly-spaced points at key frequencies — this means the embedding has learned the trigonometric representation
- Points should appear in order (token 0, 1, 2, ... around the circle)
- The radius of the circle indicates how much energy the embedding devotes to that frequency

**Something looks wrong if:**
- Points form a blob instead of a circle → the embedding hasn't learned this frequency cleanly
- Circle is lopsided or has gaps → partial learning, may need more training
- Circles at non-key frequencies appear clean → unexpected; investigate

---

### `plot_neuron_frequency_clusters`

**Source:** `plot_neuron_frequency_clusters(neuron_class: dict, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `neuron_clusters.png`

**What it shows.** Two-panel figure:
1. **Scatter plot**: each of 512 neurons plotted by dominant frequency (x-axis) vs R^2 (y-axis). Neurons above the R^2 threshold (default 0.5) are colored by their dominant frequency.
2. **Bar chart**: number of neurons per cluster (frequency group).

**How to read it.** Neurons that have learned to compute a single frequency will appear as high-R^2 points clustered at key frequency values. The bar chart shows how many neurons are devoted to each frequency.

**What to watch for:**
- Many neurons with R^2 > 0.5, clustered at key frequencies
- Roughly equal cluster sizes across key frequencies (each frequency needs similar computational resources)
- A cloud of low-R^2 neurons near the bottom — these are "unspecialized" neurons (dead or noisy)

**Something looks wrong if:**
- No high-R^2 neurons → model hasn't developed frequency-selective neurons (check if grokking occurred)
- Neurons cluster at non-key frequencies → model may have found different key frequencies than expected
- Very uneven cluster sizes → one frequency may be harder to compute

---

## Weight Matrices

Two plots showing the raw weight structure. Module: `src/viz/weight_heatmaps.py`.

### `plot_weight_heatmap`

**Source:** `plot_weight_heatmap(model, p: int) -> plt.Figure`
**Output file:** `weight_heatmaps.png`

**What it shows.** Side-by-side heatmaps of the token embedding matrix W_E (shape 114 x 128) and unembedding matrix W_U (shape 128 x 113).

**How to read it.** After grokking, both matrices should show periodic stripe patterns corresponding to the key frequencies. W_E encodes tokens as Fourier components; W_U decodes them back to class logits.

**What to watch for:**
- Periodic vertical or horizontal stripes in both W_E and W_U
- Stripe frequency should match key frequencies (count stripes to verify)
- The equals-token row in W_E (row 113) may look different from integer-token rows

**Something looks wrong if:**
- Weights look random (no stripes) → model is still in memorization phase
- Only one matrix shows structure → partial algorithm formation

---

### `plot_weight_evolution`

**Source:** `plot_weight_evolution(checkpoint_paths: list, config: dict, p: int, device: torch.device) -> plt.Figure`
**Output file:** `weight_evolution.png`

**What it shows.** W_E and W_U heatmaps at multiple training checkpoints (typically 3-4 evenly spaced). Shows the temporal transition from random initialization to structured weights.

**How to read it.** Reading left to right shows weights evolving from random → structured. The transition from noise to stripes corresponds to the circuit formation phase.

**What to watch for:**
- Early checkpoints should look random (no visible pattern)
- Late checkpoints should show clear periodic structure
- The transition should be gradual, not sudden — weight structure develops before test accuracy jumps

---

## Neuron Analysis

Three plots examining individual MLP neuron behavior. Module: `src/viz/neuron_plots.py`.

### `plot_neuron_activation_grids`

**Source:** `plot_neuron_activation_grids(neuron_activations_grid: np.ndarray, neuron_class: dict, key_freqs: np.ndarray, p: int, neurons_per_freq: int = 3) -> plt.Figure`
**Output file:** `neuron_activation_grids.png`

**What it shows.** Grid of p x p heatmaps showing individual neuron activations. Neurons are grouped by their dominant frequency, and the top `neurons_per_freq` (default 3) neurons per frequency are shown. Each heatmap shows `activation[a, b]` for one neuron across all input pairs.

**How to read it.** A neuron tuned to frequency k should produce a clean periodic pattern with k stripes in both the a and b directions. Higher R^2 neurons produce cleaner patterns.

**What to watch for:**
- Clean periodic bands for high-R^2 neurons — the number of bands should correspond to the frequency
- Neurons in the same frequency group should show similar patterns (up to phase shifts)
- Post-ReLU means all activations are non-negative — patterns show bright bands on a dark background

**Something looks wrong if:**
- Patterns look noisy even for high-R^2 neurons → possible issue with activation extraction
- Band counts don't match labeled frequencies → frequency classification may be off
- All neurons look the same → neurons haven't differentiated

---

### `plot_neuron_logit_map`

**Source:** `plot_neuron_logit_map(neuron_logit_map: np.ndarray, neuron_class: dict, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `neuron_logit_map.png`

**What it shows.** Heatmap of the neuron-logit map W_L (shape 512 x 113), with neurons sorted by dominant frequency. Shows how each neuron contributes to each output class logit.

**How to read it.** When neurons are sorted by frequency, the heatmap should show a block structure: neurons in the same frequency group produce sinusoidal patterns at the same frequency across the output classes. Each block corresponds to one key frequency.

**What to watch for:**
- Clear block structure with visually distinct blocks for each key frequency
- Within each block, rows should show sinusoidal oscillation at the block's frequency
- Clean sinusoids indicate the neuron-to-output mapping is well-organized

**Something looks wrong if:**
- No block structure → neurons aren't frequency-organized
- Blocks are noisy or fragmented → partial algorithm formation
- Some blocks are much larger/smaller than others → imbalanced frequency representation

---

### `plot_neuron_frequency_spectrum_heatmap`

**Source:** `plot_neuron_frequency_spectrum_heatmap(neuron_spectrum: np.ndarray, neuron_class: dict, key_freqs: np.ndarray, p: int) -> plt.Figure`
**Output file:** `neuron_freq_spectrum_heatmap.png`

**What it shows.** Heatmap of per-neuron frequency energy (shape 512 x 113), with neurons sorted by dominant frequency. Log scale. Row n, column k = marginal energy of frequency k in neuron n's activation grid.

**How to read it.** If neurons are cleanly frequency-selective, each row should have energy concentrated in exactly one column (its dominant frequency). The overall pattern should be block-diagonal-like.

**What to watch for:**
- Each neuron's energy should be concentrated in exactly one frequency column
- When sorted by frequency, this should produce a clear stepped or block-diagonal pattern
- Key frequency columns should contain many bright entries; other columns should be dark

**Something looks wrong if:**
- Energy is spread across multiple columns per neuron → neurons aren't frequency-selective
- No block-diagonal structure → neuron sorting may be wrong, or neurons aren't specialized

---

## Logit Analysis

Three plots examining the model's output logit table. Module: `src/viz/logit_plots.py`.

### `plot_logit_heatmap_comparison`

**Source:** `plot_logit_heatmap_comparison(logit_table: np.ndarray, restricted_logits: np.ndarray, p: int) -> plt.Figure`
**Output file:** `logit_heatmap_comparison.png`

**What it shows.** Three-panel comparison of correct-class logits (L[a, b, (a+b)%p]):
1. **Full logits**: the complete logit for the correct class at each (a, b)
2. **Restricted logits**: using only DC + key frequency Fourier components
3. **Difference**: full - restricted

**How to read it.** If the trigonometric algorithm fully explains the model's behavior, the restricted panel should look nearly identical to the full panel, and the difference panel should be near zero everywhere.

**What to watch for:**
- Restricted panel should closely match the full panel (similar patterns and magnitudes)
- Difference panel should be near zero (uniform dark/light, not structured)
- If the difference panel shows structure, there are non-key-frequency contributions that matter

**Something looks wrong if:**
- Large difference between full and restricted → key frequencies don't capture the full algorithm (try n_top > 5)
- Restricted panel looks nothing like full → Fourier decomposition may have a bug
- Difference panel shows periodic structure → there are additional important frequencies not in the key set

---

### `plot_correct_logit_surface`

**Source:** `plot_correct_logit_surface(logit_table: np.ndarray, p: int) -> plt.Figure`
**Output file:** `correct_logit_surface.png`

**What it shows.** 3D surface plot of correct-class logits over the (a, b) plane. Height at point (a, b) = logit assigned to the correct class (a+b) mod p.

**How to read it.** A model that has learned the trig algorithm produces a smooth, periodic surface. The periodicity comes from the Fourier components — you should see wave-like ridges and valleys.

**What to watch for:**
- Smooth periodic surface → the model uses a clean Fourier algorithm
- Regularity across the entire (a, b) plane (not just in training regions)
- Height should be consistently positive (correct class gets high logit)

**Something looks wrong if:**
- Surface is noisy/jagged → model is partially memorizing, not fully using the trig algorithm
- Surface is smooth in some regions but noisy in others → the algorithm works for some inputs but not others

---

### `plot_per_sample_loss_heatmap`

**Source:** `plot_per_sample_loss_heatmap(logit_table: np.ndarray, p: int, train_mask: np.ndarray | None = None) -> plt.Figure`
**Output file:** `per_sample_loss_heatmap.png`

**What it shows.** Heatmap of cross-entropy loss for each (a, b) pair. If a train mask is provided, the train/test boundary is overlaid (e.g., as a contour or color overlay) to distinguish training from test samples.

**How to read it.** After grokking, loss should be uniformly low across both training and test regions. Before grokking, training samples should have low loss while test samples have high loss.

**What to watch for:**
- Uniform low loss across the entire grid after grokking — the algorithm works everywhere
- No systematic difference between train and test regions after grokking
- If loss is lower in training regions than test regions, the model still partially relies on memorization

**Something looks wrong if:**
- High loss spots scattered across the grid → some inputs are poorly predicted
- Clear train/test boundary in loss values after grokking → generalization is incomplete
- Loss is high everywhere → model hasn't converged

---

## Trajectories

Two static plots and one animation showing how representations and parameters evolve. Module: `src/viz/trajectory_plots.py` and `src/viz/animation.py`.

### `plot_embedding_pca_evolution`

**Source:** `plot_embedding_pca_evolution(embedding_snapshots: list[np.ndarray], snapshot_epochs: list[int], p: int, n_panels: int = 6) -> plt.Figure`
**Output file:** `embedding_pca_evolution.png`

**What it shows.** Grid of PCA scatter plots showing the first two principal components of the token embedding matrix W_E at multiple training epochs. Each point is one of the 113 integer tokens, colored by token value (rainbow colormap).

**How to read it.** Reading left to right shows the embedding structure evolving from random to organized:
- **Early**: random cloud, no structure
- **Middle**: points begin to spread along a ring or curve
- **Late**: clean circle or ring with rainbow coloring (tokens ordered by value around the circle)

**What to watch for:**
- Clear transition from cloud → ring → ordered circle
- Rainbow coloring in the final panels (adjacent tokens should be adjacent on the circle)
- The circular structure emerges because the dominant Fourier components are periodic in token index

**Something looks wrong if:**
- No circular structure in late panels → embeddings haven't learned Fourier representation
- Circle but no rainbow ordering → tokens are on a circle but not ordered by value (possible with different frequency selections)

---

### `plot_weight_trajectory_pca`

**Source:** `plot_weight_trajectory_pca(param_snapshots: list[np.ndarray], snapshot_epochs: list[int], history: dict | None = None) -> plt.Figure`
**Output file:** `weight_trajectory_pca.png`

**What it shows.** The training trajectory in parameter space, projected onto its first two principal components. Each point is one checkpoint, connected by lines. Points are colored by test accuracy (red = low, green = high) if history is provided.

**How to read it.** Shows how the model moves through weight space during training. The trajectory often shows distinct phases: an initial rapid movement (memorization), a direction change (circuit formation begins), and convergence (grokking complete).

**What to watch for:**
- Sharp direction changes at phase transitions — the model "turns a corner" when switching from memorization to the trig algorithm
- Color shift from red → green should coincide with the turn
- The trajectory should eventually stabilize (convergence)

**Something looks wrong if:**
- Trajectory is a straight line → no phase transition (check if grokking happened)
- Trajectory spirals or oscillates → training may be unstable (reduce learning rate)
- No color change → test accuracy never improved

---

### `create_grokking_animation`

**Source:** `create_grokking_animation(history, fourier_snapshots, embedding_snapshots, snapshot_epochs, p, key_freqs, fps=10, output_path=None) -> FuncAnimation`
**Output file:** `grokking_animation.gif`

**What it shows.** Synchronized 4-panel animation showing the grokking process over time:
1. **Loss curves**: train/test loss with a moving epoch marker
2. **Frequency spectrum**: bar chart evolving from uniform to sparse
3. **Embedding PCA**: scatter plot evolving from cloud to circle
4. **Gini coefficient**: time series with moving epoch marker

**How to read it.** All four panels advance together in time. Watch for the coordinated transition: as Gini rises and the spectrum sharpens, the embedding forms a circle, and loss finally drops.

**What to watch for:**
- All four indicators should change in concert during the circuit formation phase
- The spectrum should visibly sharpen before the loss curve shows test improvement
- The embedding circle should form gradually, not all at once

---

## Summary of All Figures

| # | Function | Output file | Module |
|---|----------|-------------|--------|
| 1 | `plot_grokking_curves` | `training_curves.png` | `training_curves.py` |
| 2 | `plot_progress_measures` | `progress_measures.png` | `training_curves.py` |
| 3 | `plot_frequency_spectrum` | `frequency_spectrum.png` | `fourier_plots.py` |
| 4 | `plot_fourier_heatmap` | `fourier_heatmap.png` | `fourier_plots.py` |
| 5 | `plot_fourier_evolution` | `fourier_evolution.png` | `fourier_plots.py` |
| 6 | `plot_embedding_fourier` | `embedding_fourier.png` | `fourier_plots.py` |
| 7 | `plot_attention_patterns` | `attention_patterns.png` | `attention_plots.py` |
| 8 | `plot_embedding_circles` | `embedding_circles.png` | `embedding_geometry.py` |
| 9 | `plot_neuron_frequency_clusters` | `neuron_clusters.png` | `embedding_geometry.py` |
| 10 | `plot_weight_heatmap` | `weight_heatmaps.png` | `weight_heatmaps.py` |
| 11 | `plot_neuron_activation_grids` | `neuron_activation_grids.png` | `neuron_plots.py` |
| 12 | `plot_neuron_logit_map` | `neuron_logit_map.png` | `neuron_plots.py` |
| 13 | `plot_neuron_frequency_spectrum_heatmap` | `neuron_freq_spectrum_heatmap.png` | `neuron_plots.py` |
| 14 | `plot_logit_heatmap_comparison` | `logit_heatmap_comparison.png` | `logit_plots.py` |
| 15 | `plot_correct_logit_surface` | `correct_logit_surface.png` | `logit_plots.py` |
| 16 | `plot_per_sample_loss_heatmap` | `per_sample_loss_heatmap.png` | `logit_plots.py` |
| 17 | `plot_embedding_pca_evolution` | `embedding_pca_evolution.png` | `trajectory_plots.py` |
| 18 | `plot_weight_trajectory_pca` | `weight_trajectory_pca.png` | `trajectory_plots.py` |
| 19 | `plot_fourier_spectrum_strip` | `fourier_spectrum_strip.png` | `fourier_plots.py` |
| 20 | `create_grokking_animation` | `grokking_animation.gif` | `animation.py` |

Additional functions available but not in the standard pipeline:
- `plot_phase_boundaries` (`training_curves.py`)
- `plot_attention_by_input` (`attention_plots.py`)
- `plot_weight_evolution` (`weight_heatmaps.py`)
