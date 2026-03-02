# Getting Started

Practical guide to running the grokking experiments project — from installation through figure generation and the interactive dashboard.

## Prerequisites

- **Conda** with the `kripke` environment (Python 3.11)
- **PyTorch** with CUDA support (cu121)
- **GPU recommended** — full training takes ~20 minutes on a modern GPU, much longer on CPU

## Installation

```bash
conda activate kripke
cd grokking_experiments
pip install -e .
```

Verify the installation:

```bash
python -m pytest tests/ -v
```

All tests should pass. If Fourier or model tests fail, check that NumPy and PyTorch are correctly installed.

## Quick Smoke Test

Run a 100-epoch training to verify everything works:

```bash
python scripts/train_single.py --config configs/default.yaml --max-epochs 100
```

This should complete in under a minute and produce a run directory under `results/`. You won't see grokking in 100 epochs (that takes thousands), but it confirms the training loop, checkpointing, and Fourier logging all work.

Expected output: a `results/p113_d128_h4_mlp512_L1_wd1.0_s42/` directory containing:
- `config.json` — merged configuration
- `metrics.json` — training history (loss, accuracy, weight norm, Gini per epoch)
- `model.pt` — final model weights
- `fourier_snapshots.npz` — Fourier analysis at logged intervals
- `checkpoints/` — periodic model checkpoints

## Full Training Run

The default configuration trains for 40,000 epochs:

```bash
python scripts/train_single.py --config configs/default.yaml
```

For the exact Nanda et al. replication (with finer logging intervals):

```bash
python scripts/train_single.py --config configs/nanda_replication.yaml
```

**Expected timeline on GPU:** ~20 minutes for 40K epochs.

**What to expect:**
- Train accuracy reaches ~100% by epoch ~300
- Test accuracy stays near random (~1%) until epoch ~3,000–5,000
- Test accuracy then jumps to ~95%+ over a few hundred epochs
- Gini coefficient rises from ~0.3 to ~0.9

> **Note:** The exact epoch numbers vary by seed and hardware. The qualitative pattern (memorize → plateau → generalize) is robust.

## Post-Hoc Analysis

After training, run the analysis script on a completed run:

```bash
python scripts/analyze_run.py --run-dir results/<run_id>
```

This computes:
- Restricted and excluded loss decomposition
- Final Gini coefficient and key frequencies
- Embedding Fourier analysis
- Neuron frequency classification and clustering

Results are saved to `analysis.json` in the run directory.

## Generating Figures

Generate all 20 publication-quality figures plus an animation:

```bash
python scripts/generate_figures.py --run-dir results/<run_id>
```

Figures are saved to `results/<run_id>/figures/` by default. To save elsewhere:

```bash
python scripts/generate_figures.py --run-dir results/<run_id> --figures-dir ./my_figures
```

**Expected output:** 20 PNG files + 1 GIF:

| File | Content |
|------|---------|
| `training_curves.png` | 4-panel loss, accuracy, weight norm, Gini |
| `progress_measures.png` | Nanda's 4 progress measures |
| `frequency_spectrum.png` | Marginal frequency energies |
| `fourier_heatmap.png` | 2D Fourier component energy |
| `fourier_evolution.png` | Key frequency energy over time |
| `embedding_fourier.png` | Embedding matrix spectrum |
| `attention_patterns.png` | Average attention weights |
| `embedding_circles.png` | Token embedding polar projections |
| `neuron_clusters.png` | Neuron frequency scatter + bar |
| `weight_heatmaps.png` | W_E and W_U matrices |
| `neuron_activation_grids.png` | Per-neuron (a,b) activation patterns |
| `neuron_logit_map.png` | Neuron-to-output-class weights |
| `neuron_freq_spectrum_heatmap.png` | Per-neuron frequency selectivity |
| `logit_heatmap_comparison.png` | Full vs restricted logits |
| `correct_logit_surface.png` | 3D surface of correct-class logits |
| `per_sample_loss_heatmap.png` | Per-sample CE loss |
| `embedding_pca_evolution.png` | Embedding PCA over training |
| `weight_trajectory_pca.png` | Parameter space trajectory |
| `fourier_spectrum_strip.png` | Multi-epoch spectrum snapshots |
| `grokking_animation.gif` | Synchronized 4-panel animation |

See [Visualization Guide](./visualization-guide.md) for detailed descriptions of every figure.

## Interactive Dashboard

Launch the Streamlit dashboard for interactive exploration:

```bash
streamlit run dashboard/app.py
```

The dashboard has 8 tabs:

1. **Training Curves** — loss, accuracy, weight norm with interactive controls
2. **Fourier Analysis** — frequency spectrum with epoch slider
3. **Progress Measures** — Gini and key frequency evolution
4. **Mechanistic Interp** — neuron clusters, embedding spectra
5. **Weight Matrices** — W_E and W_U heatmaps at various epochs
6. **Neuron Analysis** — activation grids, logit map, frequency spectrum
7. **Logit Tables** — full vs restricted comparison, 3D surface
8. **Trajectories** — embedding PCA, weight trajectory, animation

Use the sidebar to select a run from the `results/` directory.

## Typical Workflow

1. **Train** a model with the default or Nanda replication config
2. **Check** the training log — confirm train acc hit 100% early and test acc eventually jumped
3. **Run analysis** to compute Fourier measures and neuron classifications
4. **Generate figures** for the complete mechanistic interpretability story
5. **Open the dashboard** for interactive exploration and parameter tuning
6. **Read the figures** using the [Visualization Guide](./visualization-guide.md) to understand what each plot reveals about the learned algorithm

## Configuration

All configurations are YAML files in `configs/`. The merging order is:

1. `configs/default.yaml` (base)
2. Override config (e.g., `configs/nanda_replication.yaml`)
3. CLI arguments (e.g., `--max-epochs 100`)

Key parameters to experiment with:

| Parameter | Default | Effect of changing |
|-----------|---------|-------------------|
| `weight_decay` | 1.0 | Lower → slower or no grokking; higher → faster but may prevent memorization |
| `max_epochs` | 40000 | May need more for low weight decay |
| `p` | 113 | Different primes work; larger p = harder task |
| `train_fraction` | 0.3 | More data → faster grokking; less data → slower |
| `d_model` | 128 | Smaller → fewer Fourier components representable |
| `seed` | 42 | Different seeds find different key frequency sets |
