# Grokking Experiments — Documentation

Replication of **"Progress measures for grokking via mechanistic interpretability"** (Nanda, Chan, Lieberum, Smith & Steinhardt, ICLR 2023). Trains a 1-layer transformer on modular addition (a + b mod 113), observes delayed generalization (grokking), and uses Fourier analysis to track the emergence of a trigonometric algorithm.

## Contents

| Document | Description |
|----------|-------------|
| [Getting Started](./quickstart.md) | Installation, training, figure generation, dashboard |
| [The Grokking Phenomenon](./grokking-overview.md) | What grokking is, the task, model, trig algorithm, training phases |
| [Analysis Pipeline](./analysis-pipeline.md) | Fourier decomposition, progress measures, neuron analysis — technical details |
| [Visualization Guide](./visualization-guide.md) | Complete reference for all 23 visualization functions with interpretation guidance |
| [References](./references.md) | Annotated bibliography of key papers |

## Quick Reference

```bash
# Activate environment
conda activate kripke

# Train (smoke test — 100 epochs)
python scripts/train_single.py --config configs/default.yaml --max-epochs 100

# Train (full — 40K epochs, ~20 min on GPU)
python scripts/train_single.py --config configs/nanda_replication.yaml

# Post-hoc analysis
python scripts/analyze_run.py --run-dir results/<run_id>

# Generate all figures (20 PNGs + 1 GIF)
python scripts/generate_figures.py --run-dir results/<run_id>

# Interactive dashboard
streamlit run dashboard/app.py
```

## Key Paper

Nanda, N., Chan, L., Lieberum, T., Smith, J., & Steinhardt, J. (2023). *Progress Measures for Grokking via Mechanistic Interpretability.* ICLR 2023. [arXiv:2301.05217](https://arxiv.org/abs/2301.05217)

## Project Structure

```
configs/              YAML hyperparameter configs
src/data/             Modular arithmetic data generation
src/models/           GrokkingTransformer + activation hooks
src/training/         Full-batch trainer + checkpointing
src/analysis/         Fourier analysis, progress measures, neuron analysis
src/viz/              10 visualization modules (23 functions)
scripts/              CLI entry points (train, analyze, figures)
dashboard/            Streamlit interactive explorer
tests/                Unit tests
docs/                 This documentation
```
