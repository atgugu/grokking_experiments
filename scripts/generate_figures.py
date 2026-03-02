#!/usr/bin/env python
"""Generate all publication figures from a completed run."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.modular_arithmetic import ModularArithmeticEnvironment
from src.models.transformer import GrokkingTransformer
from src.models.hooks import ActivationCache
from src.training.checkpointing import load_run_result, load_fourier_snapshots
from src.analysis.fourier import (
    compute_fourier_component_norms,
    identify_key_frequencies,
    compute_restricted_logits,
    fourier_embed_analysis,
)
from src.analysis.neuron_analysis import (
    classify_neuron_frequencies,
    compute_neuron_logit_map,
    compute_neuron_frequency_spectrum,
)
from src.viz.training_curves import plot_grokking_curves, plot_progress_measures
from src.viz.fourier_plots import (
    plot_frequency_spectrum,
    plot_fourier_heatmap,
    plot_fourier_evolution,
    plot_embedding_fourier,
    plot_fourier_spectrum_strip,
)
from src.viz.attention_plots import plot_attention_patterns
from src.viz.embedding_geometry import plot_embedding_circles, plot_neuron_frequency_clusters
from src.viz.weight_heatmaps import plot_weight_heatmap, plot_weight_evolution
from src.viz.neuron_plots import (
    plot_neuron_activation_grids,
    plot_neuron_logit_map,
    plot_neuron_frequency_spectrum_heatmap,
)
from src.viz.logit_plots import (
    plot_logit_heatmap_comparison,
    plot_correct_logit_surface,
    plot_per_sample_loss_heatmap,
)
from src.viz.trajectory_plots import (
    plot_embedding_pca_evolution,
    plot_weight_trajectory_pca,
)
from src.viz.animation import (
    create_grokking_animation,
    create_fourier_waterfall_animation,
    create_embedding_circle_animation,
    create_loss_landscape_animation,
    create_neuron_grid_animation,
)
from src.utils import get_device, setup_logging


def _load_model(config, device):
    """Create and return a GrokkingTransformer from config."""
    return GrokkingTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_mlp=config["d_mlp"],
        n_layers=config["n_layers"],
        activation=config.get("activation", "relu"),
        use_layernorm=config.get("use_layernorm", False),
        tie_embeddings=config.get("tie_embeddings", False),
        mlp_bias=config.get("mlp_bias", True),
    )


def _load_model_from_checkpoint(ckpt_path, config, device):
    """Load a full GrokkingTransformer from a checkpoint file.

    Args:
        ckpt_path: Path to checkpoint .pt file.
        config: Run config dict.
        device: Torch device.

    Returns:
        Model in eval mode on the specified device.
    """
    model = _load_model(config, device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = (
        ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt
        else ckpt["model_state_dict"]
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _load_checkpoints(ckpt_dir, config, device):
    """Load checkpoint paths sorted by epoch, and collect embeddings + param vectors.

    Returns:
        ckpt_files: Sorted list of checkpoint Path objects.
        embedding_snapshots: List of (p+1, d_model) numpy arrays.
        param_snapshots: List of 1D numpy param vectors.
        snapshot_epochs: List of epoch ints.
    """
    if not ckpt_dir.exists():
        return [], [], [], []

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"),
                        key=lambda f: int(f.stem.split("_")[1]))
    if not ckpt_files:
        return [], [], [], []

    embedding_snapshots = []
    param_snapshots = []
    snapshot_epochs = []

    for ckpt_path in ckpt_files:
        epoch = int(ckpt_path.stem.split("_")[1])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt else ckpt["model_state_dict"]

        # Extract embedding
        W_E = state_dict["W_E.weight"].cpu().numpy()
        embedding_snapshots.append(W_E)

        # Flatten all params into one vector
        params = np.concatenate([v.cpu().numpy().ravel() for v in state_dict.values()])
        param_snapshots.append(params)

        snapshot_epochs.append(epoch)

    return ckpt_files, embedding_snapshots, param_snapshots, snapshot_epochs


def _compute_loss_snapshots(ckpt_files, config, device, p):
    """Compute per-sample CE loss (p, p) for each checkpoint.

    Returns:
        List of (p, p) numpy arrays with per-sample cross-entropy loss.
    """
    import torch.nn.functional as F

    loss_snapshots = []
    for ckpt_path in ckpt_files:
        model = _load_model_from_checkpoint(ckpt_path, config, device)
        logit_table = model.get_logit_table(device).cpu()  # (p, p, p)
        # Target: (a + b) mod p
        a_grid, b_grid = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
        targets = (a_grid + b_grid) % p  # (p, p)
        # Per-sample CE loss
        logits_flat = logit_table.reshape(-1, p)  # (p*p, p)
        targets_flat = targets.reshape(-1)
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")
        loss_snapshots.append(ce.reshape(p, p).numpy())
        del model
    return loss_snapshots


def _compute_neuron_snapshots(ckpt_files, config, device, p):
    """Compute neuron activations (p, p, d_mlp) for each checkpoint.

    Returns:
        List of (p, p, d_mlp) numpy arrays.
    """
    neuron_snapshots = []
    eq_token = config["p"]  # vocab_size - 1

    for ckpt_path in ckpt_files:
        model = _load_model_from_checkpoint(ckpt_path, config, device)
        # Build full input grid
        a_vals = torch.arange(p, device=device).repeat_interleave(p)
        b_vals = torch.arange(p, device=device).repeat(p)
        eq_vals = torch.full((p * p,), eq_token, dtype=torch.long, device=device)
        full_inputs = torch.stack([a_vals, b_vals, eq_vals], dim=1)

        with ActivationCache(model) as cache:
            with torch.no_grad():
                _ = model(full_inputs)
            if cache.neuron_activations:
                # (p*p, 3, d_mlp) -> take position 2 -> (p*p, d_mlp) -> (p, p, d_mlp)
                acts = cache.neuron_activations[0][:, 2, :].cpu().numpy()
                neuron_snapshots.append(acts.reshape(p, p, -1))
            else:
                neuron_snapshots.append(np.zeros((p, p, config["d_mlp"])))
        del model
    return neuron_snapshots


def main():
    parser = argparse.ArgumentParser(description="Generate figures from grokking run")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--figures-dir", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    run_dir = Path(args.run_dir)
    fig_dir = Path(args.figures_dir) if args.figures_dir else run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load run data
    run_result = load_run_result(run_dir)
    config = run_result["config"]
    metrics = run_result["metrics"]
    history = metrics.get("history", {})
    p = config["p"]

    logger.info(f"Generating figures for {run_dir.name}")

    # 1. Training curves
    logger.info("1/24: Training curves")
    fig = plot_grokking_curves(history)
    fig.savefig(fig_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. Progress measures (need fourier snapshots)
    logger.info("2/24: Progress measures")
    fourier_snaps = load_fourier_snapshots(run_dir)
    if fourier_snaps:
        fig = plot_progress_measures(history, fourier_snaps)
        fig.savefig(fig_dir / "progress_measures.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Load final model for remaining figures
    model_path = run_dir / "model.pt"
    if not model_path.exists():
        logger.warning("No model.pt found, skipping model-dependent figures")
        return

    model = _load_model(config, device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    env = ModularArithmeticEnvironment(
        p=config["p"],
        operation=config.get("operation", "addition"),
        train_fraction=config.get("train_fraction", 0.3),
        seed=config["seed"],
    )

    # Build train mask early (used by loss heatmap + loss landscape animation)
    train_data = env.get_train_dataset()
    train_mask = np.zeros((p, p), dtype=bool)
    for i in range(len(train_data)):
        inp, _ = train_data[i]
        a_val, b_val = int(inp[0]), int(inp[1])
        train_mask[a_val, b_val] = True

    # 3. Frequency spectrum
    logger.info("3/24: Frequency spectrum")
    logit_table = model.get_logit_table(device).cpu().numpy()
    fourier_result = compute_fourier_component_norms(logit_table, p)
    freq_norms = fourier_result["frequency_norms"]
    key_freqs = identify_key_frequencies(freq_norms, n_top=5)

    fig = plot_frequency_spectrum(freq_norms, key_freqs, p)
    fig.savefig(fig_dir / "frequency_spectrum.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. 2D Fourier heatmap
    logger.info("4/24: Fourier heatmap")
    fig = plot_fourier_heatmap(fourier_result["component_norms"], p)
    fig.savefig(fig_dir / "fourier_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 5. Fourier evolution
    logger.info("5/24: Fourier evolution")
    if fourier_snaps and "frequency_norms" in fourier_snaps:
        fig = plot_fourier_evolution(fourier_snaps, key_freqs, p)
        fig.savefig(fig_dir / "fourier_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 6. Embedding Fourier
    logger.info("6/24: Embedding Fourier")
    W_E = model.W_E.weight.detach().cpu().numpy()
    embed_fourier = fourier_embed_analysis(W_E, p)
    fig = plot_embedding_fourier(embed_fourier, key_freqs, p)
    fig.savefig(fig_dir / "embedding_fourier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 7. Attention patterns
    logger.info("7/24: Attention patterns")
    full_data = env.get_full_dataset()
    full_inputs = full_data.inputs.to(device)
    with torch.no_grad():
        _ = model(full_inputs)
    attn_patterns = model.get_attention_patterns()
    if attn_patterns:
        fig = plot_attention_patterns(attn_patterns[0])
        fig.savefig(fig_dir / "attention_patterns.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 8. Embedding circles
    logger.info("8/24: Embedding circles")
    fig = plot_embedding_circles(W_E, key_freqs, p)
    fig.savefig(fig_dir / "embedding_circles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 9. Neuron frequency clusters + extract shared neuron data
    logger.info("9/24: Neuron clusters")
    neuron_acts_grid = None
    neuron_class = None
    with ActivationCache(model) as cache:
        with torch.no_grad():
            _ = model(full_inputs)
        if cache.neuron_activations:
            neuron_acts = cache.neuron_activations[0][:, 2, :].cpu().numpy()
            neuron_acts_grid = neuron_acts.reshape(p, p, -1)
            neuron_class = classify_neuron_frequencies(neuron_acts_grid, p)
            fig = plot_neuron_frequency_clusters(neuron_class, key_freqs, p)
            fig.savefig(fig_dir / "neuron_clusters.png", dpi=150, bbox_inches="tight")
            plt.close(fig)

    # 10. Weight heatmaps
    logger.info("10/24: Weight heatmaps")
    fig = plot_weight_heatmap(model, p)
    fig.savefig(fig_dir / "weight_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Weight evolution from checkpoints
    ckpt_dir = run_dir / "checkpoints"
    ckpt_files, embedding_snapshots, param_snapshots, snapshot_epochs = \
        _load_checkpoints(ckpt_dir, config, device)

    if len(ckpt_files) >= 3:
        indices = np.linspace(0, len(ckpt_files) - 1, min(4, len(ckpt_files)), dtype=int)
        selected_ckpts = [ckpt_files[i] for i in indices]
        fig = plot_weight_evolution(selected_ckpts, config, p, device)
        fig.savefig(fig_dir / "weight_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # --- New visualizations (11-20) ---

    # 11. Neuron activation grids
    logger.info("11/24: Neuron activation grids")
    if neuron_acts_grid is not None and neuron_class is not None:
        fig = plot_neuron_activation_grids(neuron_acts_grid, neuron_class, key_freqs, p)
        fig.savefig(fig_dir / "neuron_activation_grids.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 12. Neuron-logit map
    logger.info("12/24: Neuron-logit map")
    if neuron_class is not None and model.W_U is not None:
        W_U_np = model.W_U.weight.detach().cpu().numpy()  # (p, d_model)
        W_out_np = model.blocks[0].mlp[2].weight.detach().cpu().numpy()  # (d_model, d_mlp)
        # compute_neuron_logit_map expects (d_model, p) and (d_mlp, d_model)
        neuron_logit = compute_neuron_logit_map(W_U_np.T, W_out_np.T)
        fig = plot_neuron_logit_map(neuron_logit, neuron_class, key_freqs, p)
        fig.savefig(fig_dir / "neuron_logit_map.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 13. Neuron frequency spectrum heatmap
    logger.info("13/24: Neuron frequency spectrum heatmap")
    if neuron_acts_grid is not None and neuron_class is not None:
        neuron_spectrum = compute_neuron_frequency_spectrum(neuron_acts_grid, p)
        fig = plot_neuron_frequency_spectrum_heatmap(neuron_spectrum, neuron_class, key_freqs, p)
        fig.savefig(fig_dir / "neuron_freq_spectrum_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 14. Logit heatmap comparison (full vs restricted)
    logger.info("14/24: Logit heatmap comparison")
    restricted_logits = compute_restricted_logits(logit_table, p, key_freqs)
    fig = plot_logit_heatmap_comparison(logit_table, restricted_logits, p)
    fig.savefig(fig_dir / "logit_heatmap_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 15. Correct-logit 3D surface
    logger.info("15/24: Correct-logit 3D surface")
    fig = plot_correct_logit_surface(logit_table, p)
    fig.savefig(fig_dir / "correct_logit_surface.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 16. Per-sample loss heatmap
    logger.info("16/24: Per-sample loss heatmap")
    fig = plot_per_sample_loss_heatmap(logit_table, p, train_mask)
    fig.savefig(fig_dir / "per_sample_loss_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 17. Embedding PCA evolution
    logger.info("17/24: Embedding PCA evolution")
    if embedding_snapshots:
        fig = plot_embedding_pca_evolution(embedding_snapshots, snapshot_epochs, p)
        fig.savefig(fig_dir / "embedding_pca_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 18. Weight trajectory PCA
    logger.info("18/24: Weight trajectory PCA")
    if len(param_snapshots) >= 2:
        fig = plot_weight_trajectory_pca(param_snapshots, snapshot_epochs, history)
        fig.savefig(fig_dir / "weight_trajectory_pca.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 19. Fourier spectrum strip
    logger.info("19/24: Fourier spectrum strip")
    if fourier_snaps and "frequency_norms" in fourier_snaps:
        fig = plot_fourier_spectrum_strip(fourier_snaps, key_freqs, p)
        fig.savefig(fig_dir / "fourier_spectrum_strip.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # 20. Grokking animation
    logger.info("20/24: Grokking animation")
    if (fourier_snaps and "frequency_norms" in fourier_snaps
            and embedding_snapshots):
        try:
            anim = create_grokking_animation(
                history=history,
                fourier_snapshots=fourier_snaps,
                embedding_snapshots=embedding_snapshots,
                snapshot_epochs=snapshot_epochs,
                p=p,
                key_freqs=key_freqs,
                fps=10,
                output_path=str(fig_dir / "grokking_animation.gif"),
            )
            logger.info("Animation saved as grokking_animation.gif")
        except Exception as e:
            logger.warning(f"Animation generation failed: {e}")

    # 21. Fourier waterfall animation
    logger.info("21/24: Fourier waterfall animation")
    if fourier_snaps and "frequency_norms" in fourier_snaps:
        try:
            create_fourier_waterfall_animation(
                fourier_snapshots=fourier_snaps,
                history=history,
                p=p,
                key_freqs=key_freqs,
                fps=10,
                output_path=str(fig_dir / "fourier_waterfall.gif"),
            )
            logger.info("Animation saved as fourier_waterfall.gif")
        except Exception as e:
            logger.warning(f"Fourier waterfall animation failed: {e}")

    # 22. Embedding circle formation animation
    logger.info("22/24: Embedding circle animation")
    if embedding_snapshots and len(key_freqs) > 0:
        try:
            create_embedding_circle_animation(
                embedding_snapshots=embedding_snapshots,
                snapshot_epochs=snapshot_epochs,
                p=p,
                key_freqs=key_freqs,
                fps=4,
                output_path=str(fig_dir / "embedding_circles.gif"),
            )
            logger.info("Animation saved as embedding_circles.gif")
        except Exception as e:
            logger.warning(f"Embedding circle animation failed: {e}")

    # 23. Loss landscape animation
    logger.info("23/24: Loss landscape animation")
    if ckpt_files:
        try:
            loss_snaps = _compute_loss_snapshots(ckpt_files, config, device, p)
            create_loss_landscape_animation(
                loss_snapshots=loss_snaps,
                snapshot_epochs=snapshot_epochs,
                p=p,
                train_mask=train_mask,
                history=history,
                fps=4,
                output_path=str(fig_dir / "loss_landscape.gif"),
            )
            logger.info("Animation saved as loss_landscape.gif")
        except Exception as e:
            logger.warning(f"Loss landscape animation failed: {e}")

    # 24. Neuron activation grid animation
    logger.info("24/24: Neuron grid animation")
    if ckpt_files and neuron_class is not None:
        try:
            neuron_snaps = _compute_neuron_snapshots(ckpt_files, config, device, p)
            create_neuron_grid_animation(
                neuron_snapshots=neuron_snaps,
                snapshot_epochs=snapshot_epochs,
                p=p,
                key_freqs=key_freqs,
                final_neuron_class=neuron_class,
                fps=4,
                output_path=str(fig_dir / "neuron_grids.gif"),
            )
            logger.info("Animation saved as neuron_grids.gif")
        except Exception as e:
            logger.warning(f"Neuron grid animation failed: {e}")

    logger.info(f"All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
