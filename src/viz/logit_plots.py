"""Logit table visualizations: heatmap comparisons, 3D surfaces, per-sample loss."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy.special import log_softmax

matplotlib.use("Agg")


def _extract_correct_logits(logit_table: np.ndarray, p: int) -> np.ndarray:
    """Extract correct-class logit for each (a, b) pair.

    Args:
        logit_table: (p, p, p) logit table.
        p: Prime modulus.

    Returns:
        (p, p) array of correct-class logits.
    """
    a = np.arange(p)
    b = np.arange(p)
    targets = (a[:, None] + b[None, :]) % p  # (p, p)
    correct = logit_table[
        a[:, None] * np.ones((1, p), dtype=int),
        np.ones((p, 1), dtype=int) * b[None, :],
        targets,
    ]
    return correct


def plot_logit_heatmap_comparison(
    logit_table: np.ndarray,
    restricted_logits: np.ndarray,
    p: int,
) -> plt.Figure:
    """Side-by-side heatmaps of correct-class logits: full, restricted, difference.

    Args:
        logit_table: (p, p, p) full logit table.
        restricted_logits: (p, p, p) restricted logit table (key freqs only).
        p: Prime modulus.

    Returns:
        Figure with 3 panels.
    """
    full_correct = _extract_correct_logits(logit_table, p)
    restr_correct = _extract_correct_logits(restricted_logits, p)
    diff = full_correct - restr_correct

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (full_correct, "Full Model", "viridis"),
        (restr_correct, "Key Frequencies Only", "viridis"),
        (diff, "Difference (Full - Restricted)", "RdBu_r"),
    ]

    for ax, (data, title, cmap) in zip(axes, panels):
        if cmap == "RdBu_r":
            vmax = np.abs(data).max()
            im = ax.imshow(data, origin="lower", aspect="equal", cmap=cmap,
                           vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(data, origin="lower", aspect="equal", cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("b")
        ax.set_ylabel("a")
        ax.set_title(title, fontsize=11)

    fig.suptitle(f"Correct-Class Logits (p={p}): Full vs Restricted",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_correct_logit_surface(logit_table: np.ndarray, p: int) -> plt.Figure:
    """3D surface plot of correct-class logits over (a, b).

    Args:
        logit_table: (p, p, p) logit table.
        p: Prime modulus.

    Returns:
        Figure with 3D surface.
    """
    correct = _extract_correct_logits(logit_table, p)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    a = np.arange(p)
    b = np.arange(p)
    A, B = np.meshgrid(a, b, indexing="ij")

    # Use stride for large p to avoid overplotting
    stride = max(1, p // 30)
    ax.plot_surface(A, B, correct, cmap="viridis", alpha=0.8,
                    rstride=stride, cstride=stride, edgecolor="none")

    ax.set_xlabel("a")
    ax.set_ylabel("b")
    ax.set_zlabel("Correct-class logit")
    ax.set_title(f"Correct-Class Logit Surface (p={p})")
    fig.tight_layout()
    return fig


def plot_per_sample_loss_heatmap(
    logit_table: np.ndarray,
    p: int,
    train_mask: np.ndarray | None = None,
) -> plt.Figure:
    """Heatmap of per-sample cross-entropy loss with optional train/test overlay.

    Args:
        logit_table: (p, p, p) logit table.
        p: Prime modulus.
        train_mask: Optional (p, p) boolean mask, True for training samples.

    Returns:
        Figure with loss heatmap.
    """
    # Compute per-sample CE loss
    a = np.arange(p)
    b = np.arange(p)
    targets = (a[:, None] + b[None, :]) % p  # (p, p)

    # log_softmax along the class dimension
    log_probs = log_softmax(logit_table, axis=2)  # (p, p, p)
    # Extract log prob of correct class
    loss = -log_probs[
        a[:, None] * np.ones((1, p), dtype=int),
        np.ones((p, 1), dtype=int) * b[None, :],
        targets,
    ]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(loss, origin="lower", aspect="equal", cmap="hot_r")
    plt.colorbar(im, ax=ax, label="Cross-Entropy Loss")

    # Overlay train/test boundary
    if train_mask is not None:
        # Show test region with hatching
        test_mask = ~train_mask
        ax.contour(test_mask.astype(float), levels=[0.5], colors=["cyan"],
                   linewidths=0.5, alpha=0.7, origin="lower")

    ax.set_xlabel("b")
    ax.set_ylabel("a")
    ax.set_title(f"Per-Sample Loss (p={p})")
    fig.tight_layout()
    return fig
