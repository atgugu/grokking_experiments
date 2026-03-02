"""Trajectory visualizations: embedding PCA evolution and weight trajectory PCA."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from sklearn.decomposition import PCA

matplotlib.use("Agg")


def plot_embedding_pca_evolution(
    embedding_snapshots: list[np.ndarray],
    snapshot_epochs: list[int],
    p: int,
    n_panels: int = 6,
) -> plt.Figure:
    """PCA of W_E at multiple checkpoints showing token structure emergence.

    Args:
        embedding_snapshots: List of (p+1, d_model) or (p, d_model) embedding matrices.
        snapshot_epochs: List of epoch numbers for each snapshot.
        p: Prime modulus.
        n_panels: Number of panels to show (evenly sampled from snapshots).

    Returns:
        Figure with 2 x ceil(n_panels/3) grid of scatter plots.
    """
    n_snaps = len(embedding_snapshots)
    if n_snaps == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No embedding snapshots available",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    # Sample evenly
    n_show = min(n_panels, n_snaps)
    indices = np.linspace(0, n_snaps - 1, n_show, dtype=int)

    n_cols = min(3, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows),
                              squeeze=False)

    # Color tokens by index to reveal circular ordering
    colors = plt.cm.hsv(np.arange(p) / p)

    for plot_idx, snap_idx in enumerate(indices):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row, col]

        E = embedding_snapshots[snap_idx]
        # Take only integer token embeddings
        if E.shape[0] > p:
            E = E[:p, :]

        # PCA to 2D
        pca = PCA(n_components=2)
        E_2d = pca.fit_transform(E)

        ax.scatter(E_2d[:, 0], E_2d[:, 1], c=colors, s=15, alpha=0.8)
        ax.set_title(f"Epoch {snapshot_epochs[snap_idx]}", fontsize=10)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})", fontsize=8)
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})", fontsize=8)
        ax.grid(True, alpha=0.2)
        ax.set_aspect("equal", adjustable="datalim")

    # Hide unused axes
    for idx in range(n_show, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    fig.suptitle("Embedding PCA Evolution (colored by token index)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_weight_trajectory_pca(
    param_snapshots: list[np.ndarray],
    snapshot_epochs: list[int],
    history: dict | None = None,
) -> plt.Figure:
    """PCA of flattened parameter vectors showing training trajectory.

    Args:
        param_snapshots: List of 1D parameter vectors (one per checkpoint).
        snapshot_epochs: List of epoch numbers.
        history: Optional training history dict with 'test_acc' and 'eval_epochs'.

    Returns:
        Figure with 2D PCA trajectory colored by epoch or test accuracy.
    """
    n_snaps = len(param_snapshots)
    if n_snaps < 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Need at least 2 checkpoints for trajectory PCA",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig

    # Stack into (n_checkpoints, n_params) matrix
    param_matrix = np.stack(param_snapshots, axis=0)

    pca = PCA(n_components=2)
    projected = pca.fit_transform(param_matrix)

    # Try to color by test accuracy if available
    color_by_acc = False
    if history is not None:
        test_acc = history.get("test_acc", [])
        eval_epochs = history.get("eval_epochs", [])
        if test_acc and eval_epochs:
            # Interpolate test_acc at snapshot epochs
            acc_at_snaps = np.interp(snapshot_epochs, eval_epochs, test_acc)
            color_by_acc = True

    fig, ax = plt.subplots(figsize=(10, 8))

    if color_by_acc:
        sc = ax.scatter(projected[:, 0], projected[:, 1],
                        c=acc_at_snaps, cmap="RdYlGn", s=40, zorder=3,
                        edgecolors="black", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="Test Accuracy")
    else:
        sc = ax.scatter(projected[:, 0], projected[:, 1],
                        c=snapshot_epochs, cmap="viridis", s=40, zorder=3,
                        edgecolors="black", linewidths=0.5)
        plt.colorbar(sc, ax=ax, label="Epoch")

    # Connect points with line
    ax.plot(projected[:, 0], projected[:, 1], color="gray", linewidth=0.8,
            alpha=0.5, zorder=2)

    # Annotate start and end
    ax.annotate("Start", projected[0], fontsize=9, fontweight="bold",
                xytext=(10, 10), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="gray"))
    ax.annotate("End", projected[-1], fontsize=9, fontweight="bold",
                xytext=(10, 10), textcoords="offset points",
                arrowprops=dict(arrowstyle="->", color="gray"))

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title("Weight Trajectory PCA")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
