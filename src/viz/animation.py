"""Grokking animations: synchronized multi-panel GIFs for mechanistic insight."""

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.animation import FuncAnimation
from sklearn.decomposition import PCA

matplotlib.use("Agg")


def create_grokking_animation(
    history: dict,
    fourier_snapshots: dict,
    embedding_snapshots: list[np.ndarray],
    snapshot_epochs: list[int],
    p: int,
    key_freqs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
) -> FuncAnimation:
    """Create synchronized 4-panel animation of grokking dynamics.

    Panels:
        1. Loss curves (progressive draw)
        2. Fourier spectrum bars
        3. Embedding PCA scatter
        4. Gini coefficient line

    Args:
        history: Training history with 'train_loss', 'test_loss', 'eval_epochs', 'gini', 'fourier_epochs'.
        fourier_snapshots: Dict with 'frequency_norms' (n_snaps, p) and 'fourier_epochs'.
        embedding_snapshots: List of (p+1, d_model) or (p, d_model) embeddings.
        snapshot_epochs: List of epoch numbers for embedding snapshots.
        p: Prime modulus.
        key_freqs: Array of key frequency indices.
        fps: Frames per second for output.
        output_path: If provided, save animation as .gif.

    Returns:
        FuncAnimation object.
    """
    # Pre-compute all PCA embeddings
    pca_results = []
    colors = plt.cm.hsv(np.arange(p) / p)
    for E in embedding_snapshots:
        if E.shape[0] > p:
            E = E[:p, :]
        pca = PCA(n_components=2)
        E_2d = pca.fit_transform(E)
        pca_results.append(E_2d)

    # Fourier data
    freq_norms = fourier_snapshots["frequency_norms"]
    fourier_epochs = fourier_snapshots.get("fourier_epochs", np.arange(len(freq_norms)))

    # History data
    eval_epochs = np.array(history.get("eval_epochs", []))
    train_loss = np.array(history.get("train_loss", []))
    test_loss = np.array(history.get("test_loss", []))
    gini_vals = np.array(history.get("gini", []))
    gini_epochs = np.array(history.get("fourier_epochs", []))

    # Use fourier snapshot epochs as animation frames
    n_frames = len(fourier_epochs)
    key_set = set(int(k) for k in key_freqs)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_loss, ax_fourier, ax_pca, ax_gini = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # Setup loss axes
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training Curves")
    if len(train_loss) > 0:
        ax_loss.set_yscale("log")
        ax_loss.set_xlim(0, eval_epochs[-1] if len(eval_epochs) > 0 else 1)
        all_losses = np.concatenate([train_loss[train_loss > 0], test_loss[test_loss > 0]])
        if len(all_losses) > 0:
            ax_loss.set_ylim(all_losses.min() * 0.5, all_losses.max() * 2)
    ax_loss.grid(True, alpha=0.3)

    # Setup fourier axes
    ax_fourier.set_xlabel("Frequency k")
    ax_fourier.set_ylabel("Energy")
    ax_fourier.set_title("Fourier Spectrum")
    ax_fourier.grid(True, alpha=0.3, axis="y")

    # Setup PCA axes
    ax_pca.set_title("Embedding PCA")
    ax_pca.grid(True, alpha=0.2)
    ax_pca.set_aspect("equal", adjustable="datalim")

    # Setup Gini axes
    ax_gini.set_xlabel("Epoch")
    ax_gini.set_ylabel("Gini")
    ax_gini.set_title("Fourier Gini Coefficient")
    ax_gini.set_ylim(0, 1.05)
    if len(gini_epochs) > 0:
        ax_gini.set_xlim(0, gini_epochs[-1])
    ax_gini.grid(True, alpha=0.3)

    # Artists to update
    train_line, = ax_loss.plot([], [], color="#636EFA", label="Train", linewidth=1.5)
    test_line, = ax_loss.plot([], [], color="#EF553B", label="Test", linewidth=1.5)
    ax_loss.legend(fontsize=8)

    gini_line, = ax_gini.plot([], [], color="#AB63FA", linewidth=2)

    fig.tight_layout()
    title = fig.suptitle("", fontsize=12, fontweight="bold", y=1.02)

    def update(frame_idx):
        current_epoch = fourier_epochs[frame_idx]
        title.set_text(f"Epoch {int(current_epoch)}")

        # Update loss curves (show up to current epoch)
        if len(eval_epochs) > 0:
            mask = eval_epochs <= current_epoch
            train_line.set_data(eval_epochs[mask], train_loss[mask])
            test_line.set_data(eval_epochs[mask], test_loss[mask])

        # Update Fourier spectrum
        ax_fourier.cla()
        ax_fourier.set_xlabel("Frequency k")
        ax_fourier.set_ylabel("Energy")
        ax_fourier.set_title("Fourier Spectrum")
        ax_fourier.grid(True, alpha=0.3, axis="y")
        norms = freq_norms[frame_idx]
        bar_colors = ["#EF553B" if k in key_set else "#636EFA" for k in range(p)]
        ax_fourier.bar(range(p), norms, color=bar_colors, width=1.0, edgecolor="none")

        # Update embedding PCA
        ax_pca.cla()
        ax_pca.set_title("Embedding PCA")
        ax_pca.grid(True, alpha=0.2)
        # Find closest embedding snapshot
        if pca_results:
            snap_idx = np.argmin(np.abs(np.array(snapshot_epochs) - current_epoch))
            E_2d = pca_results[snap_idx]
            ax_pca.scatter(E_2d[:, 0], E_2d[:, 1], c=colors, s=10, alpha=0.8)
            ax_pca.set_aspect("equal", adjustable="datalim")

        # Update Gini line
        if len(gini_epochs) > 0:
            gini_mask = gini_epochs <= current_epoch
            gini_line.set_data(gini_epochs[gini_mask], gini_vals[gini_mask])

        return [train_line, test_line, gini_line, title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)

    if output_path is not None:
        anim.save(output_path, writer="pillow", fps=fps)

    return anim


# ---------------------------------------------------------------------------
# Helper: synchronized loss panel used by multiple animations
# ---------------------------------------------------------------------------

def _setup_loss_axes(ax, history):
    """Configure a log-scale loss axes and return (train_line, test_line, eval_epochs, train_loss, test_loss)."""
    eval_epochs = np.array(history.get("eval_epochs", []))
    train_loss = np.array(history.get("train_loss", []))
    test_loss = np.array(history.get("test_loss", []))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Curves")
    if len(train_loss) > 0:
        ax.set_yscale("log")
        ax.set_xlim(0, eval_epochs[-1] if len(eval_epochs) > 0 else 1)
        all_losses = np.concatenate([train_loss[train_loss > 0], test_loss[test_loss > 0]])
        if len(all_losses) > 0:
            ax.set_ylim(all_losses.min() * 0.5, all_losses.max() * 2)
    ax.grid(True, alpha=0.3)

    (train_line,) = ax.plot([], [], color="#636EFA", label="Train", linewidth=1.5)
    (test_line,) = ax.plot([], [], color="#EF553B", label="Test", linewidth=1.5)
    ax.legend(fontsize=8)

    return train_line, test_line, eval_epochs, train_loss, test_loss


def _update_loss_panel(current_epoch, eval_epochs, train_loss, test_loss, train_line, test_line, marker_line, ax):
    """Update loss curves up to *current_epoch* and move the vertical epoch marker."""
    if len(eval_epochs) > 0:
        mask = eval_epochs <= current_epoch
        train_line.set_data(eval_epochs[mask], train_loss[mask])
        test_line.set_data(eval_epochs[mask], test_loss[mask])
    marker_line.set_xdata([current_epoch, current_epoch])


# ---------------------------------------------------------------------------
# 1. Fourier Spectrum Waterfall
# ---------------------------------------------------------------------------

def create_fourier_waterfall_animation(
    fourier_snapshots: dict,
    history: dict,
    p: int,
    key_freqs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
) -> FuncAnimation:
    """Create 2-panel animation: epoch×frequency heatmap + loss curves.

    Left panel grows row-by-row as each Fourier snapshot arrives, showing
    the temporal emergence of each frequency. Key frequency columns are
    highlighted with red overlay lines.

    Args:
        fourier_snapshots: Dict with 'frequency_norms' (n_snaps, p//2+1) and 'fourier_epochs'.
        history: Training history dict.
        p: Prime modulus.
        key_freqs: Array of key frequency indices.
        fps: Frames per second.
        output_path: If provided, save as .gif.

    Returns:
        FuncAnimation object.
    """
    freq_norms = np.array(fourier_snapshots["frequency_norms"])
    fourier_epochs = np.array(
        fourier_snapshots.get("fourier_epochs", np.arange(len(freq_norms)))
    )
    n_frames = len(fourier_epochs)
    n_freqs = freq_norms.shape[1]

    # Build the full waterfall matrix upfront; reveal rows progressively
    waterfall = freq_norms.copy()
    vmax = np.percentile(waterfall, 99) if waterfall.size > 0 else 1.0

    fig, (ax_heat, ax_loss) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap axes
    ax_heat.set_xlabel("Frequency k")
    ax_heat.set_ylabel("Epoch")
    ax_heat.set_title("Fourier Spectrum Waterfall")

    # Initial empty image (all zeros)
    blank = np.full_like(waterfall, np.nan)
    im = ax_heat.imshow(
        blank,
        aspect="auto",
        origin="lower",
        cmap="inferno",
        vmin=0,
        vmax=vmax,
        extent=[0, n_freqs, fourier_epochs[0], fourier_epochs[-1]],
    )
    fig.colorbar(im, ax=ax_heat, label="Energy", shrink=0.8)

    # Key-frequency indicator lines
    key_set = set(int(k) for k in key_freqs)
    for k in key_set:
        if k < n_freqs:
            ax_heat.axvline(k + 0.5, color="red", alpha=0.4, linewidth=0.8, linestyle="--")

    # Loss panel
    train_line, test_line, eval_epochs, train_loss, test_loss = _setup_loss_axes(ax_loss, history)
    marker_line = ax_loss.axvline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)

    fig.tight_layout()
    title = fig.suptitle("", fontsize=12, fontweight="bold", y=1.02)

    def update(frame_idx):
        current_epoch = fourier_epochs[frame_idx]
        title.set_text(f"Epoch {int(current_epoch)}")

        # Reveal rows up to frame_idx
        revealed = blank.copy()
        revealed[: frame_idx + 1] = waterfall[: frame_idx + 1]
        im.set_data(revealed)

        _update_loss_panel(
            current_epoch, eval_epochs, train_loss, test_loss,
            train_line, test_line, marker_line, ax_loss,
        )
        return [im, train_line, test_line, marker_line, title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    if output_path is not None:
        anim.save(output_path, writer="pillow", fps=fps)
    return anim


# ---------------------------------------------------------------------------
# 2. Embedding Circle Formation
# ---------------------------------------------------------------------------

def create_embedding_circle_animation(
    embedding_snapshots: list[np.ndarray],
    snapshot_epochs: list[int],
    p: int,
    key_freqs: np.ndarray,
    fps: int = 4,
    output_path: str | None = None,
) -> FuncAnimation:
    """Animate token embeddings projected onto Fourier cos/sin bases per key frequency.

    Each subplot shows the p tokens projected onto (cos 2πk·t/p, sin 2πk·t/p) for one
    key frequency k.  Over training, scattered noise → organised circles.

    Args:
        embedding_snapshots: List of (p+1, d_model) or (p, d_model) embeddings.
        snapshot_epochs: Epoch for each snapshot.
        p: Prime modulus.
        key_freqs: Array of key frequency indices (max 4 shown).
        fps: Frames per second.
        output_path: If provided, save as .gif.

    Returns:
        FuncAnimation object.
    """
    freqs = np.array(key_freqs[:4], dtype=int)  # cap at 4
    n_freqs = len(freqs)
    n_frames = len(snapshot_epochs)

    # Fourier basis vectors for each key frequency
    t = np.arange(p)
    cos_bases = {k: np.cos(2 * np.pi * k * t / p) for k in freqs}
    sin_bases = {k: np.sin(2 * np.pi * k * t / p) for k in freqs}

    colors = plt.cm.hsv(t / p)

    fig, axes = plt.subplots(1, n_freqs, figsize=(4 * n_freqs, 4), squeeze=False)
    axes = axes[0]
    for i, k in enumerate(freqs):
        axes[i].set_title(f"Freq k={k}")
        axes[i].set_aspect("equal", adjustable="datalim")
        axes[i].grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    title = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(frame_idx):
        epoch = snapshot_epochs[frame_idx]
        title.set_text(f"Epoch {epoch}")

        E = embedding_snapshots[frame_idx]
        if E.shape[0] > p:
            E = E[:p, :]

        for i, k in enumerate(freqs):
            ax = axes[i]
            ax.cla()
            ax.set_title(f"Freq k={k}")
            ax.set_aspect("equal", adjustable="datalim")
            ax.grid(True, alpha=0.2)

            # Project each token onto Fourier basis: x = E @ cos_k, y = E @ sin_k
            # cos_bases[k] is (p,), E is (p, d_model) — we want per-token scalar
            proj_cos = E @ (E.T @ cos_bases[k]) / (np.linalg.norm(E.T @ cos_bases[k]) + 1e-12)
            proj_sin = E @ (E.T @ sin_bases[k]) / (np.linalg.norm(E.T @ sin_bases[k]) + 1e-12)
            # Simpler and more interpretable: use 1D DFT projection
            x = E.T @ cos_bases[k]  # (d_model,) — use as projection direction
            y = E.T @ sin_bases[k]  # (d_model,)
            # Per-token coordinates
            coords_x = E @ x / (np.linalg.norm(x) + 1e-12)
            coords_y = E @ y / (np.linalg.norm(y) + 1e-12)

            ax.scatter(coords_x, coords_y, c=colors, s=18, alpha=0.85)
            ax.set_xlabel("cos component")
            ax.set_ylabel("sin component")

        return []

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    if output_path is not None:
        anim.save(output_path, writer="pillow", fps=fps)
    return anim


# ---------------------------------------------------------------------------
# 3. Per-Sample Loss Landscape
# ---------------------------------------------------------------------------

def create_loss_landscape_animation(
    loss_snapshots: list[np.ndarray],
    snapshot_epochs: list[int],
    p: int,
    train_mask: np.ndarray | None = None,
    history: dict | None = None,
    fps: int = 4,
    output_path: str | None = None,
) -> FuncAnimation:
    """Animate per-sample CE loss over the full p×p input grid.

    Left panel: p×p heatmap of CE loss with optional train/test boundary.
    Right panel: synchronized loss curves (if history provided).

    Args:
        loss_snapshots: List of (p, p) per-sample CE loss arrays.
        snapshot_epochs: Epoch for each snapshot.
        p: Prime modulus.
        train_mask: (p, p) bool array — True for training samples.
        history: Training history dict (optional, for loss curves panel).
        fps: Frames per second.
        output_path: If provided, save as .gif.

    Returns:
        FuncAnimation object.
    """
    n_frames = len(snapshot_epochs)

    # Determine color scale from all snapshots
    all_losses = np.stack(loss_snapshots)
    vmax = float(np.percentile(all_losses[np.isfinite(all_losses)], 97))
    vmin = 0.0

    has_history = history is not None and len(history.get("train_loss", [])) > 0
    ncols = 2 if has_history else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5), squeeze=False)
    ax_heat = axes[0, 0]

    ax_heat.set_xlabel("b")
    ax_heat.set_ylabel("a")
    ax_heat.set_title("Per-Sample CE Loss")

    im = ax_heat.imshow(
        np.zeros((p, p)), origin="lower", cmap="viridis", vmin=vmin, vmax=vmax,
    )
    fig.colorbar(im, ax=ax_heat, label="CE Loss", shrink=0.8)

    # Train/test boundary overlay
    contour_artist = [None]
    if train_mask is not None:
        # We'll draw the boundary each frame (it doesn't change, but we redraw after cla-free approach)
        pass

    # Loss panel
    marker_line = None
    train_line = test_line = eval_epochs = train_loss_arr = test_loss_arr = None
    if has_history:
        ax_loss = axes[0, 1]
        train_line, test_line, eval_epochs, train_loss_arr, test_loss_arr = _setup_loss_axes(ax_loss, history)
        marker_line = ax_loss.axvline(0, color="gray", linewidth=1, linestyle="--", alpha=0.6)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    title = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(frame_idx):
        epoch = snapshot_epochs[frame_idx]
        title.set_text(f"Epoch {epoch}")

        im.set_data(loss_snapshots[frame_idx])

        # Redraw train/test contour
        if train_mask is not None:
            # Remove old contour
            if contour_artist[0] is not None:
                for c in contour_artist[0].collections:
                    c.remove()
            contour_artist[0] = ax_heat.contour(
                train_mask.astype(float), levels=[0.5], colors=["cyan"],
                linewidths=[1.0], origin="lower",
            )

        if marker_line is not None:
            _update_loss_panel(
                epoch, eval_epochs, train_loss_arr, test_loss_arr,
                train_line, test_line, marker_line, axes[0, 1],
            )

        return [im, title]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    if output_path is not None:
        anim.save(output_path, writer="pillow", fps=fps)
    return anim


# ---------------------------------------------------------------------------
# 4. Neuron Activation Grid Evolution
# ---------------------------------------------------------------------------

def create_neuron_grid_animation(
    neuron_snapshots: list[np.ndarray],
    snapshot_epochs: list[int],
    p: int,
    key_freqs: np.ndarray,
    final_neuron_class: dict,
    neurons_per_freq: int = 2,
    fps: int = 4,
    output_path: str | None = None,
) -> FuncAnimation:
    """Animate neuron activation heatmaps over training.

    Grid layout: rows = key frequencies, cols = top neurons per frequency
    (selected from the *final* model's classification for consistency).

    Args:
        neuron_snapshots: List of (p, p, d_mlp) activation arrays.
        snapshot_epochs: Epoch for each snapshot.
        p: Prime modulus.
        key_freqs: Array of key frequency indices.
        final_neuron_class: Neuron classification dict from final model with 'clusters'.
        neurons_per_freq: Number of neurons to show per frequency.
        fps: Frames per second.
        output_path: If provided, save as .gif.

    Returns:
        FuncAnimation object.
    """
    clusters = final_neuron_class.get("clusters", {})
    r_squared = final_neuron_class.get("r_squared", np.array([]))

    # Select neurons: for each key freq, pick top neurons_per_freq by r²
    grid_spec = []  # list of (freq, neuron_idx) tuples
    for k in key_freqs:
        k = int(k)
        neurons = clusters.get(k, [])
        if not neurons:
            continue
        # Sort by r² descending
        if len(r_squared) > 0:
            neurons_sorted = sorted(neurons, key=lambda n: r_squared[n], reverse=True)
        else:
            neurons_sorted = neurons
        for n in neurons_sorted[:neurons_per_freq]:
            grid_spec.append((k, n))

    if not grid_spec:
        # Fallback: nothing to show
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.text(0.5, 0.5, "No neuron clusters found", ha="center", va="center", transform=ax.transAxes)
        anim = FuncAnimation(fig, lambda i: [], frames=1, interval=500)
        if output_path is not None:
            anim.save(output_path, writer="pillow", fps=fps)
        return anim

    # Determine grid dimensions
    freq_list = list(dict.fromkeys(k for k, _ in grid_spec))  # unique freqs, ordered
    n_rows = len(freq_list)
    n_cols = neurons_per_freq
    freq_to_row = {k: i for i, k in enumerate(freq_list)}

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 3 * n_rows), squeeze=False)
    n_frames = len(snapshot_epochs)

    # Determine global color scale from final snapshot
    vmax = float(np.percentile(np.abs(neuron_snapshots[-1]), 99)) if neuron_snapshots else 1.0

    # Place images
    images = {}
    placed = set()
    for k, n in grid_spec:
        row = freq_to_row[k]
        col = sum(1 for kk, nn in grid_spec if kk == k and nn < n)
        if col >= n_cols:
            continue
        ax = axes[row, col]
        im = ax.imshow(np.zeros((p, p)), origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"k={k}, n={n}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        images[(k, n)] = (im, row, col)
        placed.add((row, col))

    # Hide unused axes
    for r in range(n_rows):
        for c in range(n_cols):
            if (r, c) not in placed:
                axes[r, c].axis("off")

    # Row labels
    for k, row in freq_to_row.items():
        axes[row, 0].set_ylabel(f"Freq {k}", fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    title = fig.suptitle("", fontsize=12, fontweight="bold")

    def update(frame_idx):
        epoch = snapshot_epochs[frame_idx]
        title.set_text(f"Epoch {epoch}")
        acts = neuron_snapshots[frame_idx]  # (p, p, d_mlp)

        for (k, n), (im, _, _) in images.items():
            im.set_data(acts[:, :, n])

        return [title] + [im for im, _, _ in images.values()]

    anim = FuncAnimation(fig, update, frames=n_frames, interval=1000 // fps, blit=False)
    if output_path is not None:
        anim.save(output_path, writer="pillow", fps=fps)
    return anim
