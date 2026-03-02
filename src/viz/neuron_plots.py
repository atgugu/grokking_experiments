"""Neuron-level visualizations: activation grids, logit map, frequency spectrum."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def plot_neuron_activation_grids(
    neuron_activations_grid: np.ndarray,
    neuron_class: dict,
    key_freqs: np.ndarray,
    p: int,
    neurons_per_freq: int = 3,
) -> plt.Figure:
    """Plot p x p activation heatmaps for top neurons grouped by frequency.

    Args:
        neuron_activations_grid: (p, p, d_mlp) activations on full grid.
        neuron_class: Dict from classify_neuron_frequencies with
            'dominant_freq', 'r_squared', 'clusters'.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.
        neurons_per_freq: Max neurons to show per frequency cluster.

    Returns:
        Figure with rows=frequencies, cols=neurons_per_freq of p x p heatmaps.
    """
    clusters = neuron_class["clusters"]
    r_squared = neuron_class["r_squared"]

    # Filter to key frequencies that have clusters
    active_freqs = [int(k) for k in key_freqs if int(k) in clusters]
    if not active_freqs:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No neuron clusters found for key frequencies",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.set_title("Neuron Activation Grids")
        return fig

    n_rows = len(active_freqs)
    n_cols = neurons_per_freq

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows),
                              squeeze=False)

    for row, freq in enumerate(active_freqs):
        neuron_indices = clusters[freq]
        # Sort by R^2 descending, pick top neurons_per_freq
        sorted_neurons = sorted(neuron_indices, key=lambda n: r_squared[n], reverse=True)
        selected = sorted_neurons[:n_cols]

        for col in range(n_cols):
            ax = axes[row, col]
            if col < len(selected):
                n_idx = selected[col]
                act = neuron_activations_grid[:, :, n_idx]
                im = ax.imshow(act, origin="lower", aspect="equal", cmap="viridis")
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f"k={freq}, n={n_idx}\nR²={r_squared[n_idx]:.2f}",
                             fontsize=9)
            else:
                ax.set_visible(False)

            if col == 0:
                ax.set_ylabel(f"a (freq k={freq})")
            if row == n_rows - 1:
                ax.set_xlabel("b")

    fig.suptitle("Neuron Activation Grids (grouped by dominant frequency)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_neuron_logit_map(
    neuron_logit_map: np.ndarray,
    neuron_class: dict,
    key_freqs: np.ndarray,
    p: int,
) -> plt.Figure:
    """Heatmap of the neuron-to-logit map W_L, neurons sorted by frequency.

    Args:
        neuron_logit_map: (d_mlp, p) array from compute_neuron_logit_map.
        neuron_class: Dict from classify_neuron_frequencies.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.

    Returns:
        Figure with (d_mlp, p) heatmap, neurons sorted by dominant_freq then R².
    """
    dominant_freq = neuron_class["dominant_freq"]
    r_squared = neuron_class["r_squared"]
    d_mlp = neuron_logit_map.shape[0]

    # Sort neurons: by dominant_freq, then by R² descending
    sort_order = np.lexsort((-r_squared, dominant_freq))
    sorted_map = neuron_logit_map[sort_order, :]
    sorted_freqs = dominant_freq[sort_order]

    fig, ax = plt.subplots(figsize=(14, 8))
    vmax = np.abs(sorted_map).max()
    im = ax.imshow(sorted_map, aspect="auto", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax, origin="lower")
    plt.colorbar(im, ax=ax, label="W_L weight")

    # Add horizontal separators between frequency groups
    key_set = set(int(k) for k in key_freqs)
    prev_freq = sorted_freqs[0]
    for i in range(1, d_mlp):
        if sorted_freqs[i] != prev_freq:
            if int(prev_freq) in key_set or int(sorted_freqs[i]) in key_set:
                ax.axhline(i - 0.5, color="black", linewidth=0.5, alpha=0.5)
            prev_freq = sorted_freqs[i]

    # Label key frequency groups on y-axis
    freq_positions = {}
    for i, f in enumerate(sorted_freqs):
        freq_positions.setdefault(int(f), []).append(i)
    ytick_pos = []
    ytick_labels = []
    for f in sorted(freq_positions.keys()):
        if f in key_set:
            positions = freq_positions[f]
            ytick_pos.append(np.mean(positions))
            ytick_labels.append(f"k={f}")
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_labels, fontsize=8)

    ax.set_xlabel("Output Class c")
    ax.set_ylabel("Neuron (sorted by dominant freq)")
    ax.set_title(f"Neuron-Logit Map W_L ({d_mlp} x {p})")
    fig.tight_layout()
    return fig


def plot_neuron_frequency_spectrum_heatmap(
    neuron_spectrum: np.ndarray,
    neuron_class: dict,
    key_freqs: np.ndarray,
    p: int,
) -> plt.Figure:
    """Heatmap of per-neuron frequency energy, neurons sorted by frequency.

    Args:
        neuron_spectrum: (d_mlp, p) from compute_neuron_frequency_spectrum.
        neuron_class: Dict from classify_neuron_frequencies.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.

    Returns:
        Figure with (d_mlp, p) heatmap on log scale, key freq columns highlighted.
    """
    dominant_freq = neuron_class["dominant_freq"]
    r_squared = neuron_class["r_squared"]
    d_mlp = neuron_spectrum.shape[0]

    # Sort neurons by dominant_freq then R²
    sort_order = np.lexsort((-r_squared, dominant_freq))
    sorted_spectrum = neuron_spectrum[sort_order, :]

    fig, ax = plt.subplots(figsize=(14, 8))
    # Log scale for better visibility
    log_spectrum = np.log10(sorted_spectrum + 1e-10)
    im = ax.imshow(log_spectrum, aspect="auto", cmap="hot", origin="lower")
    plt.colorbar(im, ax=ax, label="log10(frequency energy)")

    # Highlight key frequency columns
    for k in key_freqs:
        ax.axvline(int(k), color="cyan", linewidth=0.8, alpha=0.5)
        ax.text(int(k), d_mlp + 1, f"k={int(k)}", ha="center", fontsize=7,
                color="cyan", fontweight="bold", clip_on=False)

    ax.set_xlabel("Frequency k")
    ax.set_ylabel("Neuron (sorted by dominant freq)")
    ax.set_title(f"Neuron Frequency Spectrum ({d_mlp} x {p})")
    ax.set_xlim(-0.5, p - 0.5)
    fig.tight_layout()
    return fig
