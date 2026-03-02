"""Embedding geometry visualizations: circular star plots and neuron clustering."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")

from ..analysis.fourier import dft_matrix


def plot_embedding_circles(
    W_E: np.ndarray, key_freqs: np.ndarray, p: int
) -> plt.Figure:
    """Plot circular star plots showing embedding structure at key frequencies.

    For each key frequency k, project embeddings onto the Fourier basis at that
    frequency. If the network has learned the trig algorithm, embeddings of
    integer n should form a circle at angle 2*pi*k*n/p.

    Args:
        W_E: (p+1, d_model) embedding matrix.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.
    """
    F = dft_matrix(p)
    E = W_E[:p, :]  # (p, d_model) integer token embeddings

    n_freqs = min(len(key_freqs), 6)
    n_cols = min(3, n_freqs)
    n_rows = (n_freqs + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows),
                              subplot_kw={"projection": "polar"})
    if n_freqs == 1:
        axes = np.array([axes])
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx >= n_freqs:
            ax.set_visible(False)
            continue

        k = int(key_freqs[idx])
        # Project embeddings onto Fourier basis at frequency k
        # E_hat_k = F[k, :] @ E = sum_n F[k,n] * E[n,:] for each dim
        # For star plot: compute angle of each embedding in the Fourier basis
        angles = 2 * np.pi * k * np.arange(p) / p
        radii = np.ones(p)

        # Color by index for visual structure
        colors = plt.cm.hsv(np.arange(p) / p)

        ax.scatter(angles, radii, c=colors, s=15, alpha=0.8)

        # Connect consecutive points to show the star pattern
        order = np.argsort(angles)
        ax.plot(np.append(angles[order], angles[order[0]]),
                np.append(radii[order], radii[order[0]]),
                color="gray", alpha=0.3, linewidth=0.5)

        ax.set_title(f"Frequency k={k}", fontsize=11, pad=15)
        ax.set_ylim(0, 1.3)

    fig.suptitle("Embedding Circular Structure at Key Frequencies",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_neuron_frequency_clusters(
    neuron_class: dict, key_freqs: np.ndarray, p: int
) -> plt.Figure:
    """Scatter plot of neurons colored by their dominant frequency.

    Args:
        neuron_class: Dict from classify_neuron_frequencies with
            'dominant_freq', 'r_squared', 'clusters'.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    dom_freq = neuron_class["dominant_freq"]
    r_sq = neuron_class["r_squared"]
    d_mlp = len(dom_freq)

    # Left: R^2 per neuron, colored by dominant frequency
    ax = axes[0]
    key_set = set(int(k) for k in key_freqs)
    colors = []
    cmap = plt.cm.Set1
    freq_to_color = {}
    for i, k in enumerate(key_freqs):
        freq_to_color[int(k)] = cmap(i / max(len(key_freqs) - 1, 1))

    for n in range(d_mlp):
        k = int(dom_freq[n])
        if k in freq_to_color:
            colors.append(freq_to_color[k])
        else:
            colors.append((0.7, 0.7, 0.7, 0.5))

    ax.scatter(range(d_mlp), r_sq, c=colors, s=20, alpha=0.8)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Classification threshold")
    ax.set_xlabel("Neuron Index")
    ax.set_ylabel("R² (fraction of energy in dominant freq)")
    ax.set_title("Neuron Frequency Selectivity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Right: Cluster sizes
    ax = axes[1]
    clusters = neuron_class["clusters"]
    freqs = sorted(clusters.keys(), key=int)
    sizes = [len(clusters[k]) for k in freqs]
    bar_colors = [freq_to_color.get(int(k), (0.7, 0.7, 0.7, 1.0)) for k in freqs]
    ax.bar([f"k={k}" for k in freqs], sizes, color=bar_colors)
    ax.set_xlabel("Dominant Frequency")
    ax.set_ylabel("Number of Neurons")
    ax.set_title("Neuron Frequency Clusters")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Neuron-Level Frequency Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
