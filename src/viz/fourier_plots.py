"""Fourier analysis visualizations."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def plot_frequency_spectrum(
    frequency_norms: np.ndarray, key_freqs: np.ndarray, p: int
) -> plt.Figure:
    """Bar chart of frequency energy, highlighting key frequencies.

    Args:
        frequency_norms: (p,) marginal frequency energies.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    colors = ["#EF553B" if k in key_freqs else "#636EFA" for k in range(p)]
    ax.bar(range(p), frequency_norms, color=colors, width=1.0, edgecolor="none")

    for k in key_freqs:
        ax.annotate(f"k={k}", (k, frequency_norms[k]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#EF553B", fontweight="bold")

    ax.set_xlabel("Frequency k")
    ax.set_ylabel("Energy")
    ax.set_title(f"Fourier Frequency Spectrum (p={p})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_fourier_heatmap(component_norms: np.ndarray, p: int) -> plt.Figure:
    """2D heatmap of Fourier component norms |L_hat[k1, k2]|^2.

    Args:
        component_norms: (p, p) array of total Fourier energy per (k1, k2).
        p: Prime modulus.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use log scale for better visibility
    norms_log = np.log10(component_norms + 1e-10)
    im = ax.imshow(norms_log, origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="log10(energy)")

    ax.set_xlabel("Frequency k2")
    ax.set_ylabel("Frequency k1")
    ax.set_title(f"2D Fourier Component Energy (p={p})")

    # Show ticks at intervals
    tick_step = max(1, p // 10)
    ax.set_xticks(range(0, p, tick_step))
    ax.set_yticks(range(0, p, tick_step))

    fig.tight_layout()
    return fig


def plot_fourier_evolution(
    fourier_snapshots: dict, key_freqs: np.ndarray, p: int
) -> plt.Figure:
    """Plot key frequency norms over training epochs.

    Args:
        fourier_snapshots: Dict with 'frequency_norms' (n_snaps, p) and 'fourier_epochs'.
        key_freqs: Array of key frequency indices to track.
        p: Prime modulus.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    freq_norms = fourier_snapshots["frequency_norms"]  # (n_snapshots, p)
    epochs = fourier_snapshots.get("fourier_epochs", np.arange(len(freq_norms)))

    colors = plt.cm.Set1(np.linspace(0, 1, max(len(key_freqs), 1)))
    for i, k in enumerate(key_freqs):
        ax.plot(epochs, freq_norms[:, k], label=f"k={k}",
                color=colors[i % len(colors)], linewidth=2)

    # Also plot sum of non-key frequencies (noise floor)
    all_freqs = set(range(p))
    key_set = set(int(k) for k in key_freqs) | {0}
    noise_freqs = list(all_freqs - key_set)
    if noise_freqs:
        noise_energy = freq_norms[:, noise_freqs].sum(axis=1)
        ax.plot(epochs, noise_energy, label="Noise (sum)", color="gray",
                linewidth=1, linestyle="--", alpha=0.7)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Frequency Energy")
    ax.set_title("Key Frequency Evolution During Training")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def plot_embedding_fourier(
    embed_fourier: dict, key_freqs: np.ndarray, p: int
) -> plt.Figure:
    """Plot Fourier analysis of the embedding matrix.

    Args:
        embed_fourier: Dict from fourier_embed_analysis with 'frequency_energy'.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    freq_energy = embed_fourier["frequency_energy"]
    colors = ["#EF553B" if k in key_freqs else "#636EFA" for k in range(p)]
    ax.bar(range(p), freq_energy, color=colors, width=1.0, edgecolor="none")

    for k in key_freqs:
        ax.annotate(f"k={k}", (k, freq_energy[k]),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8, color="#EF553B", fontweight="bold")

    ax.set_xlabel("Frequency k")
    ax.set_ylabel("Energy")
    ax.set_title(f"Embedding Matrix (W_E) Fourier Spectrum (p={p})")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    return fig


def plot_fourier_spectrum_strip(
    fourier_snapshots: dict,
    key_freqs: np.ndarray,
    p: int,
    n_panels: int = 6,
) -> plt.Figure:
    """Side-by-side bar charts of Fourier spectra at multiple epochs.

    Shows the full sparsification story in one image: from uniform to sparse.

    Args:
        fourier_snapshots: Dict with 'frequency_norms' (n_snaps, p) and 'fourier_epochs'.
        key_freqs: Array of key frequency indices.
        p: Prime modulus.
        n_panels: Number of epochs to sample.

    Returns:
        Figure with 1 row x n_panels columns of bar charts.
    """
    freq_norms = fourier_snapshots["frequency_norms"]  # (n_snapshots, p)
    epochs = fourier_snapshots.get("fourier_epochs", np.arange(len(freq_norms)))

    n_snaps = len(freq_norms)
    n_show = min(n_panels, n_snaps)
    indices = np.linspace(0, n_snaps - 1, n_show, dtype=int)

    fig, axes = plt.subplots(1, n_show, figsize=(4 * n_show, 4), sharey=True,
                              squeeze=False)
    axes = axes[0]

    key_set = set(int(k) for k in key_freqs)

    for i, snap_idx in enumerate(indices):
        ax = axes[i]
        norms = freq_norms[snap_idx]
        colors = ["#EF553B" if k in key_set else "#636EFA" for k in range(p)]
        ax.bar(range(p), norms, color=colors, width=1.0, edgecolor="none")
        ax.set_xlabel("Freq k", fontsize=8)
        ax.set_title(f"Epoch {int(epochs[snap_idx])}", fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        if i == 0:
            ax.set_ylabel("Energy")

    fig.suptitle("Fourier Spectrum Evolution", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig
