"""Neuron-level analysis for understanding the learned algorithm.

Analyzes individual MLP neurons to determine which Fourier frequencies
each neuron computes, following Nanda et al. (2023) Section 4.

Key findings from the paper:
- Neurons cluster into groups, each computing a single frequency
- The neuron-logit map W_L = W_out @ W_U reveals frequency selectivity
- Neuron activations on the full (a,b) grid show periodic patterns
"""

import numpy as np
from .fourier import dft_matrix


def compute_neuron_logit_map(W_U: np.ndarray, W_out: np.ndarray) -> np.ndarray:
    """Compute the neuron-to-logit map W_L = W_out @ W_U.T.

    Each row of W_L shows how neuron i contributes to each output class.

    Args:
        W_U: (d_model, p) unembedding weight matrix.
        W_out: (d_mlp, d_model) output weights of MLP (second linear layer).

    Returns:
        (d_mlp, p) neuron-logit map.
    """
    return W_out @ W_U  # (d_mlp, d_model) @ (d_model, p) = (d_mlp, p)


def classify_neuron_frequencies(
    neuron_activations: np.ndarray,
    p: int,
    threshold: float = 0.5,
) -> dict:
    """Classify each neuron's dominant frequency from its activation pattern.

    For each neuron, compute the 2D DFT of its activations over the (a, b) grid,
    then identify which frequency carries the most energy.

    Args:
        neuron_activations: (p, p, d_mlp) activations on full grid.
            neuron_activations[a, b, n] = activation of neuron n for input (a, b).
        p: Prime modulus.
        threshold: Minimum fraction of energy in dominant frequency to classify.

    Returns:
        Dict with:
            'dominant_freq': (d_mlp,) dominant frequency per neuron
            'r_squared': (d_mlp,) fraction of energy in dominant freq
            'clusters': dict mapping frequency -> list of neuron indices
    """
    F = dft_matrix(p)
    d_mlp = neuron_activations.shape[2]

    dominant_freq = np.zeros(d_mlp, dtype=int)
    r_squared = np.zeros(d_mlp)

    for n in range(d_mlp):
        act_n = neuron_activations[:, :, n]  # (p, p)
        # 2D DFT
        act_hat = F @ act_n @ F.T
        norms = np.abs(act_hat) ** 2

        # Marginal frequency energy
        freq_energy = np.zeros(p)
        for k in range(p):
            freq_energy[k] = norms[k, :].sum() + norms[:, k].sum() - norms[k, k]

        total_energy = freq_energy.sum()
        if total_energy > 0:
            # Skip DC
            freq_energy_no_dc = freq_energy.copy()
            freq_energy_no_dc[0] = 0
            k_dom = np.argmax(freq_energy_no_dc)
            dominant_freq[n] = k_dom
            r_squared[n] = freq_energy_no_dc[k_dom] / total_energy
        else:
            dominant_freq[n] = 0
            r_squared[n] = 0.0

    # Build clusters
    clusters = {}
    for n in range(d_mlp):
        if r_squared[n] >= threshold:
            k = int(dominant_freq[n])
            clusters.setdefault(k, []).append(n)

    return {
        "dominant_freq": dominant_freq,
        "r_squared": r_squared,
        "clusters": clusters,
    }


def compute_neuron_frequency_spectrum(
    neuron_activations: np.ndarray, p: int
) -> np.ndarray:
    """Compute 2D DFT magnitude for each neuron.

    Args:
        neuron_activations: (p, p, d_mlp) activations on full grid.
        p: Prime modulus.

    Returns:
        (d_mlp, p) array where [n, k] = marginal frequency-k energy for neuron n.
    """
    F = dft_matrix(p)
    d_mlp = neuron_activations.shape[2]
    spectrum = np.zeros((d_mlp, p))

    for n in range(d_mlp):
        act_n = neuron_activations[:, :, n]
        act_hat = F @ act_n @ F.T
        norms = np.abs(act_hat) ** 2
        for k in range(p):
            spectrum[n, k] = norms[k, :].sum() + norms[:, k].sum() - norms[k, k]

    return spectrum
