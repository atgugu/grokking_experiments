"""Fourier analysis for mechanistic interpretability of grokking.

Implements 2D Discrete Fourier Transform on the logit table to identify
key frequencies used by the network's trigonometric algorithm.

Reference: Nanda et al. (2023), "Progress measures for grokking via
mechanistic interpretability", ICLR 2023.
"""

import numpy as np
import torch


def dft_matrix(p: int) -> np.ndarray:
    """Construct the p x p DFT matrix.

    F[k, n] = exp(-2*pi*i*k*n / p) / sqrt(p)

    The matrix is unitary: F @ F.conj().T = I.

    Returns:
        (p, p) complex128 DFT matrix.
    """
    n = np.arange(p)
    k = np.arange(p)
    F = np.exp(-2j * np.pi * np.outer(k, n) / p) / np.sqrt(p)
    return F


def fourier_transform_2d(L: np.ndarray, p: int) -> np.ndarray:
    """Compute 2D DFT of a logit slice L(a, b) for a single output class.

    L_hat[k1, k2] = sum_{a,b} L[a,b] * F[k1,a] * F[k2,b]

    Equivalently: L_hat = F @ L @ F.T

    Args:
        L: (p, p) real or complex array (logit table for one output class).
        p: Prime modulus.

    Returns:
        (p, p) complex array of Fourier coefficients.
    """
    F = dft_matrix(p)
    return F @ L @ F.T


def compute_fourier_component_norms(logit_table: np.ndarray, p: int) -> dict:
    """Compute 2D Fourier component norms from the full logit table.

    For each output class c, compute the 2D DFT of L[:,:,c], then
    aggregate norms across classes.

    Args:
        logit_table: (p, p, p) array where [a, b, c] = logit for class c given (a, b).
        p: Prime modulus.

    Returns:
        Dict with:
            'component_norms': (p, p) total Fourier energy per (k1, k2) frequency pair
            'frequency_norms': (p,) marginal norm per frequency k (summing over k1, k2 where k1==k or k2==k)
            'per_class_norms': (p, p, p) norms per output class per freq pair
    """
    # 2D FFT along the (a, b) axes — equivalent to F @ L[:,:,c] @ F.T
    # with DFT matrix F[k,n] = exp(-2πi·k·n/p)/√p, but O(p² log p) per class.
    L_hat = np.fft.fft2(logit_table, axes=(0, 1), norm='ortho')
    per_class_norms = np.abs(L_hat) ** 2  # (p, p, p)
    component_norms = per_class_norms.sum(axis=2)  # (p, p)

    # Marginal frequency norms: vectorized
    frequency_norms = component_norms.sum(axis=1) + component_norms.sum(axis=0) - np.diag(component_norms)

    return {
        "component_norms": component_norms,
        "frequency_norms": frequency_norms,
        "per_class_norms": per_class_norms,
    }


def identify_key_frequencies(frequency_norms: np.ndarray, n_top: int = 5) -> np.ndarray:
    """Identify the top-N non-DC frequencies by energy.

    Excludes frequency 0 (DC component).

    Args:
        frequency_norms: (p,) array of marginal frequency energies.
        n_top: Number of top frequencies to return.

    Returns:
        Array of n_top frequency indices, sorted by energy descending.
    """
    # Exclude DC (k=0)
    norms_no_dc = frequency_norms.copy()
    norms_no_dc[0] = 0.0

    # Also handle symmetry: freq k and p-k are conjugate pairs, same information
    # Just return the top n_top distinct frequencies
    top_indices = np.argsort(norms_no_dc)[::-1][:n_top]
    return top_indices


def compute_gini_coefficient(frequency_norms: np.ndarray) -> float:
    """Compute Gini coefficient of frequency energy distribution.

    Gini = 0 means all frequencies have equal energy (uniform).
    Gini = 1 means all energy is in a single frequency (maximally sparse).

    A high Gini indicates the network has learned a sparse Fourier representation,
    which is the hallmark of the trig algorithm discovered by grokking.

    Args:
        frequency_norms: (p,) array of non-negative frequency energies.

    Returns:
        Gini coefficient in [0, 1].
    """
    norms = np.sort(frequency_norms)
    n = len(norms)
    if norms.sum() == 0:
        return 0.0
    cumulative = np.cumsum(norms)
    # Gini = 1 - 2 * area under Lorenz curve
    gini = 1.0 - 2.0 * cumulative.sum() / (n * norms.sum()) + 1.0 / n
    return float(gini)


def compute_restricted_logits(
    logit_table: np.ndarray, p: int, key_freqs: np.ndarray
) -> np.ndarray:
    """Compute logits restricted to DC + key frequency components.

    Zeroes out all Fourier components except those involving
    frequency 0 (DC) or any key frequency.

    Args:
        logit_table: (p, p, p) logit table.
        p: Prime modulus.
        key_freqs: Array of key frequency indices.

    Returns:
        (p, p, p) restricted logit table (inverse DFT of masked components).
    """
    # Build mask: 1 where (k1 in allowed) and (k2 in allowed), else 0
    allowed_arr = np.zeros(p)
    allowed_arr[0] = 1.0
    allowed_arr[key_freqs] = 1.0
    mask = np.outer(allowed_arr, allowed_arr)

    # Forward 2D FFT, mask in frequency domain, inverse 2D FFT
    L_hat = np.fft.fft2(logit_table, axes=(0, 1), norm='ortho')
    L_hat_masked = L_hat * mask[:, :, np.newaxis]
    restricted = np.real(np.fft.ifft2(L_hat_masked, axes=(0, 1), norm='ortho'))

    return restricted


def compute_excluded_logits(
    logit_table: np.ndarray, p: int, key_freqs: np.ndarray
) -> np.ndarray:
    """Compute logits excluding DC + key frequency components.

    This is the complement: L_excluded = L - L_restricted.

    Args:
        logit_table: (p, p, p) logit table.
        p: Prime modulus.
        key_freqs: Array of key frequency indices.

    Returns:
        (p, p, p) excluded logit table.
    """
    restricted = compute_restricted_logits(logit_table, p, key_freqs)
    return logit_table - restricted


def fourier_embed_analysis(W_E: np.ndarray, p: int) -> dict:
    """Analyze embedding matrix via 1D Fourier transform.

    Computes the DFT of each embedding dimension across the p integer tokens
    (excluding the equals token).

    Args:
        W_E: (p+1, d_model) embedding matrix (first p rows = integer tokens).
        p: Prime modulus.

    Returns:
        Dict with:
            'fourier_magnitudes': (p, d_model) DFT magnitudes
            'frequency_energy': (p,) energy per frequency across all dimensions
    """
    # Take only the integer token embeddings
    E = W_E[:p, :]  # (p, d_model)

    # 1D FFT along token dimension for each embedding dim
    E_hat = np.fft.fft(E, axis=0, norm='ortho')  # (p, d_model) complex
    magnitudes = np.abs(E_hat)

    # Energy per frequency: sum across dimensions
    frequency_energy = (magnitudes ** 2).sum(axis=1)

    return {
        "fourier_magnitudes": magnitudes,
        "frequency_energy": frequency_energy,
    }
