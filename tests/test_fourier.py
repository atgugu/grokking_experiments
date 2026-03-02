"""Tests for Fourier analysis functions."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fourier import (
    dft_matrix,
    fourier_transform_2d,
    compute_fourier_component_norms,
    identify_key_frequencies,
    compute_gini_coefficient,
    compute_restricted_logits,
    compute_excluded_logits,
    fourier_embed_analysis,
)


class TestDFTMatrix:
    """Test DFT matrix construction."""

    def test_shape(self):
        F = dft_matrix(7)
        assert F.shape == (7, 7)

    def test_unitarity(self):
        """F @ F^H should equal I."""
        for p in [5, 7, 11, 13]:
            F = dft_matrix(p)
            product = F @ F.conj().T
            np.testing.assert_allclose(product, np.eye(p), atol=1e-10)

    def test_inverse_roundtrip(self):
        """F^H @ F @ x should equal x."""
        p = 11
        F = dft_matrix(p)
        x = np.random.randn(p)
        x_hat = F @ x
        x_recovered = F.conj().T @ x_hat
        np.testing.assert_allclose(x_recovered.real, x, atol=1e-10)

    def test_known_signal(self):
        """DFT of a pure sinusoid should have energy at that frequency."""
        p = 13
        k_target = 3
        F = dft_matrix(p)
        # Create signal: cos(2*pi*k*n/p)
        n = np.arange(p)
        signal = np.cos(2 * np.pi * k_target * n / p)
        spectrum = F @ signal
        magnitudes = np.abs(spectrum) ** 2
        # Energy should be concentrated at k_target and p-k_target (conjugate)
        peak_freq = np.argmax(magnitudes)
        assert peak_freq in [k_target, p - k_target]


class TestFourierTransform2D:
    """Test 2D Fourier transform."""

    def test_shape(self):
        p = 7
        L = np.random.randn(p, p)
        L_hat = fourier_transform_2d(L, p)
        assert L_hat.shape == (p, p)

    def test_inverse_roundtrip(self):
        """2D DFT followed by inverse should recover original."""
        p = 7
        L = np.random.randn(p, p)
        F = dft_matrix(p)
        L_hat = F @ L @ F.T
        L_recovered = F.conj().T @ L_hat @ F.conj()
        np.testing.assert_allclose(L_recovered.real, L, atol=1e-10)


class TestComponentNorms:
    """Test Fourier component norm computation."""

    def test_shape(self):
        p = 7
        logit_table = np.random.randn(p, p, p)
        result = compute_fourier_component_norms(logit_table, p)
        assert result["component_norms"].shape == (p, p)
        assert result["frequency_norms"].shape == (p,)
        assert result["per_class_norms"].shape == (p, p, p)

    def test_nonnegative(self):
        p = 7
        logit_table = np.random.randn(p, p, p)
        result = compute_fourier_component_norms(logit_table, p)
        assert np.all(result["component_norms"] >= 0)
        assert np.all(result["frequency_norms"] >= 0)


class TestGiniCoefficient:
    """Test Gini coefficient computation."""

    def test_uniform_is_zero(self):
        """All equal values should give Gini close to 0."""
        norms = np.ones(100)
        gini = compute_gini_coefficient(norms)
        assert abs(gini) < 0.02

    def test_sparse_is_high(self):
        """One dominant value should give high Gini."""
        norms = np.zeros(100)
        norms[42] = 1000.0
        gini = compute_gini_coefficient(norms)
        assert gini > 0.95

    def test_bounds(self):
        """Gini should be in [0, 1]."""
        for _ in range(10):
            norms = np.abs(np.random.randn(50))
            gini = compute_gini_coefficient(norms)
            assert 0.0 <= gini <= 1.0

    def test_zero_norms(self):
        norms = np.zeros(10)
        gini = compute_gini_coefficient(norms)
        assert gini == 0.0


class TestKeyFrequencies:
    """Test key frequency identification."""

    def test_excludes_dc(self):
        norms = np.zeros(113)
        norms[0] = 1e6  # DC should be excluded
        norms[14] = 100
        norms[35] = 80
        top = identify_key_frequencies(norms, n_top=2)
        assert 0 not in top
        assert 14 in top

    def test_correct_count(self):
        norms = np.random.rand(113)
        top = identify_key_frequencies(norms, n_top=5)
        assert len(top) == 5


class TestRestrictedExcluded:
    """Test restricted and excluded logit decomposition."""

    def test_complementarity(self):
        """restricted + excluded should equal original."""
        p = 7
        logit_table = np.random.randn(p, p, p)
        key_freqs = np.array([1, 3])
        restricted = compute_restricted_logits(logit_table, p, key_freqs)
        excluded = compute_excluded_logits(logit_table, p, key_freqs)
        np.testing.assert_allclose(restricted + excluded, logit_table, atol=1e-10)

    def test_restricted_shape(self):
        p = 7
        logit_table = np.random.randn(p, p, p)
        key_freqs = np.array([1, 3])
        restricted = compute_restricted_logits(logit_table, p, key_freqs)
        assert restricted.shape == (p, p, p)

    def test_excluded_shape(self):
        p = 7
        logit_table = np.random.randn(p, p, p)
        key_freqs = np.array([1, 3])
        excluded = compute_excluded_logits(logit_table, p, key_freqs)
        assert excluded.shape == (p, p, p)


class TestEmbedFourier:
    """Test embedding Fourier analysis."""

    def test_output_keys(self):
        p = 7
        W_E = np.random.randn(p + 1, 32)
        result = fourier_embed_analysis(W_E, p)
        assert "fourier_magnitudes" in result
        assert "frequency_energy" in result

    def test_shapes(self):
        p = 7
        d_model = 32
        W_E = np.random.randn(p + 1, d_model)
        result = fourier_embed_analysis(W_E, p)
        assert result["fourier_magnitudes"].shape == (p, d_model)
        assert result["frequency_energy"].shape == (p,)
