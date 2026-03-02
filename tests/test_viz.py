"""Tests for new visualization functions."""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

matplotlib.use("Agg")

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
from src.viz.fourier_plots import plot_fourier_spectrum_strip
from src.viz.animation import (
    create_grokking_animation,
    create_fourier_waterfall_animation,
    create_embedding_circle_animation,
    create_loss_landscape_animation,
    create_neuron_grid_animation,
)


# Shared test parameters
P = 7
D_MLP = 16
D_MODEL = 8
KEY_FREQS = np.array([1, 3])


def _make_neuron_class(d_mlp=D_MLP, p=P):
    """Create a mock neuron_class dict."""
    dominant_freq = np.random.randint(0, p, size=d_mlp)
    r_squared = np.random.rand(d_mlp)
    # Assign some neurons to key freq clusters
    clusters = {}
    for n in range(d_mlp):
        if r_squared[n] > 0.3:
            k = int(dominant_freq[n])
            clusters.setdefault(k, []).append(n)
    return {
        "dominant_freq": dominant_freq,
        "r_squared": r_squared,
        "clusters": clusters,
    }


class TestNeuronPlots:
    """Tests for neuron_plots.py functions."""

    def test_activation_grids_returns_figure(self):
        acts = np.random.randn(P, P, D_MLP)
        neuron_class = _make_neuron_class()
        fig = plot_neuron_activation_grids(acts, neuron_class, KEY_FREQS, P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_activation_grids_empty_clusters(self):
        """Should handle no clusters for key frequencies gracefully."""
        acts = np.random.randn(P, P, D_MLP)
        neuron_class = {
            "dominant_freq": np.zeros(D_MLP, dtype=int),
            "r_squared": np.zeros(D_MLP),
            "clusters": {},
        }
        fig = plot_neuron_activation_grids(acts, neuron_class, KEY_FREQS, P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_logit_map_returns_figure(self):
        neuron_logit = np.random.randn(D_MLP, P)
        neuron_class = _make_neuron_class()
        fig = plot_neuron_logit_map(neuron_logit, neuron_class, KEY_FREQS, P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_freq_spectrum_heatmap_returns_figure(self):
        spectrum = np.random.rand(D_MLP, P)
        neuron_class = _make_neuron_class()
        fig = plot_neuron_frequency_spectrum_heatmap(spectrum, neuron_class, KEY_FREQS, P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestLogitPlots:
    """Tests for logit_plots.py functions."""

    def test_heatmap_comparison_returns_figure(self):
        logit_table = np.random.randn(P, P, P)
        restricted = np.random.randn(P, P, P)
        fig = plot_logit_heatmap_comparison(logit_table, restricted, P)
        assert isinstance(fig, plt.Figure)
        # Should have 3 subplots
        assert len(fig.axes) >= 3
        plt.close(fig)

    def test_correct_logit_surface_returns_figure(self):
        logit_table = np.random.randn(P, P, P)
        fig = plot_correct_logit_surface(logit_table, P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_per_sample_loss_no_mask(self):
        logit_table = np.random.randn(P, P, P)
        fig = plot_per_sample_loss_heatmap(logit_table, P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_per_sample_loss_with_mask(self):
        logit_table = np.random.randn(P, P, P)
        train_mask = np.random.rand(P, P) > 0.5
        fig = plot_per_sample_loss_heatmap(logit_table, P, train_mask)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestTrajectoryPlots:
    """Tests for trajectory_plots.py functions."""

    def test_embedding_pca_returns_figure(self):
        snapshots = [np.random.randn(P + 1, D_MODEL) for _ in range(4)]
        epochs = [0, 100, 500, 1000]
        fig = plot_embedding_pca_evolution(snapshots, epochs, P, n_panels=4)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_embedding_pca_empty_snapshots(self):
        fig = plot_embedding_pca_evolution([], [], P)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_embedding_pca_single_snapshot(self):
        snapshots = [np.random.randn(P + 1, D_MODEL)]
        fig = plot_embedding_pca_evolution(snapshots, [0], P, n_panels=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_weight_trajectory_returns_figure(self):
        n_params = 100
        snapshots = [np.random.randn(n_params) for _ in range(5)]
        epochs = [0, 100, 200, 500, 1000]
        fig = plot_weight_trajectory_pca(snapshots, epochs)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_weight_trajectory_single_checkpoint(self):
        """Should handle gracefully with < 2 snapshots."""
        fig = plot_weight_trajectory_pca([np.random.randn(100)], [0])
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_weight_trajectory_with_history(self):
        n_params = 100
        snapshots = [np.random.randn(n_params) for _ in range(5)]
        epochs = [0, 100, 200, 500, 1000]
        history = {
            "test_acc": [0.1, 0.2, 0.3, 0.8, 0.99],
            "eval_epochs": [0, 100, 200, 500, 1000],
        }
        fig = plot_weight_trajectory_pca(snapshots, epochs, history)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestFourierSpectrumStrip:
    """Tests for plot_fourier_spectrum_strip."""

    def test_returns_figure(self):
        n_snaps = 10
        fourier_snaps = {
            "frequency_norms": np.random.rand(n_snaps, P),
            "fourier_epochs": np.arange(0, n_snaps * 100, 100),
        }
        fig = plot_fourier_spectrum_strip(fourier_snaps, KEY_FREQS, P, n_panels=6)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_single_snapshot(self):
        fourier_snaps = {
            "frequency_norms": np.random.rand(1, P),
            "fourier_epochs": np.array([0]),
        }
        fig = plot_fourier_spectrum_strip(fourier_snaps, KEY_FREQS, P, n_panels=1)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestGrokkingAnimation:
    """Tests for create_grokking_animation."""

    def test_returns_animation(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 3
        history = {
            "train_loss": [2.0, 1.0, 0.5],
            "test_loss": [3.0, 2.5, 0.6],
            "eval_epochs": [0, 50, 100],
            "gini": [0.1, 0.5, 0.9],
            "fourier_epochs": [0, 50, 100],
        }
        fourier_snaps = {
            "frequency_norms": np.random.rand(n_snaps, P),
            "fourier_epochs": np.array([0, 50, 100]),
        }
        embeddings = [np.random.randn(P + 1, D_MODEL) for _ in range(n_snaps)]
        snap_epochs = [0, 50, 100]

        anim = create_grokking_animation(
            history, fourier_snaps, embeddings, snap_epochs, P, KEY_FREQS, fps=5
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


class TestFourierWaterfallAnimation:
    """Tests for create_fourier_waterfall_animation."""

    def test_returns_animation(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 5
        fourier_snaps = {
            "frequency_norms": np.random.rand(n_snaps, P),
            "fourier_epochs": np.arange(0, n_snaps * 200, 200),
        }
        history = {
            "train_loss": np.linspace(2.0, 0.1, 10).tolist(),
            "test_loss": np.linspace(3.0, 0.2, 10).tolist(),
            "eval_epochs": np.linspace(0, 800, 10).tolist(),
        }
        anim = create_fourier_waterfall_animation(
            fourier_snaps, history, P, KEY_FREQS, fps=5,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_single_snapshot(self):
        from matplotlib.animation import FuncAnimation
        fourier_snaps = {
            "frequency_norms": np.random.rand(1, P),
            "fourier_epochs": np.array([0]),
        }
        history = {"train_loss": [2.0], "test_loss": [3.0], "eval_epochs": [0]}
        anim = create_fourier_waterfall_animation(
            fourier_snaps, history, P, KEY_FREQS, fps=5,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


class TestEmbeddingCircleAnimation:
    """Tests for create_embedding_circle_animation."""

    def test_returns_animation(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 3
        embeddings = [np.random.randn(P + 1, D_MODEL) for _ in range(n_snaps)]
        snap_epochs = [0, 500, 1000]
        anim = create_embedding_circle_animation(
            embeddings, snap_epochs, P, KEY_FREQS, fps=4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_single_key_freq(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 2
        embeddings = [np.random.randn(P, D_MODEL) for _ in range(n_snaps)]
        snap_epochs = [0, 1000]
        anim = create_embedding_circle_animation(
            embeddings, snap_epochs, P, np.array([2]), fps=4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


class TestLossLandscapeAnimation:
    """Tests for create_loss_landscape_animation."""

    def test_without_history(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 3
        loss_snaps = [np.random.rand(P, P) * 5 for _ in range(n_snaps)]
        snap_epochs = [0, 500, 1000]
        anim = create_loss_landscape_animation(
            loss_snaps, snap_epochs, P, fps=4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_with_history_and_mask(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 3
        loss_snaps = [np.random.rand(P, P) * 5 for _ in range(n_snaps)]
        snap_epochs = [0, 500, 1000]
        train_mask = np.random.rand(P, P) > 0.5
        history = {
            "train_loss": [2.0, 1.0, 0.5],
            "test_loss": [3.0, 2.5, 0.6],
            "eval_epochs": [0, 500, 1000],
        }
        anim = create_loss_landscape_animation(
            loss_snaps, snap_epochs, P, train_mask=train_mask, history=history, fps=4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")


class TestNeuronGridAnimation:
    """Tests for create_neuron_grid_animation."""

    def test_returns_animation(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 3
        neuron_snaps = [np.random.randn(P, P, D_MLP) for _ in range(n_snaps)]
        snap_epochs = [0, 500, 1000]
        neuron_class = _make_neuron_class()
        anim = create_neuron_grid_animation(
            neuron_snaps, snap_epochs, P, KEY_FREQS, neuron_class, fps=4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")

    def test_empty_clusters(self):
        from matplotlib.animation import FuncAnimation
        n_snaps = 2
        neuron_snaps = [np.random.randn(P, P, D_MLP) for _ in range(n_snaps)]
        snap_epochs = [0, 1000]
        neuron_class = {
            "dominant_freq": np.zeros(D_MLP, dtype=int),
            "r_squared": np.zeros(D_MLP),
            "clusters": {},
        }
        anim = create_neuron_grid_animation(
            neuron_snaps, snap_epochs, P, KEY_FREQS, neuron_class, fps=4,
        )
        assert isinstance(anim, FuncAnimation)
        plt.close("all")
