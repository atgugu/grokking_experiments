"""Visualization modules for grokking experiments."""

from .training_curves import plot_grokking_curves, plot_progress_measures, plot_phase_boundaries
from .fourier_plots import (
    plot_frequency_spectrum,
    plot_fourier_heatmap,
    plot_fourier_evolution,
    plot_embedding_fourier,
    plot_fourier_spectrum_strip,
)
from .attention_plots import plot_attention_patterns, plot_attention_by_input
from .embedding_geometry import plot_embedding_circles, plot_neuron_frequency_clusters
from .weight_heatmaps import plot_weight_heatmap, plot_weight_evolution
from .neuron_plots import (
    plot_neuron_activation_grids,
    plot_neuron_logit_map,
    plot_neuron_frequency_spectrum_heatmap,
)
from .logit_plots import (
    plot_logit_heatmap_comparison,
    plot_correct_logit_surface,
    plot_per_sample_loss_heatmap,
)
from .trajectory_plots import (
    plot_embedding_pca_evolution,
    plot_weight_trajectory_pca,
)
from .animation import (
    create_grokking_animation,
    create_fourier_waterfall_animation,
    create_embedding_circle_animation,
    create_loss_landscape_animation,
    create_neuron_grid_animation,
)
