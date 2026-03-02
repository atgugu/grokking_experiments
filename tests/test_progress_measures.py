"""Tests for progress measures."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import GrokkingTransformer
from src.data.modular_arithmetic import ModularArithmeticEnvironment
from src.analysis.progress_measures import compute_all_progress_measures


class TestProgressMeasures:
    """Test the progress measure computation."""

    @pytest.fixture
    def setup(self):
        p = 7
        env = ModularArithmeticEnvironment(p=p, seed=42)
        model = GrokkingTransformer(p=p, d_model=32, n_heads=2, d_mlp=64, n_layers=1)
        train_data = env.get_train_dataset()
        return model, train_data, p

    def test_return_keys(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"),
        )
        assert "restricted_loss" in result
        assert "excluded_loss" in result
        assert "gini" in result
        assert "weight_norm" in result
        assert "key_frequencies" in result
        assert "frequency_norms" in result

    def test_restricted_loss_is_finite(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"),
        )
        assert np.isfinite(result["restricted_loss"])

    def test_excluded_loss_is_finite(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"),
        )
        assert np.isfinite(result["excluded_loss"])

    def test_gini_in_bounds(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"),
        )
        assert 0.0 <= result["gini"] <= 1.0

    def test_weight_norm_positive(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"),
        )
        assert result["weight_norm"] > 0

    def test_frequency_norms_shape(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"),
        )
        assert len(result["frequency_norms"]) == p

    def test_key_frequencies_count(self, setup):
        model, train_data, p = setup
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, device=torch.device("cpu"), n_top=3,
        )
        assert len(result["key_frequencies"]) == 3

    def test_with_specified_key_freqs(self, setup):
        model, train_data, p = setup
        key_freqs = np.array([1, 2, 3])
        result = compute_all_progress_measures(
            model, train_data.inputs, train_data.target_tensor,
            p=p, key_frequencies=key_freqs, device=torch.device("cpu"),
        )
        np.testing.assert_array_equal(result["key_frequencies"], key_freqs)
