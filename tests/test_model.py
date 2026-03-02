"""Tests for the GrokkingTransformer model."""

import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import GrokkingTransformer, TransformerBlock


class TestTransformerBlock:
    """Test the transformer block."""

    def test_output_shape(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256)
        x = torch.randn(8, 3, 64)
        out = block(x)
        assert out.shape == (8, 3, 64)

    def test_no_layernorm_by_default(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256)
        assert block.use_layernorm is False
        assert not hasattr(block, "norm1") or not block.use_layernorm

    def test_with_layernorm(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256, use_layernorm=True)
        assert block.use_layernorm is True

    def test_relu_activation(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256, activation="relu")
        assert isinstance(block.mlp[1], torch.nn.ReLU)

    def test_gelu_activation(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256, activation="gelu")
        assert isinstance(block.mlp[1], torch.nn.GELU)

    def test_attention_weights_cached(self):
        block = TransformerBlock(d_model=64, n_heads=4, d_mlp=256)
        x = torch.randn(8, 3, 64)
        _ = block(x)
        assert block._attn_weights is not None
        assert block._attn_weights.shape == (8, 3, 3)


class TestGrokkingTransformer:
    """Test the full model."""

    def test_output_shape(self):
        model = GrokkingTransformer(p=113, d_model=128, n_heads=4, d_mlp=512)
        x = torch.randint(0, 114, (16, 3))
        out = model(x)
        assert out.shape == (16, 113)

    def test_output_shape_small(self):
        model = GrokkingTransformer(p=7, d_model=32, n_heads=2, d_mlp=64)
        x = torch.randint(0, 8, (4, 3))
        out = model(x)
        assert out.shape == (4, 7)

    def test_param_count(self):
        model = GrokkingTransformer(p=113, d_model=128, n_heads=4, d_mlp=512, n_layers=1)
        n_params = model.count_parameters()
        assert n_params > 0
        # Rough check: W_E(114*128) + W_pos(3*128) + attn + MLP + W_U
        assert n_params > 100000

    def test_no_layernorm_default(self):
        model = GrokkingTransformer(p=113, d_model=128, n_heads=4, d_mlp=512)
        for block in model.blocks:
            assert block.use_layernorm is False

    def test_separate_embeddings(self):
        model = GrokkingTransformer(p=113, d_model=128, n_heads=4, d_mlp=512, tie_embeddings=False)
        assert model.W_U is not None
        assert model._tie_embeddings is False

    def test_tied_embeddings(self):
        model = GrokkingTransformer(p=113, d_model=128, n_heads=4, d_mlp=512, tie_embeddings=True)
        assert model.W_U is None
        assert model._tie_embeddings is True

    def test_prediction_at_equals_position(self):
        """Model should use position 2 (=) for prediction, not mean-pool."""
        model = GrokkingTransformer(p=7, d_model=32, n_heads=2, d_mlp=64)
        x = torch.randint(0, 8, (4, 3))
        out = model(x)
        # Output should be based on position 2 only
        assert out.shape == (4, 7)

    def test_get_logit_table(self):
        model = GrokkingTransformer(p=7, d_model=32, n_heads=2, d_mlp=64)
        table = model.get_logit_table(torch.device("cpu"))
        assert table.shape == (7, 7, 7)

    def test_get_attention_patterns(self):
        model = GrokkingTransformer(p=7, d_model=32, n_heads=2, d_mlp=64)
        x = torch.randint(0, 8, (4, 3))
        _ = model(x)
        patterns = model.get_attention_patterns()
        assert len(patterns) == 1  # 1 layer
        assert patterns[0].shape == (4, 3, 3)

    def test_vocab_size(self):
        model = GrokkingTransformer(p=113)
        assert model.vocab_size == 114
        assert model.num_classes == 113

    def test_single_layer(self):
        model = GrokkingTransformer(p=7, d_model=32, n_heads=2, d_mlp=64, n_layers=1)
        assert len(model.blocks) == 1

    def test_multi_layer(self):
        model = GrokkingTransformer(p=7, d_model=32, n_heads=2, d_mlp=64, n_layers=3)
        assert len(model.blocks) == 3
