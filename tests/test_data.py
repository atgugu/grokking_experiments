"""Tests for modular arithmetic data generation."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.modular_arithmetic import ModularArithmeticEnvironment, ModularArithmeticDataset
from src.utils import run_id, _OP_SUFFIXES


class TestModularArithmeticEnvironment:
    """Test the data environment."""

    def test_default_p113(self):
        env = ModularArithmeticEnvironment(p=113, seed=42)
        assert env.p == 113
        assert env.vocab_size == 114  # 113 + 1 (equals token)
        assert env.num_classes == 113
        assert env.equals_token == 113

    def test_total_pairs(self):
        env = ModularArithmeticEnvironment(p=113, seed=42)
        assert len(env.a_all) == 113 * 113
        assert len(env.b_all) == 113 * 113
        assert len(env.targets_all) == 113 * 113

    def test_train_test_split_sizes(self):
        env = ModularArithmeticEnvironment(p=113, train_fraction=0.3, seed=42)
        n_total = 113 * 113
        n_train = int(n_total * 0.3)
        assert len(env.train_indices) == n_train
        assert len(env.test_indices) == n_total - n_train

    def test_train_test_no_overlap(self):
        env = ModularArithmeticEnvironment(p=113, seed=42)
        train_set = set(env.train_indices.tolist())
        test_set = set(env.test_indices.tolist())
        assert len(train_set & test_set) == 0

    def test_train_test_cover_all(self):
        env = ModularArithmeticEnvironment(p=113, seed=42)
        all_indices = set(range(113 * 113))
        covered = set(env.train_indices.tolist()) | set(env.test_indices.tolist())
        assert covered == all_indices

    def test_addition_correctness(self):
        env = ModularArithmeticEnvironment(p=7, operation="addition", seed=0)
        for i in range(len(env.a_all)):
            a, b = env.a_all[i], env.b_all[i]
            assert env.targets_all[i] == (a + b) % 7

    def test_multiplication_correctness(self):
        env = ModularArithmeticEnvironment(p=7, operation="multiplication", seed=0)
        for i in range(len(env.a_all)):
            a, b = env.a_all[i], env.b_all[i]
            assert env.targets_all[i] == (a * b) % 7

    def test_subtraction_correctness(self):
        env = ModularArithmeticEnvironment(p=7, operation="subtraction", seed=0)
        for i in range(len(env.a_all)):
            a, b = env.a_all[i], env.b_all[i]
            assert env.targets_all[i] == (a - b) % 7

    def test_x2_plus_y2_correctness(self):
        env = ModularArithmeticEnvironment(p=7, operation="x2_plus_y2", seed=0)
        for i in range(len(env.a_all)):
            a, b = env.a_all[i], env.b_all[i]
            assert env.targets_all[i] == (a * a + b * b) % 7

    def test_x3_plus_xy_correctness(self):
        env = ModularArithmeticEnvironment(p=7, operation="x3_plus_xy", seed=0)
        for i in range(len(env.a_all)):
            a, b = env.a_all[i], env.b_all[i]
            assert env.targets_all[i] == (a * a * a + a * b) % 7

    def test_x2_plus_y2_symmetry(self):
        """x2_plus_y2 should be symmetric: f(a,b) == f(b,a)."""
        env = ModularArithmeticEnvironment(p=7, operation="x2_plus_y2", seed=0)
        p = env.p
        for a in range(p):
            for b in range(p):
                idx_ab = a * p + b
                idx_ba = b * p + a
                assert env.targets_all[idx_ab] == env.targets_all[idx_ba]

    def test_x3_plus_xy_asymmetry(self):
        """x3_plus_xy should be asymmetric: f(a,b) != f(b,a) for some a,b."""
        env = ModularArithmeticEnvironment(p=7, operation="x3_plus_xy", seed=0)
        p = env.p
        found_asymmetric = False
        for a in range(p):
            for b in range(p):
                idx_ab = a * p + b
                idx_ba = b * p + a
                if env.targets_all[idx_ab] != env.targets_all[idx_ba]:
                    found_asymmetric = True
                    break
            if found_asymmetric:
                break
        assert found_asymmetric, "Expected x3_plus_xy to be asymmetric"

    def test_unknown_operation_raises(self):
        with pytest.raises(ValueError, match="Unknown operation"):
            ModularArithmeticEnvironment(p=7, operation="division", seed=0)

    def test_deterministic_split(self):
        env1 = ModularArithmeticEnvironment(p=113, seed=42)
        env2 = ModularArithmeticEnvironment(p=113, seed=42)
        np.testing.assert_array_equal(env1.train_indices, env2.train_indices)
        np.testing.assert_array_equal(env1.test_indices, env2.test_indices)

    def test_different_seed_different_split(self):
        env1 = ModularArithmeticEnvironment(p=113, seed=42)
        env2 = ModularArithmeticEnvironment(p=113, seed=99)
        assert not np.array_equal(env1.train_indices, env2.train_indices)

    def test_small_prime(self):
        env = ModularArithmeticEnvironment(p=5, seed=0)
        assert len(env.a_all) == 25
        assert env.vocab_size == 6
        assert env.num_classes == 5

    def test_get_train_dataset(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_train_dataset()
        assert len(ds) == len(env.train_indices)

    def test_get_test_dataset(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_test_dataset()
        assert len(ds) == len(env.test_indices)

    def test_get_full_dataset(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_full_dataset()
        assert len(ds) == 49  # 7*7


class TestModularArithmeticDataset:
    """Test the dataset."""

    def test_tensor_shapes(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_train_dataset()
        inputs, target = ds[0]
        assert inputs.shape == (3,)  # [a, b, =]
        assert target.shape == ()

    def test_equals_token(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_train_dataset()
        for i in range(len(ds)):
            inputs, _ = ds[i]
            assert inputs[2].item() == 7  # equals token

    def test_input_range(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_train_dataset()
        for i in range(len(ds)):
            inputs, target = ds[i]
            assert 0 <= inputs[0].item() < 7
            assert 0 <= inputs[1].item() < 7
            assert 0 <= target.item() < 7

    def test_inputs_property(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_train_dataset()
        assert ds.inputs.shape == (len(ds), 3)
        assert ds.inputs.dtype == torch.long

    def test_target_tensor_property(self):
        env = ModularArithmeticEnvironment(p=7, seed=0)
        ds = env.get_train_dataset()
        assert ds.target_tensor.shape == (len(ds),)
        assert ds.target_tensor.dtype == torch.long

    def test_full_dataset_canonical_order(self):
        """Full dataset should have a_vals in row-major order: 0,0,...,0,1,1,..."""
        env = ModularArithmeticEnvironment(p=5, seed=0)
        ds = env.get_full_dataset()
        # First p entries should all have a=0
        for i in range(5):
            assert ds.a_vals[i] == 0
            assert ds.b_vals[i] == i


class TestRunIdOperationSuffixes:
    """Test that run_id produces correct suffixes for each operation."""

    BASE_CONFIG = {
        "p": 113, "d_model": 128, "n_heads": 4, "d_mlp": 512,
        "n_layers": 1, "weight_decay": 1.0, "seed": 42,
    }

    def test_addition_no_suffix(self):
        config = {**self.BASE_CONFIG, "operation": "addition"}
        rid = run_id(config)
        assert rid == "p113_d128_h4_mlp512_L1_wd1.0_s42"

    def test_subtraction_suffix(self):
        config = {**self.BASE_CONFIG, "operation": "subtraction"}
        rid = run_id(config)
        assert rid.endswith("_sub")

    def test_multiplication_suffix(self):
        config = {**self.BASE_CONFIG, "operation": "multiplication"}
        rid = run_id(config)
        assert rid.endswith("_mul")

    def test_x2_plus_y2_suffix(self):
        config = {**self.BASE_CONFIG, "operation": "x2_plus_y2"}
        rid = run_id(config)
        assert rid.endswith("_x2y2")

    def test_x3_plus_xy_suffix(self):
        config = {**self.BASE_CONFIG, "operation": "x3_plus_xy"}
        rid = run_id(config)
        assert rid.endswith("_x3xy")

    def test_default_operation_no_suffix(self):
        """No operation key should default to addition (no suffix)."""
        rid = run_id(self.BASE_CONFIG)
        assert rid == "p113_d128_h4_mlp512_L1_wd1.0_s42"
