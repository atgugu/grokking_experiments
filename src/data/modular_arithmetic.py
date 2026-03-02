"""Modular arithmetic data generation for grokking experiments.

Task: a + b mod p (default p=113)
Input: 3 tokens [a, b, =] where = is token index p
Output: (a + b) mod p, an integer in [0, p)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class ModularArithmeticEnvironment:
    """Generates all (a, b) pairs for modular addition and splits train/test.

    Attributes:
        p: Prime modulus.
        operation: Currently only 'addition' supported.
        train_fraction: Fraction of p^2 pairs used for training.
        vocab_size: p + 1 (p integers + equals token).
        num_classes: p (output is in [0, p)).
        equals_token: Index p (the equals sign token).
    """

    def __init__(
        self,
        p: int = 113,
        operation: str = "addition",
        train_fraction: float = 0.3,
        seed: int = 42,
    ):
        self.p = p
        self.operation = operation
        self.train_fraction = train_fraction
        self.vocab_size = p + 1  # p integers + equals token
        self.num_classes = p
        self.equals_token = p

        # Generate all p^2 pairs
        a_all, b_all = np.meshgrid(np.arange(p), np.arange(p), indexing="ij")
        self.a_all = a_all.flatten()
        self.b_all = b_all.flatten()
        self.targets_all = self._compute_targets(self.a_all, self.b_all)

        # Random train/test split
        rng = np.random.RandomState(seed)
        n_total = p * p
        n_train = int(n_total * train_fraction)
        perm = rng.permutation(n_total)
        self.train_indices = np.sort(perm[:n_train])
        self.test_indices = np.sort(perm[n_train:])

    def _compute_targets(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute operation result."""
        if self.operation == "addition":
            return (a + b) % self.p
        elif self.operation == "subtraction":
            return (a - b) % self.p
        elif self.operation == "multiplication":
            return (a * b) % self.p
        else:
            raise ValueError(f"Unknown operation: {self.operation}")

    def get_train_dataset(self) -> "ModularArithmeticDataset":
        """Return training dataset."""
        return ModularArithmeticDataset(
            self.a_all[self.train_indices],
            self.b_all[self.train_indices],
            self.targets_all[self.train_indices],
            self.equals_token,
        )

    def get_test_dataset(self) -> "ModularArithmeticDataset":
        """Return test dataset."""
        return ModularArithmeticDataset(
            self.a_all[self.test_indices],
            self.b_all[self.test_indices],
            self.targets_all[self.test_indices],
            self.equals_token,
        )

    def get_full_dataset(self) -> "ModularArithmeticDataset":
        """Return all p^2 pairs in canonical order (for Fourier analysis)."""
        return ModularArithmeticDataset(
            self.a_all,
            self.b_all,
            self.targets_all,
            self.equals_token,
        )


class ModularArithmeticDataset(Dataset):
    """PyTorch dataset for modular arithmetic.

    Each sample is ([a, b, =], target) where:
    - Input is a 3-token sequence: [a_token, b_token, equals_token]
    - Target is (a op b) mod p
    """

    def __init__(
        self,
        a_vals: np.ndarray,
        b_vals: np.ndarray,
        targets: np.ndarray,
        equals_token: int,
    ):
        self.a_vals = a_vals
        self.b_vals = b_vals
        self.targets = targets
        self.equals_token = equals_token
        self._inputs = torch.stack([
            torch.tensor(a_vals, dtype=torch.long),
            torch.tensor(b_vals, dtype=torch.long),
            torch.full((len(a_vals),), equals_token, dtype=torch.long),
        ], dim=1)  # (N, 3)
        self._targets = torch.tensor(targets, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._inputs[idx], self._targets[idx]

    @property
    def inputs(self) -> torch.Tensor:
        """All inputs as (N, 3) tensor."""
        return self._inputs

    @property
    def target_tensor(self) -> torch.Tensor:
        """All targets as (N,) tensor."""
        return self._targets
