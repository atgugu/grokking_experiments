"""Progress measures for tracking grokking via Fourier analysis.

Implements the 4 progress measures from Nanda et al. (2023):
1. Restricted loss: cross-entropy using only DC + key frequency components
2. Excluded loss: cross-entropy using only non-key frequency components
3. Gini coefficient: sparsity of frequency energy distribution
4. Weight norm: L2 norm of all model parameters

The key insight: during grokking, restricted_loss decreases (useful algorithm
forms) while excluded_loss increases (noise is regularized away).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fourier import (
    compute_fourier_component_norms,
    compute_gini_coefficient,
    compute_restricted_logits,
    compute_excluded_logits,
    identify_key_frequencies,
)


@torch.no_grad()
def compute_all_progress_measures(
    model: nn.Module,
    train_inputs: torch.Tensor,
    train_targets: torch.Tensor,
    p: int,
    key_frequencies: np.ndarray | None = None,
    device: torch.device = torch.device("cpu"),
    n_top: int = 5,
) -> dict:
    """Compute all progress measures for the current model state.

    Args:
        model: GrokkingTransformer model.
        train_inputs: (N, 3) training input tokens.
        train_targets: (N,) training targets.
        p: Prime modulus.
        key_frequencies: Pre-computed key frequencies. If None, auto-detected.
        device: Compute device.
        n_top: Number of top frequencies if auto-detecting.

    Returns:
        Dict with:
            'restricted_loss': CE loss using only key frequency logits
            'excluded_loss': CE loss using only non-key frequency logits
            'gini': Gini coefficient of frequency norms
            'weight_norm': L2 norm of all parameters
            'key_frequencies': Array of key frequency indices
            'frequency_norms': (p,) array of marginal frequency energies
    """
    model.eval()

    # Get full logit table
    logit_table = model.get_logit_table(device).cpu().numpy()

    # Fourier analysis
    result = compute_fourier_component_norms(logit_table, p)
    freq_norms = result["frequency_norms"]

    if key_frequencies is None:
        key_frequencies = identify_key_frequencies(freq_norms, n_top=n_top)

    gini = compute_gini_coefficient(freq_norms)

    # Restricted and excluded logits
    restricted = compute_restricted_logits(logit_table, p, key_frequencies)
    excluded = compute_excluded_logits(logit_table, p, key_frequencies)

    # Compute losses on training data using restricted/excluded logits
    # Map training (a, b) pairs to their logits in the table
    a_vals = train_inputs[:, 0].cpu().numpy()
    b_vals = train_inputs[:, 1].cpu().numpy()
    targets_np = train_targets.cpu().numpy()

    # Extract logits for training pairs
    restricted_logits = torch.tensor(
        restricted[a_vals, b_vals, :], dtype=torch.float32, device=device,
    )
    excluded_logits = torch.tensor(
        excluded[a_vals, b_vals, :], dtype=torch.float32, device=device,
    )

    ce = nn.CrossEntropyLoss()
    restricted_loss = ce(restricted_logits, train_targets.to(device)).item()
    excluded_loss = ce(excluded_logits, train_targets.to(device)).item()

    # Weight norm
    w_norm = sum(p_param.data.norm() ** 2 for p_param in model.parameters()).sqrt().item()

    model.train()
    return {
        "restricted_loss": restricted_loss,
        "excluded_loss": excluded_loss,
        "gini": gini,
        "weight_norm": w_norm,
        "key_frequencies": key_frequencies,
        "frequency_norms": freq_norms,
    }
