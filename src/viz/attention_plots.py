"""Attention pattern visualizations."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")


def plot_attention_patterns(
    attn_weights: torch.Tensor,
    token_labels: list[str] | None = None,
) -> plt.Figure:
    """Plot attention pattern heatmaps, averaged over batch.

    Since nn.MultiheadAttention with need_weights=True returns averaged weights,
    this shows the (3, 3) attention pattern averaged over all (a, b) inputs.

    Args:
        attn_weights: (batch, seq, seq) attention weights.
        token_labels: Labels for positions (default: ['a', 'b', '=']).
    """
    if token_labels is None:
        token_labels = ["a", "b", "="]

    # Average over batch
    avg_attn = attn_weights.mean(dim=0).cpu().numpy()  # (3, 3)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(avg_attn, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels)
    ax.set_yticks(range(len(token_labels)))
    ax.set_yticklabels(token_labels)
    ax.set_xlabel("Key")
    ax.set_ylabel("Query")
    ax.set_title("Average Attention Pattern")

    # Annotate cells
    for i in range(len(token_labels)):
        for j in range(len(token_labels)):
            ax.text(j, i, f"{avg_attn[i, j]:.2f}",
                    ha="center", va="center", fontsize=11,
                    color="white" if avg_attn[i, j] > 0.5 else "black")

    fig.tight_layout()
    return fig


def plot_attention_by_input(
    model,
    inputs: torch.Tensor,
    a_vals: np.ndarray,
    b_vals: np.ndarray,
    p: int,
    n_examples: int = 9,
) -> plt.Figure:
    """Plot attention patterns for specific input examples.

    Args:
        model: GrokkingTransformer model.
        inputs: (N, 3) input tensor.
        a_vals: (N,) a values.
        b_vals: (N,) b values.
        p: Prime modulus.
        n_examples: Number of examples to show.
    """
    token_labels = ["a", "b", "="]
    n_cols = 3
    n_rows = (n_examples + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).flatten()

    # Select evenly spaced examples
    indices = np.linspace(0, len(inputs) - 1, n_examples, dtype=int)

    model.eval()
    with torch.no_grad():
        _ = model(inputs)
        attn_patterns = model.get_attention_patterns()

    if not attn_patterns:
        plt.close(fig)
        return plt.figure()

    attn = attn_patterns[0]  # (batch, seq, seq) for first block

    for idx, ax in enumerate(axes):
        if idx >= n_examples:
            ax.axis("off")
            continue

        i = indices[idx]
        pattern = attn[i].cpu().numpy()  # (3, 3)
        im = ax.imshow(pattern, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(3))
        ax.set_xticklabels(token_labels, fontsize=8)
        ax.set_yticks(range(3))
        ax.set_yticklabels(token_labels, fontsize=8)
        a, b = int(a_vals[i]), int(b_vals[i])
        ax.set_title(f"({a}, {b}) -> {(a + b) % p}", fontsize=9)

    fig.suptitle("Attention Patterns by Input", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig
