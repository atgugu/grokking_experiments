"""Weight matrix visualizations."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")


def plot_weight_heatmap(model, p: int) -> plt.Figure:
    """Plot heatmaps of W_E and W_U weight matrices.

    Args:
        model: GrokkingTransformer model.
        p: Prime modulus.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # W_E: token embedding (p+1, d_model)
    W_E = model.W_E.weight.detach().cpu().numpy()
    ax = axes[0]
    im = ax.imshow(W_E[:p, :].T, aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(W_E[:p]).max(), vmax=np.abs(W_E[:p]).max())
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Token Index")
    ax.set_ylabel("Embedding Dimension")
    ax.set_title(f"W_E (Token Embedding, {p} x {W_E.shape[1]})")

    # W_U: unembedding (d_model, p) or check for tied
    if model.W_U is not None:
        W_U = model.W_U.weight.detach().cpu().numpy()  # (p, d_model)
        ax = axes[1]
        im = ax.imshow(W_U, aspect="auto", cmap="RdBu_r",
                        vmin=-np.abs(W_U).max(), vmax=np.abs(W_U).max())
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Output Class")
        ax.set_title(f"W_U (Unembedding, {W_U.shape[0]} x {W_U.shape[1]})")
    else:
        axes[1].text(0.5, 0.5, "Tied Embeddings\n(W_U = W_E[:p])",
                     ha="center", va="center", fontsize=14, transform=axes[1].transAxes)
        axes[1].set_title("W_U (Tied)")

    fig.suptitle("Weight Matrices", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_weight_evolution(
    checkpoint_paths: list,
    config: dict,
    p: int,
    device: torch.device,
) -> plt.Figure:
    """Plot W_E evolution across checkpoints.

    Args:
        checkpoint_paths: List of Path objects to checkpoint files.
        config: Model config dict.
        p: Prime modulus.
        device: Compute device.
    """
    from ..models.transformer import GrokkingTransformer

    n_ckpts = len(checkpoint_paths)
    fig, axes = plt.subplots(2, n_ckpts, figsize=(5 * n_ckpts, 8))
    if n_ckpts == 1:
        axes = axes.reshape(2, 1)

    for i, ckpt_path in enumerate(checkpoint_paths):
        epoch = int(ckpt_path.stem.split("_")[1])
        model = GrokkingTransformer(
            p=config["p"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_mlp=config["d_mlp"],
            n_layers=config["n_layers"],
            activation=config.get("activation", "relu"),
            use_layernorm=config.get("use_layernorm", False),
            tie_embeddings=config.get("tie_embeddings", False),
            mlp_bias=config.get("mlp_bias", True),
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))

        # W_E
        W_E = model.W_E.weight.detach().cpu().numpy()[:p, :]
        ax = axes[0, i]
        vmax = np.abs(W_E).max()
        im = ax.imshow(W_E.T, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"W_E @ epoch {epoch}", fontsize=9)
        if i == 0:
            ax.set_ylabel("Dim")
        ax.set_xlabel("Token")

        # W_U
        if model.W_U is not None:
            W_U = model.W_U.weight.detach().cpu().numpy()
            ax = axes[1, i]
            vmax = np.abs(W_U).max()
            im = ax.imshow(W_U, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"W_U @ epoch {epoch}", fontsize=9)
            if i == 0:
                ax.set_ylabel("Class")
            ax.set_xlabel("Dim")

    fig.suptitle("Weight Evolution Over Training", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig
