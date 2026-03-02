"""Training curve visualizations for grokking experiments."""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")


def plot_grokking_curves(history: dict) -> plt.Figure:
    """Plot 4-panel grokking overview: loss, accuracy, weight norm, Gini.

    Args:
        history: Training history dict with keys:
            eval_epochs, train_loss, test_loss, train_acc, test_acc,
            weight_norm, gini, fourier_epochs

    Returns:
        matplotlib Figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    eval_epochs = history.get("eval_epochs", [])
    fourier_epochs = history.get("fourier_epochs", [])

    # Panel 1: Loss curves (log scale)
    ax = axes[0, 0]
    if history.get("train_loss"):
        ax.semilogy(eval_epochs[:len(history["train_loss"])], history["train_loss"],
                     label="Train", color="#636EFA")
    if history.get("test_loss"):
        ax.semilogy(eval_epochs[:len(history["test_loss"])], history["test_loss"],
                     label="Test", color="#EF553B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss (log scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: Accuracy
    ax = axes[0, 1]
    if history.get("train_acc"):
        ax.plot(eval_epochs[:len(history["train_acc"])], history["train_acc"],
                label="Train", color="#636EFA")
    if history.get("test_acc"):
        ax.plot(eval_epochs[:len(history["test_acc"])], history["test_acc"],
                label="Test", color="#EF553B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: Weight norm
    ax = axes[1, 0]
    if history.get("weight_norm"):
        ax.plot(eval_epochs[:len(history["weight_norm"])], history["weight_norm"],
                color="#00CC96")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Total Weight Norm")
    ax.grid(True, alpha=0.3)

    # Panel 4: Gini coefficient
    ax = axes[1, 1]
    if history.get("gini"):
        ax.plot(fourier_epochs[:len(history["gini"])], history["gini"],
                color="#AB63FA")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gini")
    ax.set_title("Fourier Gini Coefficient")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Grokking Dynamics", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_progress_measures(history: dict, fourier_snapshots: dict) -> plt.Figure:
    """Plot progress measures over training.

    Shows restricted loss, excluded loss, Gini, and weight norm evolution.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    eval_epochs = history.get("eval_epochs", [])
    fourier_epochs = history.get("fourier_epochs", [])

    # We need restricted/excluded loss from analysis; use available data
    ax = axes[0, 0]
    if history.get("train_loss"):
        ax.semilogy(eval_epochs[:len(history["train_loss"])], history["train_loss"],
                     label="Full Train Loss", color="#636EFA")
    if history.get("test_loss"):
        ax.semilogy(eval_epochs[:len(history["test_loss"])], history["test_loss"],
                     label="Full Test Loss", color="#EF553B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Gini evolution
    ax = axes[0, 1]
    if history.get("gini"):
        ax.plot(fourier_epochs[:len(history["gini"])], history["gini"],
                color="#AB63FA", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Gini")
    ax.set_title("Fourier Sparsity (Gini)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Weight norm
    ax = axes[1, 0]
    if history.get("weight_norm"):
        ax.plot(eval_epochs[:len(history["weight_norm"])], history["weight_norm"],
                color="#00CC96", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L2 Norm")
    ax.set_title("Weight Norm")
    ax.grid(True, alpha=0.3)

    # Key frequency norms evolution
    ax = axes[1, 1]
    if "frequency_norms" in fourier_snapshots:
        freq_norms_arr = fourier_snapshots["frequency_norms"]  # (n_snapshots, p)
        f_epochs = fourier_snapshots.get("fourier_epochs", np.arange(len(freq_norms_arr)))
        # Plot top 5 frequencies from final snapshot
        final_norms = freq_norms_arr[-1]
        final_norms_no_dc = final_norms.copy()
        final_norms_no_dc[0] = 0
        top_k = np.argsort(final_norms_no_dc)[::-1][:5]
        colors = plt.cm.Set1(np.linspace(0, 1, 5))
        for i, k in enumerate(top_k):
            ax.plot(f_epochs, freq_norms_arr[:, k], label=f"k={k}", color=colors[i])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Frequency Energy")
        ax.set_title("Key Frequency Evolution")
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Progress Measures", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_phase_boundaries(history: dict) -> plt.Figure:
    """Plot train/test accuracy with annotated phase boundaries.

    Marks: memorization epoch (train acc > 95%), grokking epoch (test acc > 95%).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    eval_epochs = history.get("eval_epochs", [])
    train_acc = history.get("train_acc", [])
    test_acc = history.get("test_acc", [])

    if train_acc:
        ax.plot(eval_epochs[:len(train_acc)], train_acc, label="Train", color="#636EFA", linewidth=2)
    if test_acc:
        ax.plot(eval_epochs[:len(test_acc)], test_acc, label="Test", color="#EF553B", linewidth=2)

    # Find phase boundaries
    mem_epoch = None
    grok_epoch = None
    for i, acc in enumerate(train_acc):
        if acc > 0.95 and mem_epoch is None:
            mem_epoch = eval_epochs[i]
    for i, acc in enumerate(test_acc):
        if acc > 0.95 and grok_epoch is None:
            grok_epoch = eval_epochs[i]

    if mem_epoch is not None:
        ax.axvline(mem_epoch, color="#636EFA", linestyle="--", alpha=0.5, label=f"Memorization (~{mem_epoch})")
    if grok_epoch is not None:
        ax.axvline(grok_epoch, color="#EF553B", linestyle="--", alpha=0.5, label=f"Grokking (~{grok_epoch})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Grokking Phase Boundaries")
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig
