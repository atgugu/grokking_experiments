#!/usr/bin/env python
"""Cross-run comparison plots for weight decay sweep.

Loads metrics.json from all sweep runs and produces:
1. Overlaid test accuracy curves (hero figure)
2. Grokking time vs weight decay (log-log)
3. Gini evolution comparison
4. Weight norm trajectories
5. Key frequency comparison table (console)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

DEFAULT_WD_VALUES = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


def load_all_runs(results_root, wd_values, logger):
    """Load metrics for all wd values. Returns dict {wd: metrics_dict}."""
    runs = {}
    for wd in wd_values:
        rid = f"p113_d128_h4_mlp512_L1_wd{wd}_s42"
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing: {metrics_path}")
            continue
        with open(metrics_path) as f:
            runs[wd] = json.load(f)
        logger.info(f"Loaded wd={wd}: {rid}")
    return runs


def get_colormap(wd_values):
    """Return a color for each wd value using a perceptually distinct colormap."""
    cmap = plt.get_cmap("viridis", len(wd_values) + 2)
    n = max(len(wd_values) - 1, 1)
    return {wd: cmap(i / n) for i, wd in enumerate(wd_values)}


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def plot_test_accuracy_overlay(runs, colors, fig_dir):
    """Figure 1: Test accuracy vs epoch for all wd values."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for wd in sorted(runs.keys()):
        metrics = runs[wd]
        history = metrics["history"]
        epochs = history["eval_epochs"]
        test_acc = history["test_acc"]
        ax.plot(epochs, test_acc, color=colors[wd], label=f"wd={wd}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Grokking: Test Accuracy vs Weight Decay", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

    fig.tight_layout()
    fig.savefig(fig_dir / "wd_sweep_test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_test_accuracy_overlay_log(runs, colors, fig_dir):
    """Figure 1b: Same but with log-scale x-axis for better visibility of early dynamics."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for wd in sorted(runs.keys()):
        metrics = runs[wd]
        history = metrics["history"]
        epochs = history["eval_epochs"]
        test_acc = history["test_acc"]
        # Skip epoch 0 for log scale
        start = 1 if epochs[0] == 0 else 0
        ax.plot(epochs[start:], test_acc[start:], color=colors[wd],
                label=f"wd={wd}", linewidth=1.5)

    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Grokking: Test Accuracy vs Weight Decay (log scale)", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "wd_sweep_test_accuracy_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_grokking_time_vs_wd(runs, colors, fig_dir):
    """Figure 2: Grokking epoch vs weight decay on log-log scale."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    wd_grok = []
    wd_no_grok = []

    for wd in sorted(runs.keys()):
        history = runs[wd]["history"]
        grok_epoch = find_grokking_epoch(history)
        if grok_epoch is not None:
            wd_grok.append((wd, grok_epoch))
        else:
            wd_no_grok.append(wd)

    if wd_grok:
        wds, epochs = zip(*wd_grok)
        ax.scatter(wds, epochs, c=[colors[w] for w in wds], s=100, zorder=5, edgecolors="black")
        ax.plot(wds, epochs, "k--", alpha=0.3, zorder=3)

        for w, e in wd_grok:
            ax.annotate(f"{e}", (w, e), textcoords="offset points",
                        xytext=(8, 5), fontsize=9, color=colors[w])

    # Mark non-grokking runs
    for wd in wd_no_grok:
        ax.axvline(x=wd, color=colors[wd], linestyle=":", alpha=0.5)
        ax.annotate(f"wd={wd}\n(no grok)", (wd, ax.get_ylim()[1] * 0.8),
                    fontsize=9, ha="center", color=colors[wd])

    ax.set_xlabel("Weight Decay", fontsize=12)
    ax.set_ylabel("Grokking Epoch (test acc > 95%)", fontsize=12)
    ax.set_title("Phase Transition: Grokking Time vs Weight Decay", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fig_dir / "wd_sweep_grokking_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_gini_evolution(runs, colors, fig_dir):
    """Figure 3: Gini coefficient vs epoch for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for wd in sorted(runs.keys()):
        history = runs[wd]["history"]
        fourier_epochs = history.get("fourier_epochs", [])
        gini = history.get("gini", [])
        if fourier_epochs and gini:
            ax.plot(fourier_epochs, gini, color=colors[wd],
                    label=f"wd={wd}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title("Fourier Sparsity (Gini) Evolution Across Weight Decays", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "wd_sweep_gini_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_weight_norm_trajectories(runs, colors, fig_dir):
    """Figure 4: Weight norm vs epoch for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for wd in sorted(runs.keys()):
        history = runs[wd]["history"]
        epochs = history.get("eval_epochs", [])
        w_norm = history.get("weight_norm", [])
        if epochs and w_norm:
            ax.plot(epochs, w_norm, color=colors[wd],
                    label=f"wd={wd}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weight Norm (L2)", fontsize=12)
    ax.set_title("Weight Norm Trajectories: Rise-then-Fall in Grokking", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "wd_sweep_weight_norm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_train_loss_overlay(runs, colors, fig_dir):
    """Bonus: Train loss comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for wd in sorted(runs.keys()):
        history = runs[wd]["history"]
        epochs = history.get("eval_epochs", [])
        train_loss = history.get("train_loss", [])
        if epochs and train_loss:
            start = 1 if epochs[0] == 0 else 0
            ax.plot(epochs[start:], train_loss[start:], color=colors[wd],
                    label=f"wd={wd}", linewidth=1.5)

    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss Across Weight Decays", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fig_dir / "wd_sweep_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def print_frequency_table(runs, logger):
    """Print key frequency comparison to console."""
    print("\n" + "=" * 80)
    print("KEY FREQUENCY COMPARISON")
    print("=" * 80)
    print(f"{'WD':>6} | {'Gini':>6} | {'Test Acc':>8} | {'Grok Epoch':>10} | Key Frequencies")
    print("-" * 80)

    for wd in sorted(runs.keys()):
        metrics = runs[wd]
        history = metrics["history"]
        grok_epoch = find_grokking_epoch(history)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
        gini = metrics.get("final_gini", 0)
        test_acc = metrics.get("final_test_acc", 0)
        key_freqs = metrics.get("final_key_frequencies", [])
        print(f"{wd:>6} | {gini:>6.3f} | {test_acc:>8.4f} | {grok_str:>10} | {key_freqs}")

    print("=" * 80)

    # Check if frequencies are consistent across grokking runs
    grok_freqs = {}
    for wd in sorted(runs.keys()):
        grok_epoch = find_grokking_epoch(runs[wd]["history"])
        if grok_epoch is not None:
            freqs = set(runs[wd].get("final_key_frequencies", []))
            grok_freqs[wd] = freqs

    if len(grok_freqs) >= 2:
        all_freqs = list(grok_freqs.values())
        common = all_freqs[0]
        for s in all_freqs[1:]:
            common = common & s
        union = all_freqs[0]
        for s in all_freqs[1:]:
            union = union | s

        print(f"\nFrequencies common to ALL grokking runs: {sorted(common)}")
        print(f"Union of all grokking run frequencies:   {sorted(union)}")
        if common == union:
            print("=> All grokking runs select the SAME frequencies (intrinsic to Z/113Z)")
        else:
            print("=> Frequencies VARY across weight decays")


def main():
    parser = argparse.ArgumentParser(description="Compare weight decay sweep results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/wd_sweep_comparison/)")
    parser.add_argument("--wd-values", type=float, nargs="+", default=None,
                        help="WD values to compare (default: 0.01 0.1 0.3 0.5 1.0 2.0 5.0)")
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    wd_values = args.wd_values if args.wd_values else DEFAULT_WD_VALUES
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "wd_sweep_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = load_all_runs(results_root, wd_values, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(wd_values)} runs")
    colors = get_colormap(sorted(runs.keys()))

    # Generate plots
    logger.info("1/6: Test accuracy overlay")
    plot_test_accuracy_overlay(runs, colors, fig_dir)

    logger.info("2/6: Test accuracy overlay (log scale)")
    plot_test_accuracy_overlay_log(runs, colors, fig_dir)

    logger.info("3/6: Grokking time vs weight decay")
    plot_grokking_time_vs_wd(runs, colors, fig_dir)

    logger.info("4/6: Gini evolution")
    plot_gini_evolution(runs, colors, fig_dir)

    logger.info("5/6: Weight norm trajectories")
    plot_weight_norm_trajectories(runs, colors, fig_dir)

    logger.info("6/6: Train loss overlay")
    plot_train_loss_overlay(runs, colors, fig_dir)

    # Console summary
    print_frequency_table(runs, logger)

    logger.info(f"\nAll comparison figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
