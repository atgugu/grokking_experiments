#!/usr/bin/env python
"""Cross-run comparison plots for train fraction sweep.

Loads metrics.json from all sweep runs and produces:
1. Overlaid test accuracy curves
2. Test accuracy curves (log x-axis)
3. Grokking time vs train fraction
4. Gini evolution comparison
5. Weight norm trajectories
6. Train loss overlay (log-log)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

DEFAULT_TF_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
P = 113  # modular arithmetic prime


def load_all_runs(results_root, tf_values, logger):
    """Load metrics for all tf values. Returns dict {tf: metrics_dict}."""
    runs = {}
    for tf in tf_values:
        # tf=0.3 is the default run (no _tf suffix)
        if tf == 0.3:
            rid = "p113_d128_h4_mlp512_L1_wd1.0_s42"
        else:
            rid = f"p113_d128_h4_mlp512_L1_wd1.0_s42_tf{tf}"
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing: {metrics_path}")
            continue
        with open(metrics_path) as f:
            runs[tf] = json.load(f)
        logger.info(f"Loaded tf={tf}: {rid}")
    return runs


def get_colormap(tf_values):
    """Return a color for each tf value using a perceptually distinct colormap."""
    cmap = plt.get_cmap("viridis", len(tf_values) + 2)
    n = max(len(tf_values) - 1, 1)
    return {tf: cmap(i / n) for i, tf in enumerate(tf_values)}


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def plot_test_accuracy_overlay(runs, colors, fig_dir):
    """Figure 1: Test accuracy vs epoch for all tf values."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for tf in sorted(runs.keys()):
        metrics = runs[tf]
        history = metrics["history"]
        epochs = history["eval_epochs"]
        test_acc = history["test_acc"]
        ax.plot(epochs, test_acc, color=colors[tf], label=f"tf={tf}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Grokking: Test Accuracy vs Train Fraction", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

    fig.tight_layout()
    fig.savefig(fig_dir / "tf_sweep_test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_test_accuracy_overlay_log(runs, colors, fig_dir):
    """Figure 2: Same but with log-scale x-axis."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for tf in sorted(runs.keys()):
        metrics = runs[tf]
        history = metrics["history"]
        epochs = history["eval_epochs"]
        test_acc = history["test_acc"]
        start = 1 if epochs[0] == 0 else 0
        ax.plot(epochs[start:], test_acc[start:], color=colors[tf],
                label=f"tf={tf}", linewidth=1.5)

    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Grokking: Test Accuracy vs Train Fraction (log scale)", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "tf_sweep_test_accuracy_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_grokking_time_vs_tf(runs, colors, fig_dir):
    """Figure 3: Grokking epoch vs train fraction."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    total_pairs = P * P  # 12769
    tf_grok = []
    tf_no_grok = []

    for tf in sorted(runs.keys()):
        history = runs[tf]["history"]
        grok_epoch = find_grokking_epoch(history)
        if grok_epoch is not None:
            tf_grok.append((tf, grok_epoch))
        else:
            tf_no_grok.append(tf)

    if tf_grok:
        tfs, epochs = zip(*tf_grok)
        ax.scatter(tfs, epochs, c=[colors[t] for t in tfs], s=100, zorder=5, edgecolors="black")
        ax.plot(tfs, epochs, "k--", alpha=0.3, zorder=3)

        for t, e in tf_grok:
            n_train = int(round(t * total_pairs))
            n_test = total_pairs - n_train
            ax.annotate(f"ep={e}\n({n_train}/{n_test})",
                        (t, e), textcoords="offset points",
                        xytext=(10, 5), fontsize=8, color=colors[t])

    # Mark non-grokking runs
    for tf in tf_no_grok:
        ax.axvline(x=tf, color=colors[tf], linestyle=":", alpha=0.5)
        n_train = int(round(tf * total_pairs))
        ax.annotate(f"tf={tf}\n({n_train} train)\n(no grok)",
                    (tf, ax.get_ylim()[1] * 0.8 if ax.get_ylim()[1] > 1 else 30000),
                    fontsize=8, ha="center", color=colors[tf])

    ax.set_xlabel("Train Fraction", fontsize=12)
    ax.set_ylabel("Grokking Epoch (test acc > 95%)", fontsize=12)
    ax.set_title("Phase Transition: Grokking Time vs Train Fraction", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fig_dir / "tf_sweep_grokking_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_gini_evolution(runs, colors, fig_dir):
    """Figure 4: Gini coefficient vs epoch for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for tf in sorted(runs.keys()):
        history = runs[tf]["history"]
        fourier_epochs = history.get("fourier_epochs", [])
        gini = history.get("gini", [])
        if fourier_epochs and gini:
            ax.plot(fourier_epochs, gini, color=colors[tf],
                    label=f"tf={tf}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title("Fourier Sparsity (Gini) Evolution Across Train Fractions", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "tf_sweep_gini_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_weight_norm_trajectories(runs, colors, fig_dir):
    """Figure 5: Weight norm vs epoch for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for tf in sorted(runs.keys()):
        history = runs[tf]["history"]
        epochs = history.get("eval_epochs", [])
        w_norm = history.get("weight_norm", [])
        if epochs and w_norm:
            ax.plot(epochs, w_norm, color=colors[tf],
                    label=f"tf={tf}", linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weight Norm (L2)", fontsize=12)
    ax.set_title("Weight Norm Trajectories Across Train Fractions", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "tf_sweep_weight_norm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_train_loss_overlay(runs, colors, fig_dir):
    """Figure 6: Train loss comparison (log-log)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for tf in sorted(runs.keys()):
        history = runs[tf]["history"]
        epochs = history.get("eval_epochs", [])
        train_loss = history.get("train_loss", [])
        if epochs and train_loss:
            start = 1 if epochs[0] == 0 else 0
            ax.plot(epochs[start:], train_loss[start:], color=colors[tf],
                    label=f"tf={tf}", linewidth=1.5)

    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss Across Train Fractions", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fig_dir / "tf_sweep_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def print_frequency_table(runs, logger):
    """Print key frequency comparison to console."""
    total_pairs = P * P  # 12769

    print("\n" + "=" * 100)
    print("KEY FREQUENCY COMPARISON \u2014 TRAIN FRACTION SWEEP")
    print("=" * 100)
    print(f"{'TF':>6} | {'# Train':>7} | {'# Test':>7} | {'Gini':>6} | {'Test Acc':>8} | {'Grok Epoch':>10} | Key Frequencies")
    print("-" * 100)

    for tf in sorted(runs.keys()):
        metrics = runs[tf]
        history = metrics["history"]
        grok_epoch = find_grokking_epoch(history)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
        gini = metrics.get("final_gini", 0)
        test_acc = metrics.get("final_test_acc", 0)
        key_freqs = metrics.get("final_key_frequencies", [])
        n_train = int(round(tf * total_pairs))
        n_test = total_pairs - n_train
        print(f"{tf:>6} | {n_train:>7} | {n_test:>7} | {gini:>6.3f} | {test_acc:>8.4f} | {grok_str:>10} | {key_freqs}")

    print("=" * 100)

    # Check if frequencies are consistent across grokking runs
    grok_freqs = {}
    for tf in sorted(runs.keys()):
        grok_epoch = find_grokking_epoch(runs[tf]["history"])
        if grok_epoch is not None:
            freqs = set(runs[tf].get("final_key_frequencies", []))
            grok_freqs[tf] = freqs

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
            print("=> Frequencies VARY across train fractions")


def main():
    parser = argparse.ArgumentParser(description="Compare train fraction sweep results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/tf_sweep_comparison/)")
    parser.add_argument("--tf-values", type=float, nargs="+", default=None,
                        help="TF values to compare (default: 0.05 0.1 0.2 0.3 0.4 0.5 0.7)")
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    tf_values = args.tf_values if args.tf_values else DEFAULT_TF_VALUES
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "tf_sweep_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = load_all_runs(results_root, tf_values, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(tf_values)} runs")
    colors = get_colormap(sorted(runs.keys()))

    # Generate plots
    logger.info("1/6: Test accuracy overlay")
    plot_test_accuracy_overlay(runs, colors, fig_dir)

    logger.info("2/6: Test accuracy overlay (log scale)")
    plot_test_accuracy_overlay_log(runs, colors, fig_dir)

    logger.info("3/6: Grokking time vs train fraction")
    plot_grokking_time_vs_tf(runs, colors, fig_dir)

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
