#!/usr/bin/env python
"""Cross-run comparison plots for operation sweep.

Loads metrics.json from all sweep runs and produces:
1. Overlaid test accuracy curves
2. Test accuracy curves (log x-axis)
3. Grokking time bar chart
4. Gini evolution comparison
5. Weight norm trajectories
6. Train loss overlay (log-log)
7. Frequency composition subplots (key scientific plot)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, _OP_SUFFIXES

SWEEP_OPERATIONS = ["addition", "subtraction", "multiplication", "x2_plus_y2", "x3_plus_xy"]

OP_LABELS = {
    "addition": "a + b",
    "subtraction": "a \u2212 b",
    "multiplication": "a \u00d7 b",
    "x2_plus_y2": "a\u00b2 + b\u00b2",
    "x3_plus_xy": "a\u00b3 + ab",
}


def _run_id_for_op(op):
    """Return the run directory name for a given operation."""
    suffix = _OP_SUFFIXES.get(op)
    base = "p113_d128_h4_mlp512_L1_wd1.0_s42"
    if suffix is not None:
        return f"{base}_{suffix}"
    return base


def load_all_runs(results_root, operations, logger):
    """Load metrics for all operations. Returns dict {op: metrics_dict}."""
    runs = {}
    for op in operations:
        rid = _run_id_for_op(op)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing: {metrics_path}")
            continue
        with open(metrics_path) as f:
            runs[op] = json.load(f)
        logger.info(f"Loaded {op}: {rid}")
    return runs


def get_colormap(operations):
    """Return a color for each operation using tab10 (qualitative)."""
    cmap = plt.get_cmap("tab10")
    return {op: cmap(i % 10) for i, op in enumerate(operations)}


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def _op_label(op):
    """Format operation for plot labels."""
    return OP_LABELS.get(op, op)


def plot_test_accuracy_overlay(runs, colors, fig_dir):
    """Figure 1: Test accuracy vs epoch for all operations."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        history = runs[op]["history"]
        epochs = history["eval_epochs"]
        test_acc = history["test_acc"]
        ax.plot(epochs, test_acc, color=colors[op], label=_op_label(op), linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Grokking: Test Accuracy vs Operation", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="_nolegend_")

    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_test_accuracy_overlay_log(runs, colors, fig_dir):
    """Figure 2: Same but with log-scale x-axis."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        history = runs[op]["history"]
        epochs = history["eval_epochs"]
        test_acc = history["test_acc"]
        start = 1 if epochs[0] == 0 else 0
        ax.plot(epochs[start:], test_acc[start:], color=colors[op],
                label=_op_label(op), linewidth=1.5)

    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Grokking: Test Accuracy vs Operation (log scale)", fontsize=14)
    ax.set_xscale("log")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10, loc="center right")
    ax.grid(True, alpha=0.3, which="both")
    ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_test_accuracy_log.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_grokking_time_bar(runs, colors, fig_dir):
    """Figure 3: Grokking epoch bar chart (categorical x-axis)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    ops = []
    grok_epochs = []
    bar_colors = []
    no_grok = []

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        history = runs[op]["history"]
        grok_epoch = find_grokking_epoch(history)
        if grok_epoch is not None:
            ops.append(_op_label(op))
            grok_epochs.append(grok_epoch)
            bar_colors.append(colors[op])
        else:
            no_grok.append(op)

    if ops:
        bars = ax.bar(ops, grok_epochs, color=bar_colors, edgecolor="black", linewidth=0.8)
        for bar, epoch in zip(bars, grok_epochs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    str(epoch), ha="center", va="bottom", fontsize=10, fontweight="bold")

    if no_grok:
        no_grok_labels = [_op_label(op) for op in no_grok]
        ax.set_title(f"Grokking Time by Operation\n(no grok: {', '.join(no_grok_labels)})", fontsize=14)
    else:
        ax.set_title("Grokking Time by Operation", fontsize=14)

    ax.set_xlabel("Operation", fontsize=12)
    ax.set_ylabel("Grokking Epoch (test acc > 95%)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_grokking_time.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_gini_evolution(runs, colors, fig_dir):
    """Figure 4: Gini coefficient vs epoch for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        history = runs[op]["history"]
        fourier_epochs = history.get("fourier_epochs", [])
        gini = history.get("gini", [])
        if fourier_epochs and gini:
            ax.plot(fourier_epochs, gini, color=colors[op],
                    label=_op_label(op), linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_title("Fourier Sparsity (Gini) Evolution Across Operations", fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_gini_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_weight_norm_trajectories(runs, colors, fig_dir):
    """Figure 5: Weight norm vs epoch for all runs."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        history = runs[op]["history"]
        epochs = history.get("eval_epochs", [])
        w_norm = history.get("weight_norm", [])
        if epochs and w_norm:
            ax.plot(epochs, w_norm, color=colors[op],
                    label=_op_label(op), linewidth=1.5)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Weight Norm (L2)", fontsize=12)
    ax.set_title("Weight Norm Trajectories Across Operations", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_weight_norm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_train_loss_overlay(runs, colors, fig_dir):
    """Figure 6: Train loss comparison (log-log)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        history = runs[op]["history"]
        epochs = history.get("eval_epochs", [])
        train_loss = history.get("train_loss", [])
        if epochs and train_loss:
            start = 1 if epochs[0] == 0 else 0
            ax.plot(epochs[start:], train_loss[start:], color=colors[op],
                    label=_op_label(op), linewidth=1.5)

    ax.set_xlabel("Epoch (log scale)", fontsize=12)
    ax.set_ylabel("Train Loss", fontsize=12)
    ax.set_title("Train Loss Across Operations", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_train_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_frequency_composition(runs, colors, results_root, fig_dir):
    """Figure 7: Per-operation frequency spectrum subplots.

    Shows which Fourier frequencies each operation selects — the key
    scientific plot for comparing internal representations.
    """
    available_ops = [op for op in SWEEP_OPERATIONS if op in runs]
    n_ops = len(available_ops)
    if n_ops == 0:
        return

    fig, axes = plt.subplots(n_ops, 1, figsize=(12, 3 * n_ops), sharex=True)
    if n_ops == 1:
        axes = [axes]

    for ax, op in zip(axes, available_ops):
        rid = _run_id_for_op(op)
        snap_path = results_root / rid / "fourier_snapshots.npz"

        if snap_path.exists():
            data = np.load(snap_path)
            freq_norms = data["frequency_norms"]
            # Use the last snapshot (final state)
            final_norms = freq_norms[-1]
            p = len(final_norms)
            # Only plot up to p//2 (symmetric)
            half_p = p // 2 + 1
            freqs = np.arange(half_p)
            norms = final_norms[:half_p]
            ax.bar(freqs, norms, color=colors[op], alpha=0.8, edgecolor="none")
        else:
            # Fall back to key_frequencies from metrics
            key_freqs = runs[op].get("final_key_frequencies", [])
            if key_freqs:
                ax.bar(key_freqs, [1.0] * len(key_freqs), color=colors[op], alpha=0.8)

        ax.set_ylabel("Norm", fontsize=10)
        ax.set_title(f"{_op_label(op)} (mod 113)", fontsize=11, fontweight="bold",
                     color=colors[op])
        ax.grid(True, alpha=0.3, axis="y")

    axes[-1].set_xlabel("Frequency", fontsize=12)
    fig.suptitle("Fourier Frequency Composition by Operation", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "op_sweep_frequency_composition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def print_frequency_table(runs, logger):
    """Print key frequency comparison to console."""
    print("\n" + "=" * 100)
    print("KEY FREQUENCY COMPARISON \u2014 OPERATION SWEEP")
    print("=" * 100)
    print(f"{'Operation':>15} | {'Gini':>6} | {'Test Acc':>8} | {'Grok Epoch':>10} | Key Frequencies")
    print("-" * 100)

    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        metrics = runs[op]
        history = metrics["history"]
        grok_epoch = find_grokking_epoch(history)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
        gini = metrics.get("final_gini", 0)
        test_acc = metrics.get("final_test_acc", 0)
        key_freqs = metrics.get("final_key_frequencies", [])
        print(f"{_op_label(op):>15} | {gini:>6.3f} | {test_acc:>8.4f} | {grok_str:>10} | {key_freqs}")

    print("=" * 100)

    # Frequency intersection/union analysis across grokking runs
    grok_freqs = {}
    for op in SWEEP_OPERATIONS:
        if op not in runs:
            continue
        grok_epoch = find_grokking_epoch(runs[op]["history"])
        if grok_epoch is not None:
            freqs = set(runs[op].get("final_key_frequencies", []))
            grok_freqs[op] = freqs

    if len(grok_freqs) >= 2:
        all_freq_sets = list(grok_freqs.values())
        common = all_freq_sets[0]
        for s in all_freq_sets[1:]:
            common = common & s
        union = all_freq_sets[0]
        for s in all_freq_sets[1:]:
            union = union | s

        print(f"\nFrequencies common to ALL grokking operations: {sorted(common)}")
        print(f"Union of all grokking operation frequencies:   {sorted(union)}")

        for op, freqs in grok_freqs.items():
            unique = freqs - common
            if unique:
                print(f"  {_op_label(op)} unique frequencies: {sorted(unique)}")

        if common == union:
            print("=> All grokking operations select the SAME frequencies")
        else:
            print("=> Different operations select DIFFERENT frequencies (expected for non-additive operations)")


def main():
    parser = argparse.ArgumentParser(description="Compare operation sweep results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/op_sweep_comparison/)")
    parser.add_argument("--operations", type=str, nargs="+", default=None,
                        help="Operations to compare (default: all 5)")
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    operations = args.operations if args.operations else SWEEP_OPERATIONS
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "op_sweep_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load all runs
    runs = load_all_runs(results_root, operations, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(operations)} runs")
    colors = get_colormap(SWEEP_OPERATIONS)

    # Generate plots
    logger.info("1/7: Test accuracy overlay")
    plot_test_accuracy_overlay(runs, colors, fig_dir)

    logger.info("2/7: Test accuracy overlay (log scale)")
    plot_test_accuracy_overlay_log(runs, colors, fig_dir)

    logger.info("3/7: Grokking time bar chart")
    plot_grokking_time_bar(runs, colors, fig_dir)

    logger.info("4/7: Gini evolution")
    plot_gini_evolution(runs, colors, fig_dir)

    logger.info("5/7: Weight norm trajectories")
    plot_weight_norm_trajectories(runs, colors, fig_dir)

    logger.info("6/7: Train loss overlay")
    plot_train_loss_overlay(runs, colors, fig_dir)

    logger.info("7/7: Frequency composition")
    plot_frequency_composition(runs, colors, results_root, fig_dir)

    # Console summary
    print_frequency_table(runs, logger)

    logger.info(f"\nAll comparison figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
