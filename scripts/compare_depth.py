#!/usr/bin/env python
"""Cross-run comparison plots for depth × operation sweep.

Loads metrics.json from all sweep runs and produces:
1. 5×3 heatmap: (operation × n_layers) → grokking epoch ("Never" if no grok)
2. Test accuracy curves grouped by n_layers
3. Grokking time bar chart per operation, grouped by depth
4. Gini evolution comparison (per-layer, per-operation)
5. Weight norm trajectories
6. Key question: does x³+ab grok at n_layers ≥ 2?
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging, _OP_SUFFIXES

SWEEP_OPERATIONS = ["addition", "subtraction", "multiplication", "x2_plus_y2", "x3_plus_xy"]
SWEEP_LAYERS = [1, 2, 3]

OP_LABELS = {
    "addition": "a + b",
    "subtraction": "a \u2212 b",
    "multiplication": "a \u00d7 b",
    "x2_plus_y2": "a\u00b2 + b\u00b2",
    "x3_plus_xy": "a\u00b3 + ab",
}

LAYER_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c"}
OP_MARKERS = {
    "addition": "o",
    "subtraction": "s",
    "multiplication": "^",
    "x2_plus_y2": "D",
    "x3_plus_xy": "*",
}


def _run_id_for(op, n_layers):
    """Return the run directory name for a given (operation, n_layers) pair."""
    suffix = _OP_SUFFIXES.get(op)
    base = f"p113_d128_h4_mlp512_L{n_layers}_wd1.0_s42"
    if suffix is not None:
        return f"{base}_{suffix}"
    return base


def load_all_runs(results_root, operations, layers, logger):
    """Load metrics for all (op, n_layers) combos. Returns dict {(op, n_layers): metrics_dict}."""
    runs = {}
    for n_layers in layers:
        for op in operations:
            rid = _run_id_for(op, n_layers)
            metrics_path = results_root / rid / "metrics.json"
            if not metrics_path.exists():
                logger.warning(f"Missing: {metrics_path}")
                continue
            with open(metrics_path) as f:
                runs[(op, n_layers)] = json.load(f)
            logger.info(f"Loaded ({op}, L={n_layers}): {rid}")
    return runs


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def plot_grokking_heatmap(runs, operations, layers, fig_dir):
    """Figure 1: Heatmap of grokking epoch for (operation × n_layers).

    Green = grokked (log scale), red = never grokked.
    This is the key plot showing which depth unlocks which operations.
    """
    # Build matrix: rows=operations, cols=layers
    n_ops = len(operations)
    n_layers_vals = len(layers)
    matrix = np.full((n_ops, n_layers_vals), np.nan)
    cell_text = []

    for i, op in enumerate(operations):
        row_text = []
        for j, n_layers in enumerate(layers):
            key = (op, n_layers)
            if key not in runs:
                row_text.append("Missing")
                continue
            history = runs[key].get("history", {})
            grok_epoch = find_grokking_epoch(history)
            if grok_epoch is not None:
                matrix[i, j] = grok_epoch
                row_text.append(str(grok_epoch))
            else:
                matrix[i, j] = -1  # sentinel for "never"
                row_text.append("Never")
        cell_text.append(row_text)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Create custom colormap: gray for Never, green gradient for grokked
    # We'll use a masked array approach
    grokked_mask = matrix > 0
    never_mask = matrix == -1
    missing_mask = np.isnan(matrix)

    # Plot background: all cells
    with np.errstate(divide="ignore", invalid="ignore"):
        display = np.where(grokked_mask, np.log10(np.where(grokked_mask, matrix + 1, 1)), 0)
    max_val = display[grokked_mask].max() if grokked_mask.any() else 4
    display[never_mask] = -1  # different from 0

    # Custom colormap
    from matplotlib.colors import LinearSegmentedColormap
    green_cmap = LinearSegmentedColormap.from_list("greens", ["#d4f5c9", "#1a7a1a"])

    # Draw each cell manually
    for i in range(n_ops):
        for j in range(n_layers_vals):
            text = cell_text[i][j]
            if text == "Missing":
                color = "#cccccc"
                text_color = "black"
            elif text == "Never":
                color = "#e74c3c"
                text_color = "white"
            else:
                # Log-scale green intensity
                epoch = matrix[i, j]
                t = min(1.0, np.log10(epoch + 1) / max_val) if max_val > 0 else 0.5
                # Invert: fewer epochs = darker green (better)
                t_inv = 1.0 - t
                r = 0.212 + t_inv * (0.831 - 0.212)
                g = 0.482 + t_inv * (0.980 - 0.482)
                b = 0.192 + t_inv * (0.804 - 0.192)
                color = (r, g, b)
                text_color = "black" if t_inv < 0.7 else "white"

            rect = plt.Rectangle([j - 0.5, i - 0.5], 1, 1, facecolor=color,
                                  edgecolor="white", linewidth=2)
            ax.add_patch(rect)
            ax.text(j, i, text, ha="center", va="center", fontsize=11,
                    fontweight="bold", color=text_color)

    ax.set_xticks(range(n_layers_vals))
    ax.set_xticklabels([f"L={l}" for l in layers], fontsize=12)
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels([OP_LABELS.get(op, op) for op in operations], fontsize=12)
    ax.set_xlim(-0.5, n_layers_vals - 0.5)
    ax.set_ylim(-0.5, n_ops - 0.5)
    ax.set_xlabel("Number of Layers", fontsize=13)
    ax.set_ylabel("Operation", fontsize=13)
    ax.set_title("Grokking Epoch by Depth × Operation\n(green=grokked, red=Never)", fontsize=14)

    # Legend patches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#1a7a1a", label="Grokked early"),
        Patch(facecolor="#d4f5c9", label="Grokked late"),
        Patch(facecolor="#e74c3c", label="Never grokked"),
        Patch(facecolor="#cccccc", label="Missing"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_dir / "depth_grokking_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: depth_grokking_heatmap.png")


def plot_test_accuracy_by_layer(runs, operations, layers, fig_dir):
    """Figure 2: One subplot per n_layers, showing all operations' test accuracy curves."""
    n_layers_vals = len(layers)
    fig, axes = plt.subplots(1, n_layers_vals, figsize=(6 * n_layers_vals, 5), sharey=True)
    if n_layers_vals == 1:
        axes = [axes]

    op_colors = {op: plt.get_cmap("tab10")(i) for i, op in enumerate(SWEEP_OPERATIONS)}

    for ax, n_layers in zip(axes, layers):
        for op in operations:
            key = (op, n_layers)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            epochs = history.get("eval_epochs", [])
            test_acc = history.get("test_acc", [])
            if epochs and test_acc:
                ax.plot(epochs, test_acc, color=op_colors[op],
                        label=OP_LABELS.get(op, op), linewidth=1.5)

        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_title(f"n_layers = {n_layers}", fontsize=13, fontweight="bold")
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    fig.suptitle("Test Accuracy by Depth and Operation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "depth_test_accuracy_by_layer.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: depth_test_accuracy_by_layer.png")


def plot_test_accuracy_by_op(runs, operations, layers, fig_dir):
    """Figure 3: One subplot per operation, showing all depths' test accuracy curves."""
    n_ops = len(operations)
    ncols = min(3, n_ops)
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, op in enumerate(operations):
        ax = axes_flat[idx]
        for n_layers in layers:
            key = (op, n_layers)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            epochs = history.get("eval_epochs", [])
            test_acc = history.get("test_acc", [])
            if epochs and test_acc:
                ax.plot(epochs, test_acc, color=LAYER_COLORS[n_layers],
                        label=f"L={n_layers}", linewidth=1.5)

        ax.set_title(OP_LABELS.get(op, op), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    # Hide unused subplots
    for idx in range(n_ops, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Test Accuracy", fontsize=10)
    fig.suptitle("Test Accuracy by Operation and Depth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "depth_test_accuracy_by_op.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: depth_test_accuracy_by_op.png")


def plot_grokking_time_grouped(runs, operations, layers, fig_dir):
    """Figure 4: Grouped bar chart — grokking epoch per operation, grouped by depth."""
    ops_with_labels = [(op, OP_LABELS.get(op, op)) for op in operations]
    x = np.arange(len(operations))
    width = 0.25
    offsets = np.linspace(-(len(layers) - 1) / 2, (len(layers) - 1) / 2, len(layers)) * width

    fig, ax = plt.subplots(figsize=(12, 6))
    never_ops = set()

    for layer_idx, n_layers in enumerate(layers):
        grok_epochs = []
        labels_used = []
        xs_used = []
        for i, op in enumerate(operations):
            key = (op, n_layers)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            grok_epoch = find_grokking_epoch(history)
            if grok_epoch is not None:
                grok_epochs.append(grok_epoch)
                xs_used.append(x[i] + offsets[layer_idx])
            else:
                never_ops.add(op)

        if grok_epochs:
            bars = ax.bar(xs_used, grok_epochs, width * 0.85,
                          color=LAYER_COLORS[n_layers], label=f"L={n_layers}",
                          edgecolor="black", linewidth=0.8, alpha=0.85)
            for bar, epoch in zip(bars, grok_epochs):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                        str(epoch), ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS.get(op, op) for op in operations], fontsize=11)
    ax.set_ylabel("Grokking Epoch (test acc > 95%)", fontsize=11)
    ax.set_xlabel("Operation", fontsize=11)

    title = "Grokking Time by Depth and Operation"
    if never_ops:
        never_labels = [OP_LABELS.get(op, op) for op in never_ops]
        title += f"\n(no grok at any depth: {', '.join(never_labels)})"
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_dir / "depth_grokking_time_grouped.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: depth_grokking_time_grouped.png")


def plot_gini_by_op(runs, operations, layers, fig_dir):
    """Figure 5: Gini evolution per operation, one subplot each."""
    n_ops = len(operations)
    ncols = min(3, n_ops)
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=True)
    axes_flat = np.array(axes).flatten()

    for idx, op in enumerate(operations):
        ax = axes_flat[idx]
        for n_layers in layers:
            key = (op, n_layers)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            fourier_epochs = history.get("fourier_epochs", [])
            gini = history.get("gini", [])
            if fourier_epochs and gini:
                ax.plot(fourier_epochs, gini, color=LAYER_COLORS[n_layers],
                        label=f"L={n_layers}", linewidth=1.5)

        ax.set_title(OP_LABELS.get(op, op), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for idx in range(n_ops, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    axes_flat[0].set_ylabel("Gini Coefficient", fontsize=10)
    fig.suptitle("Fourier Sparsity (Gini) by Operation and Depth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "depth_gini_by_op.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: depth_gini_by_op.png")


def plot_weight_norm_by_op(runs, operations, layers, fig_dir):
    """Figure 6: Weight norm evolution per operation, one subplot each."""
    n_ops = len(operations)
    ncols = min(3, n_ops)
    nrows = (n_ops + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), sharey=False)
    axes_flat = np.array(axes).flatten()

    for idx, op in enumerate(operations):
        ax = axes_flat[idx]
        for n_layers in layers:
            key = (op, n_layers)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            epochs = history.get("eval_epochs", [])
            w_norm = history.get("weight_norm", [])
            if epochs and w_norm:
                ax.plot(epochs, w_norm, color=LAYER_COLORS[n_layers],
                        label=f"L={n_layers}", linewidth=1.5)

        ax.set_title(OP_LABELS.get(op, op), fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylabel("Weight Norm", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    for idx in range(n_ops, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("Weight Norm by Operation and Depth", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "depth_weight_norm_by_op.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: depth_weight_norm_by_op.png")


def print_depth_summary(runs, operations, layers, logger):
    """Print a console table and key finding for x³+ab."""
    print("\n" + "=" * 110)
    print("DEPTH \u00d7 OPERATION SWEEP \u2014 SUMMARY")
    print("=" * 110)
    header = f"{'L':>3} | {'Operation':>12} | {'Grok Epoch':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | {'W Norm':>8}"
    print(header)
    print("-" * 110)

    for n_layers in layers:
        for op in operations:
            key = (op, n_layers)
            if key not in runs:
                print(f"{n_layers:>3} | {OP_LABELS.get(op, op):>12} | {'MISSING':>10} | {'—':>9} | {'—':>8} | {'—':>6} | {'—':>8}")
                continue
            metrics = runs[key]
            history = metrics.get("history", {})
            grok_epoch = find_grokking_epoch(history)
            grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
            train_acc = metrics.get("final_train_acc", 0)
            test_acc = metrics.get("final_test_acc", 0)
            gini = metrics.get("final_gini", 0)
            w_norm = metrics.get("final_weight_norm", 0)
            print(f"{n_layers:>3} | {OP_LABELS.get(op, op):>12} | {grok_str:>10} | {train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {w_norm:>8.1f}")
        print("-" * 110)

    print("=" * 110)

    # Key finding: x³+ab at each depth
    print("\n\u2502 KEY FINDING: Does depth unlock x\u00b3+ab? (a\u00b3 + ab mod 113)")
    x3_key = "x3_plus_xy"
    for n_layers in layers:
        key = (x3_key, n_layers)
        if key not in runs:
            status = "MISSING"
        else:
            history = runs[key].get("history", {})
            grok_epoch = find_grokking_epoch(history)
            test_acc = runs[key].get("final_test_acc", 0)
            if grok_epoch is not None:
                status = f"GROKKED at epoch {grok_epoch} (test_acc={test_acc:.4f})"
            else:
                status = f"Never grokked (test_acc={test_acc:.4f})"
        print(f"\u2502   n_layers={n_layers}: {status}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare depth sweep results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Root results directory")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/depth_sweep_comparison/)")
    parser.add_argument("--operations", type=str, nargs="+", default=None,
                        help="Operations to compare (default: all 5)")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="n_layers values to compare (default: 1 2 3)")
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    operations = args.operations if args.operations else SWEEP_OPERATIONS
    layers = args.layers if args.layers else SWEEP_LAYERS
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "depth_sweep_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, operations, layers, logger)
    if not runs:
        logger.error("No runs found! Run the depth sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(layers) * len(operations)} runs")

    logger.info("1/6: Grokking heatmap")
    plot_grokking_heatmap(runs, operations, layers, fig_dir)

    logger.info("2/6: Test accuracy by layer")
    plot_test_accuracy_by_layer(runs, operations, layers, fig_dir)

    logger.info("3/6: Test accuracy by operation")
    plot_test_accuracy_by_op(runs, operations, layers, fig_dir)

    logger.info("4/6: Grokking time grouped bar chart")
    plot_grokking_time_grouped(runs, operations, layers, fig_dir)

    logger.info("5/6: Gini evolution by operation")
    plot_gini_by_op(runs, operations, layers, fig_dir)

    logger.info("6/6: Weight norm by operation")
    plot_weight_norm_by_op(runs, operations, layers, fig_dir)

    print_depth_summary(runs, operations, layers, logger)

    logger.info(f"\nAll comparison figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
