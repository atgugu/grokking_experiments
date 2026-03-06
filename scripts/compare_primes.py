#!/usr/bin/env python
"""Cross-run comparison plots for the prime p sweep.

Loads metrics.json from all sweep runs and produces:
1. Scaling law: log(grok_epoch) vs log(p) — the key scaling curve
2. Key frequency / p ratio per prime — do they cluster near p/4, p/3?
3. Number of key frequencies vs p — constant ~5 or scaling?
4. Gini coefficient vs p (final + evolution curves)
5. Test accuracy curves per prime (all on one plot)
6. Frequency fingerprint heatmap (one row per p)
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logging

SWEEP_PRIMES = [7, 11, 13, 17, 23, 31, 43, 59, 67, 89, 97, 113]


def _run_id_for(p):
    return f"p{p}_d128_h4_mlp512_L1_wd1.0_s42"


def load_all_runs(results_root, primes, logger):
    """Load metrics for all primes. Returns dict {p: metrics_dict}."""
    runs = {}
    for p in primes:
        rid = _run_id_for(p)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing: {metrics_path}")
            continue
        with open(metrics_path) as f:
            runs[p] = json.load(f)
        logger.info(f"Loaded p={p}: {rid}")
    return runs


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def plot_scaling_law(runs, primes, fig_dir):
    """Figure 1: Grokking epoch vs p on log-log scale with power-law fit."""
    available = [(p, runs[p]) for p in primes if p in runs]
    grok_data = []
    for p, metrics in available:
        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        if grok_epoch is not None:
            grok_data.append((p, grok_epoch))

    if len(grok_data) < 2:
        print("  Skipping scaling law: not enough grokked runs")
        return

    ps = np.array([d[0] for d in grok_data])
    epochs = np.array([d[1] for d in grok_data])

    # Fit log-log line
    log_p = np.log10(ps)
    log_e = np.log10(epochs)
    coeffs = np.polyfit(log_p, log_e, 1)
    exponent = coeffs[0]
    fit_line = 10 ** np.polyval(coeffs, log_p)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(ps, epochs, s=80, zorder=5, color="#1f77b4", label="Grokked runs")

    # Annotate each point with p value
    for p, epoch in grok_data:
        ax.annotate(f"p={p}", (p, epoch), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#333333")

    ax.plot(ps, fit_line, "r--", linewidth=1.5,
            label=f"Power-law fit: epoch ~ p^{exponent:.2f}")

    # Mark never-grokked
    never = [(p, metrics.get("final_test_acc", 0)) for p, metrics in available
             if find_grokking_epoch(metrics.get("history", {})) is None]
    if never:
        ps_never = [d[0] for d in never]
        # Place them at the top of the plot as triangles
        ymax = epochs.max() * 2 if len(epochs) > 0 else 40000
        ax.scatter(ps_never, [ymax] * len(ps_never), marker="v", s=100,
                   color="#e74c3c", zorder=5, label="Never grokked (plotted at top)")
        for p, _ in never:
            ax.annotate(f"p={p}", (p, ymax), textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#e74c3c")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Prime modulus p", fontsize=12)
    ax.set_ylabel("Grokking Epoch (test acc > 95%)", fontsize=12)
    ax.set_title(f"Grokking Scaling Law: epoch ~ p^{exponent:.2f}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_scaling_law.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: primes_scaling_law.png (exponent={exponent:.2f})")


def plot_key_frequency_ratios(runs, primes, fig_dir):
    """Figure 2: Key frequencies / p for each prime — do they follow universal ratios?

    Each prime gets one row; horizontal axis = frequency / p (normalized to [0, 1]).
    """
    available = [p for p in primes if p in runs]
    if not available:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(available) * 0.5 + 1)))

    cmap = cm.get_cmap("tab10")
    reference_ratios = [1/6, 1/5, 1/4, 1/3, 2/5, 1/2]

    for yi, p in enumerate(available):
        metrics = runs[p]
        key_freqs = metrics.get("final_key_frequencies", [])
        if not key_freqs:
            continue
        ratios = [f / p for f in key_freqs]
        color = cmap(yi % 10)
        ax.scatter(ratios, [yi] * len(ratios), color=color, s=80, zorder=4,
                   label=f"p={p}" if yi < 12 else None)
        for ratio in ratios:
            ax.annotate(f"{ratio:.2f}", (ratio, yi), textcoords="offset points",
                        xytext=(0, 5), fontsize=6, ha="center", color=color)

    # Reference vertical lines
    ref_labels = ["1/6", "1/5", "1/4", "1/3", "2/5", "1/2"]
    for ratio, label in zip(reference_ratios, ref_labels):
        ax.axvline(ratio, color="gray", linestyle="--", alpha=0.4, linewidth=1)
        ax.text(ratio, len(available) + 0.1, label, ha="center", fontsize=7,
                color="gray", rotation=0)

    ax.set_yticks(range(len(available)))
    ax.set_yticklabels([f"p={p}" for p in available], fontsize=10)
    ax.set_xlabel("Key frequency / p", fontsize=11)
    ax.set_title("Key Frequency Ratios (freq/p) Across Primes\n(dashed = canonical ratios 1/6, 1/4, 1/3, 1/2)",
                 fontsize=12, fontweight="bold")
    ax.set_xlim(0, 0.55)
    ax.set_ylim(-0.5, len(available) - 0.5)
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_key_freq_ratios.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: primes_key_freq_ratios.png")


def plot_n_key_frequencies(runs, primes, fig_dir):
    """Figure 3: Number of key frequencies vs p."""
    available = [p for p in primes if p in runs]
    if not available:
        return

    n_freqs = []
    grokked = []
    for p in available:
        metrics = runs[p]
        key_freqs = metrics.get("final_key_frequencies", [])
        n_freqs.append(len(key_freqs))
        history = metrics.get("history", {})
        grokked.append(find_grokking_epoch(history) is not None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: n_freqs vs p
    ax = axes[0]
    colors = ["#2ca02c" if g else "#e74c3c" for g in grokked]
    bars = ax.bar(range(len(available)), n_freqs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([f"p={p}" for p in available], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Number of Key Frequencies", fontsize=11)
    ax.set_title("Number of Key Frequencies vs Prime p", fontsize=12, fontweight="bold")
    ax.axhline(5, color="gray", linestyle="--", alpha=0.5, label="Nanda baseline (5)")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#2ca02c", label="Grokked"),
                  Patch(facecolor="#e74c3c", label="Never grokked")]
    ax.legend(handles=legend_els, fontsize=9)

    # Right: n_freqs vs log(p) with scatter + optional fit
    ax2 = axes[1]
    ps_arr = np.array(available)
    nf_arr = np.array(n_freqs)
    g_arr = np.array(grokked)
    ax2.scatter(ps_arr[g_arr], nf_arr[g_arr], s=80, color="#2ca02c", zorder=4, label="Grokked")
    ax2.scatter(ps_arr[~g_arr], nf_arr[~g_arr], s=80, color="#e74c3c", marker="x", zorder=4, label="Never grokked")
    for p, nf in zip(available, n_freqs):
        ax2.annotate(str(p), (p, nf), textcoords="offset points", xytext=(4, 3), fontsize=8)
    ax2.set_xscale("log")
    ax2.set_xlabel("Prime p (log scale)", fontsize=11)
    ax2.set_ylabel("Number of Key Frequencies", fontsize=11)
    ax2.set_title("Key Frequency Count vs p (log scale)", fontsize=12, fontweight="bold")
    ax2.axhline(5, color="gray", linestyle="--", alpha=0.5, label="Nanda baseline (5)")
    ax2.legend(fontsize=9)
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_n_key_frequencies.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: primes_n_key_frequencies.png")


def plot_gini_vs_p(runs, primes, fig_dir):
    """Figure 4a: Final Gini vs p bar chart.
     Figure 4b: Gini evolution curves per prime.
    """
    available = [p for p in primes if p in runs]
    if not available:
        return

    # --- 4a: bar chart ---
    ginis = [runs[p].get("final_gini", 0) for p in available]
    grokked = [find_grokking_epoch(runs[p].get("history", {})) is not None for p in available]

    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ["#2ca02c" if g else "#e74c3c" for g in grokked]
    ax.bar(range(len(available)), ginis, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([f"p={p}" for p in available], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Final Gini Coefficient", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fourier Sparsity (Gini) vs Prime p\n(green=grokked, red=never grokked)",
                 fontsize=12, fontweight="bold")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="High-sparsity threshold")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_gini_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: primes_gini_bar.png")

    # --- 4b: Gini evolution ---
    cmap = cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(available):
        metrics = runs[p]
        history = metrics.get("history", {})
        fourier_epochs = history.get("fourier_epochs", [])
        gini = history.get("gini", [])
        if fourier_epochs and gini:
            color = cmap(i / max(len(available) - 1, 1))
            ax.plot(fourier_epochs, gini, color=color, label=f"p={p}", linewidth=1.5, alpha=0.85)

    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Gini Coefficient", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Fourier Sparsity Evolution per Prime", fontsize=12, fontweight="bold")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_gini_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: primes_gini_evolution.png")


def plot_test_accuracy_curves(runs, primes, fig_dir):
    """Figure 5: Test accuracy curves for all primes overlaid."""
    available = [p for p in primes if p in runs]
    if not available:
        return

    cmap = cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, p in enumerate(available):
        metrics = runs[p]
        history = metrics.get("history", {})
        eval_epochs = history.get("eval_epochs", [])
        test_acc = history.get("test_acc", [])
        if eval_epochs and test_acc:
            color = cmap(i / max(len(available) - 1, 1))
            ax.plot(eval_epochs, test_acc, color=color, label=f"p={p}", linewidth=1.5, alpha=0.85)

    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Test Accuracy Curves Across Primes", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: primes_test_accuracy.png")


def plot_frequency_fingerprints(runs, primes, fig_dir):
    """Figure 6: Frequency fingerprint heatmap.

    Each row = one prime p.
    Columns = frequency index k (0..p-1), normalized by p so x-axis is [0,1].
    Color = 2D Fourier power at frequency k (summed over the second dimension).
    """
    available = [p for p in primes if p in runs]
    if not available:
        return

    # We'll use a fixed-width grid and interpolate (nearest) so all rows line up.
    n_cols = 200  # normalized grid
    n_rows = len(available)

    heatmap = np.zeros((n_rows, n_cols))
    grokked = []

    for i, p in enumerate(available):
        metrics = runs[p]
        history = metrics.get("history", {})
        # Try to get key_freq_norms from the last fourier snapshot if available
        # Otherwise use key_frequencies as a sparse signal
        key_freqs = metrics.get("final_key_frequencies", [])
        row = np.zeros(n_cols)
        for f in key_freqs:
            col_idx = int(f / p * n_cols)
            col_idx = min(col_idx, n_cols - 1)
            row[col_idx] += 1.0
        heatmap[i] = row
        grokked.append(find_grokking_epoch(history) is not None)

    fig, ax = plt.subplots(figsize=(12, max(4, n_rows * 0.5 + 1)))

    # Normalize each row for visibility
    row_max = heatmap.max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1, row_max)
    heatmap_norm = heatmap / row_max

    im = ax.imshow(heatmap_norm, aspect="auto", cmap="YlOrRd",
                   extent=[0, 1, n_rows - 0.5, -0.5], vmin=0, vmax=1)

    # Reference lines
    for ratio, label in [(1/6, "p/6"), (1/4, "p/4"), (1/3, "p/3"), (1/2, "p/2")]:
        ax.axvline(ratio, color="white", linestyle="--", alpha=0.6, linewidth=1)
        ax.text(ratio, -0.7, label, ha="center", fontsize=8, color="white")

    ax.set_yticks(range(n_rows))
    row_labels = [f"p={p} {'*' if g else ''}" for p, g in zip(available, grokked)]
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_xlabel("Normalized frequency (f / p)", fontsize=11)
    ax.set_title("Frequency Fingerprints per Prime\n(* = grokked; dashed = canonical ratios p/6, p/4, p/3, p/2)",
                 fontsize=12, fontweight="bold")

    plt.colorbar(im, ax=ax, label="Relative frequency power")

    fig.tight_layout()
    fig.savefig(fig_dir / "primes_frequency_fingerprints.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: primes_frequency_fingerprints.png")


def print_prime_summary(runs, primes, logger):
    """Console summary table."""
    print("\n" + "=" * 95)
    print("PRIME p SWEEP — SUMMARY")
    print("=" * 95)
    header = f"{'p':>5} | {'Pairs':>5} | {'N_train':>7} | {'Grok Epoch':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | {'N_freqs':>7} | Key Freqs"
    print(header)
    print("-" * 95)

    grok_ps = []
    grok_epochs = []
    for p in primes:
        if p not in runs:
            print(f"{p:>5} | {'—':>5} | {'—':>7} | {'MISSING':>10} | {'—':>9} | {'—':>8} | {'—':>6} | {'—':>7} | —")
            continue
        metrics = runs[p]
        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        n_pairs = p * p
        n_train = int(n_pairs * 0.3)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
        train_acc = metrics.get("final_train_acc", 0)
        test_acc = metrics.get("final_test_acc", 0)
        gini = metrics.get("final_gini", 0)
        key_freqs = metrics.get("final_key_frequencies", [])
        n_freqs = len(key_freqs)

        if grok_epoch is not None:
            grok_ps.append(p)
            grok_epochs.append(grok_epoch)

        print(f"{p:>5} | {n_pairs:>5} | {n_train:>7} | {grok_str:>10} | {train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {n_freqs:>7} | {key_freqs}")

    print("=" * 95)

    # Scaling law estimate
    if len(grok_ps) >= 3:
        log_p = np.log10(grok_ps)
        log_e = np.log10(grok_epochs)
        coeffs = np.polyfit(log_p, log_e, 1)
        print(f"\nScaling law fit (n={len(grok_ps)} grokked primes): grok_epoch ~ p^{coeffs[0]:.2f}")
        print(f"  (log10 intercept = {coeffs[1]:.2f}, so epoch ≈ {10**coeffs[1]:.1f} * p^{coeffs[0]:.2f})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare prime p sweep results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/primes_comparison/)")
    parser.add_argument("--primes", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "primes_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, primes, logger)
    if not runs:
        logger.error("No runs found! Run the prime sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(primes)} runs")

    logger.info("1/6: Scaling law plot")
    plot_scaling_law(runs, primes, fig_dir)

    logger.info("2/6: Key frequency ratios")
    plot_key_frequency_ratios(runs, primes, fig_dir)

    logger.info("3/6: Number of key frequencies")
    plot_n_key_frequencies(runs, primes, fig_dir)

    logger.info("4/6: Gini vs p")
    plot_gini_vs_p(runs, primes, fig_dir)

    logger.info("5/6: Test accuracy curves")
    plot_test_accuracy_curves(runs, primes, fig_dir)

    logger.info("6/6: Frequency fingerprints")
    plot_frequency_fingerprints(runs, primes, fig_dir)

    print_prime_summary(runs, primes, logger)

    logger.info(f"\nAll comparison figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
