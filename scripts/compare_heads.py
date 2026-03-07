#!/usr/bin/env python
"""Cross-run comparison plots for the attention head count sweep.

Loads metrics.json (and fourier_snapshots.npz where available) from all sweep
runs and produces:
1. Test accuracy curves for all n_heads values
2. n_key_freq vs n_heads — the core mechanistic plot (energy-fraction threshold)
3. Gini coefficient vs n_heads
4. Grokking epoch vs n_heads
5. Fourier spectrum strip for each n_heads

Key design decision: intrinsic frequency count is determined by energy-fraction
threshold (min K such that top-K freqs hold ≥90% of non-DC energy), NOT the
hardcoded n_top=5 from metrics.json. This avoids the artifact of always returning 5.
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

from src.analysis.fourier import identify_key_frequencies, compute_gini_coefficient
from src.utils import setup_logging

SWEEP_HEADS = [1, 2, 4, 8, 16]
ENERGY_THRESHOLD = 0.90  # fraction of non-DC energy to capture


def _run_id_for(n_heads):
    return f"p113_d128_h{n_heads}_mlp512_L1_wd1.0_s42"


def load_all_runs(results_root, heads, logger):
    """Load metrics and optionally fourier snapshots for all n_heads values."""
    runs = {}
    for n_heads in heads:
        rid = _run_id_for(n_heads)
        run_dir = results_root / rid
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing: {metrics_path}")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)

        # Try to load fourier snapshots for richer frequency analysis
        snap_path = run_dir / "fourier_snapshots.npz"
        if snap_path.exists():
            metrics["_fourier_snapshots"] = dict(np.load(snap_path, allow_pickle=True))
            logger.info(f"Loaded n_heads={n_heads}: {rid} (+ fourier_snapshots)")
        else:
            logger.info(f"Loaded n_heads={n_heads}: {rid} (no fourier_snapshots)")

        runs[n_heads] = metrics
    return runs


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def compute_intrinsic_n_key_freq(frequency_norms, threshold=ENERGY_THRESHOLD):
    """Compute intrinsic frequency count via energy-fraction threshold.

    Returns the minimum K such that the top-K non-DC frequencies hold
    >= threshold fraction of total non-DC energy.

    Also returns top-10 frequencies for full spectrum analysis.
    """
    norms = np.array(frequency_norms, dtype=float)
    norms[0] = 0.0  # zero out DC

    total_energy = norms.sum()
    if total_energy == 0:
        return 0, []

    # Sort descending
    sorted_indices = np.argsort(norms)[::-1]
    sorted_norms = norms[sorted_indices]
    cumulative = np.cumsum(sorted_norms)

    # Find min K for threshold
    k_threshold = int(np.searchsorted(cumulative, threshold * total_energy)) + 1
    k_threshold = min(k_threshold, len(norms))

    # Top-10 frequencies
    top10 = sorted_indices[:10].tolist()

    return k_threshold, top10


def get_final_frequency_norms(metrics, n_heads, logger):
    """Extract final frequency_norms from fourier_snapshots or fall back to metrics.

    Returns (frequency_norms array or None, source_description).
    """
    snaps = metrics.get("_fourier_snapshots")
    if snaps is not None and "frequency_norms" in snaps:
        fn = np.array(snaps["frequency_norms"])
        if fn.ndim == 2 and len(fn) > 0:
            # Shape: (n_snapshots, p) — take the last snapshot
            return fn[-1], "fourier_snapshots (last)"
        elif fn.ndim == 1:
            return fn, "fourier_snapshots"

    # Fall back to history gini/key_freqs from metrics
    history = metrics.get("history", {})
    freq_norms_history = history.get("frequency_norms", [])
    if freq_norms_history:
        fn = np.array(freq_norms_history[-1]) if isinstance(freq_norms_history[0], list) else np.array(freq_norms_history)
        return fn, "metrics history"

    logger.warning(f"n_heads={n_heads}: no frequency_norms found; using key_frequencies from metrics")
    return None, "none"


def plot_test_accuracy_curves(runs, heads, fig_dir):
    """Figure 1: Test accuracy over time for all n_heads values."""
    available = [h for h in heads if h in runs]
    if not available:
        return

    cmap = cm.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, n_heads in enumerate(available):
        metrics = runs[n_heads]
        history = metrics.get("history", {})
        eval_epochs = history.get("eval_epochs", [])
        test_acc = history.get("test_acc", [])
        if eval_epochs and test_acc:
            color = cmap(i / max(len(available) - 1, 1))
            d_head = 128 // n_heads
            ax.plot(eval_epochs, test_acc, color=color,
                    label=f"h={n_heads} (d_head={d_head})", linewidth=1.8, alpha=0.9)

    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Test Accuracy Curves by n_heads\n(p=113, wd=1.0, lr=1e-3)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "heads_test_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heads_test_accuracy.png")


def plot_n_key_freq_vs_heads(runs, heads, fig_dir, logger):
    """Figure 2: n_key_freq vs n_heads — the core mechanistic plot.

    Uses energy-fraction threshold (not hardcoded n_top=5) to find intrinsic
    frequency count. Also overlays n_top=10 raw results for comparison.
    """
    available = [h for h in heads if h in runs]
    if not available:
        return

    n_heads_arr = []
    k_threshold_arr = []
    k_top10_arr = []
    grokked_arr = []

    for n_heads in available:
        metrics = runs[n_heads]
        history = metrics.get("history", {})
        grokked = find_grokking_epoch(history) is not None
        grokked_arr.append(grokked)

        freq_norms, source = get_final_frequency_norms(metrics, n_heads, logger)

        if freq_norms is not None:
            k_thr, top10 = compute_intrinsic_n_key_freq(freq_norms, ENERGY_THRESHOLD)
        else:
            # Fall back to len(key_freqs) from metrics
            key_freqs = metrics.get("final_key_frequencies", [])
            k_thr = len(key_freqs)
            top10 = list(key_freqs)[:10]

        n_heads_arr.append(n_heads)
        k_threshold_arr.append(k_thr)
        # Count unique freqs in top10 that are above noise (excluding p-f mirror)
        p = 113
        unique_freqs = set()
        for f in top10:
            f = int(f)
            if f != 0:
                unique_freqs.add(min(f, p - f))
        k_top10_arr.append(len(unique_freqs))

        logger.info(f"n_heads={n_heads}: k_threshold={k_thr}, unique_top10={len(unique_freqs)}, "
                    f"grokked={grokked}, source={source}")

    n_heads_np = np.array(n_heads_arr)
    k_thr_np = np.array(k_threshold_arr)
    k_top10_np = np.array(k_top10_arr)
    grokked_np = np.array(grokked_arr)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: energy-threshold method
    ax = axes[0]
    colors = ["#2ca02c" if g else "#e74c3c" for g in grokked_np]
    ax.scatter(n_heads_np, k_thr_np, s=120, color=colors, zorder=5, edgecolors="black", linewidth=0.8)
    for nh, kv, g in zip(n_heads_np, k_thr_np, grokked_np):
        ax.annotate(str(kv), (nh, kv), textcoords="offset points", xytext=(8, 4), fontsize=10)

    # Reference lines
    ax.plot(n_heads_np, n_heads_np, "b--", alpha=0.5, linewidth=1.5, label="n_key_freq = n_heads (Nanda)")
    ax.axhline(5, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="n_key_freq = 5 (task-determined)")

    ax.set_xlabel("n_heads", fontsize=11)
    ax.set_ylabel(f"Intrinsic n_key_freq (≥{int(ENERGY_THRESHOLD*100)}% energy)", fontsize=11)
    ax.set_title(f"Key Frequency Count vs n_heads\n(energy threshold: {int(ENERGY_THRESHOLD*100)}% non-DC energy)",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(n_heads_np)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor="#2ca02c", label="Grokked"),
                  Patch(facecolor="#e74c3c", label="Never grokked")]
    ax2leg = ax.legend(handles=legend_els + ax.get_legend_handles_labels()[0],
                       fontsize=8, loc="upper left")

    # Right: unique frequencies in top-10 (complementary view)
    ax2 = axes[1]
    ax2.scatter(n_heads_np, k_top10_np, s=120, color=colors, zorder=5, edgecolors="black", linewidth=0.8)
    for nh, kv in zip(n_heads_np, k_top10_np):
        ax2.annotate(str(kv), (nh, kv), textcoords="offset points", xytext=(8, 4), fontsize=10)

    ax2.plot(n_heads_np, n_heads_np, "b--", alpha=0.5, linewidth=1.5, label="n_key_freq = n_heads")
    ax2.axhline(5, color="orange", linestyle="--", alpha=0.7, linewidth=1.5, label="n_key_freq = 5")

    ax2.set_xlabel("n_heads", fontsize=11)
    ax2.set_ylabel("Unique frequencies in top-10 (excl. mirrors)", fontsize=11)
    ax2.set_title("Unique Key Frequencies (top-10, de-mirrored)\nvs n_heads",
                  fontsize=11, fontweight="bold")
    ax2.set_xticks(n_heads_np)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "heads_n_key_freq.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heads_n_key_freq.png")


def plot_gini_vs_heads(runs, heads, fig_dir):
    """Figure 3: Gini coefficient vs n_heads."""
    available = [h for h in heads if h in runs]
    if not available:
        return

    ginis = [runs[h].get("final_gini", 0) for h in available]
    grokked = [find_grokking_epoch(runs[h].get("history", {})) is not None for h in available]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: final Gini bar chart
    ax = axes[0]
    colors = ["#2ca02c" if g else "#e74c3c" for g in grokked]
    ax.bar(range(len(available)), ginis, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(available)))
    ax.set_xticklabels([f"h={h}" for h in available], fontsize=10)
    ax.set_ylabel("Final Gini Coefficient", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Fourier Sparsity (Gini) vs n_heads\n(green=grokked, red=never grokked)",
                 fontsize=11, fontweight="bold")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, axis="y", alpha=0.3)

    # Right: Gini evolution curves
    cmap = cm.get_cmap("viridis")
    ax2 = axes[1]
    for i, n_heads in enumerate(available):
        metrics = runs[n_heads]
        history = metrics.get("history", {})
        fourier_epochs = history.get("fourier_epochs", [])
        gini_hist = history.get("gini", [])
        if fourier_epochs and gini_hist:
            color = cmap(i / max(len(available) - 1, 1))
            d_head = 128 // n_heads
            ax2.plot(fourier_epochs, gini_hist, color=color,
                     label=f"h={n_heads} (d_head={d_head})", linewidth=1.8, alpha=0.85)

    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Gini Coefficient", fontsize=11)
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Gini Evolution per n_heads", fontsize=11, fontweight="bold")
    ax2.axhline(0.95, color="gray", linestyle="--", alpha=0.4)
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "heads_gini.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heads_gini.png")


def plot_grokking_epoch_vs_heads(runs, heads, fig_dir):
    """Figure 4: Grokking epoch vs n_heads."""
    available = [h for h in heads if h in runs]
    if not available:
        return

    grok_epochs = []
    grokked = []
    for n_heads in available:
        metrics = runs[n_heads]
        history = metrics.get("history", {})
        ge = find_grokking_epoch(history)
        grok_epochs.append(ge)
        grokked.append(ge is not None)

    fig, ax = plt.subplots(figsize=(8, 5))

    grok_nh = [nh for nh, ge in zip(available, grok_epochs) if ge is not None]
    grok_ep = [ge for ge in grok_epochs if ge is not None]
    never_nh = [nh for nh, ge in zip(available, grok_epochs) if ge is None]

    if grok_nh:
        ax.scatter(grok_nh, grok_ep, s=120, color="#2ca02c", zorder=5,
                   edgecolors="black", linewidth=0.8, label="Grokked")
        for nh, ep in zip(grok_nh, grok_ep):
            ax.annotate(f"{ep:,}", (nh, ep), textcoords="offset points",
                        xytext=(8, 4), fontsize=9)

    if never_nh:
        ymax = max(grok_ep) * 1.5 if grok_ep else 40000
        ax.scatter(never_nh, [ymax] * len(never_nh), marker="v", s=120,
                   color="#e74c3c", zorder=5, edgecolors="black",
                   linewidth=0.8, label="Never grokked (plotted at top)")
        for nh in never_nh:
            ax.annotate(f"h={nh}", (nh, ymax), textcoords="offset points",
                        xytext=(8, 4), fontsize=9, color="#e74c3c")

    ax.set_xlabel("n_heads", fontsize=11)
    ax.set_ylabel("Grokking Epoch (test acc > 95%)", fontsize=11)
    ax.set_title("Grokking Speed vs n_heads\n(p=113, wd=1.0, lr=1e-3)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(available)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "heads_grokking_epoch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heads_grokking_epoch.png")


def plot_fourier_spectrum_strips(runs, heads, fig_dir, logger):
    """Figure 5: Fourier spectrum strip (bar chart) for each n_heads.

    One subplot per n_heads. X-axis = frequency index, Y-axis = normalized energy.
    Shows which frequencies dominate and how many are above noise floor.
    """
    available = [h for h in heads if h in runs]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), squeeze=False)

    cmap = cm.get_cmap("viridis")
    p = 113

    for i, n_heads in enumerate(available):
        ax = axes[i, 0]
        metrics = runs[n_heads]
        history = metrics.get("history", {})
        grokked = find_grokking_epoch(history) is not None
        d_head = 128 // n_heads
        color_base = cmap(i / max(n - 1, 1))

        freq_norms, source = get_final_frequency_norms(metrics, n_heads, logger)

        if freq_norms is not None:
            fn = np.array(freq_norms, dtype=float)
            fn[0] = 0.0  # zero DC for display
            fn_normalized = fn / fn.max() if fn.max() > 0 else fn

            # Find top-10 freqs
            top10_idx = np.argsort(fn)[::-1][:10]
            bar_colors = ["#e74c3c" if j in top10_idx else "#aaaaaa" for j in range(len(fn))]
            ax.bar(range(len(fn)), fn_normalized, color=bar_colors, alpha=0.8, linewidth=0)

            # Mark the 90% energy threshold
            k_thr, _ = compute_intrinsic_n_key_freq(np.array(freq_norms), ENERGY_THRESHOLD)
            threshold_freq = np.sort(fn)[::-1][k_thr - 1] / fn.max() if fn.max() > 0 else 0
            ax.axhline(threshold_freq, color="blue", linestyle="--", alpha=0.6,
                       linewidth=1, label=f"90% energy threshold (K={k_thr})")
        else:
            key_freqs = metrics.get("final_key_frequencies", [])
            bar_colors = ["#e74c3c" if j in key_freqs else "#aaaaaa" for j in range(p)]
            heights = [1.0 if j in key_freqs else 0.1 for j in range(p)]
            ax.bar(range(p), heights, color=bar_colors, alpha=0.8, linewidth=0)

        grok_str = "GROKKED" if grokked else "not grokked"
        ax.set_title(f"n_heads={n_heads} (d_head={d_head}) — {grok_str}", fontsize=10, fontweight="bold")
        ax.set_xlim(-1, p)
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Norm. energy", fontsize=8)
        if i == n - 1:
            ax.set_xlabel("Frequency index k", fontsize=10)
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, axis="y", alpha=0.2)

        # Annotate top frequencies
        if freq_norms is not None and fn.max() > 0:
            for f_idx in np.argsort(fn)[::-1][:5]:
                if fn_normalized[f_idx] > 0.3:
                    ax.annotate(str(f_idx), (f_idx, fn_normalized[f_idx]),
                                textcoords="offset points", xytext=(0, 3),
                                fontsize=6, ha="center", color="#c0392b")

    fig.suptitle("Fourier Spectrum per n_heads (red bars = top-10 freqs)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(fig_dir / "heads_spectrum_strips.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: heads_spectrum_strips.png")


def print_heads_summary(runs, heads, logger):
    """Console summary table."""
    print("\n" + "=" * 100)
    print("ATTENTION HEAD SWEEP — SUMMARY")
    print("=" * 100)
    header = (f"{'n_heads':>7} | {'d_head':>6} | {'Grok Epoch':>10} | "
              f"{'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | "
              f"{'K (90%)':>7} | {'K (top10)':>9} | Key Freqs (saved)")
    print(header)
    print("-" * 100)

    for n_heads in heads:
        if n_heads not in runs:
            d_head = 128 // n_heads
            print(f"{n_heads:>7} | {d_head:>6} | {'MISSING':>10} | {'—':>9} | {'—':>8} | {'—':>6} | {'—':>7} | {'—':>9} | —")
            continue

        metrics = runs[n_heads]
        history = metrics.get("history", {})
        d_head = 128 // n_heads
        grok_epoch = find_grokking_epoch(history)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
        train_acc = metrics.get("final_train_acc", 0)
        test_acc = metrics.get("final_test_acc", 0)
        gini = metrics.get("final_gini", 0)
        key_freqs_saved = metrics.get("final_key_frequencies", [])

        freq_norms, _ = get_final_frequency_norms(metrics, n_heads, logger)
        if freq_norms is not None:
            k_thr, top10 = compute_intrinsic_n_key_freq(np.array(freq_norms))
            p = 113
            unique = set()
            for f in top10:
                f = int(f)
                if f != 0:
                    unique.add(min(f, p - f))
            k_top10 = len(unique)
        else:
            k_thr = len(key_freqs_saved)
            k_top10 = len(key_freqs_saved)

        print(f"{n_heads:>7} | {d_head:>6} | {grok_str:>10} | {train_acc:>9.4f} | "
              f"{test_acc:>8.4f} | {gini:>6.3f} | {k_thr:>7} | {k_top10:>9} | {key_freqs_saved}")

    print("=" * 100)
    print()
    print("Interpretation guide:")
    print(f"  K (90%):   min K s.t. top-K freqs hold ≥90% of non-DC Fourier energy")
    print(f"  K (top10): unique frequencies in top-10 after de-mirroring (f and p-f counted once)")
    print(f"  Key Freqs: saved n_top=5 value from metrics.json (artifact-prone)")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare attention head sweep results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for figures (default: results/heads_comparison/)")
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    heads = args.heads if args.heads else SWEEP_HEADS
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "heads_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, heads, logger)
    if not runs:
        logger.error("No runs found! Run the heads sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(heads)} runs")

    logger.info("1/5: Test accuracy curves")
    plot_test_accuracy_curves(runs, heads, fig_dir)

    logger.info("2/5: n_key_freq vs n_heads (core mechanistic plot)")
    plot_n_key_freq_vs_heads(runs, heads, fig_dir, logger)

    logger.info("3/5: Gini vs n_heads")
    plot_gini_vs_heads(runs, heads, fig_dir)

    logger.info("4/5: Grokking epoch vs n_heads")
    plot_grokking_epoch_vs_heads(runs, heads, fig_dir)

    logger.info("5/5: Fourier spectrum strips")
    plot_fourier_spectrum_strips(runs, heads, fig_dir, logger)

    print_heads_summary(runs, heads, logger)

    logger.info(f"\nAll comparison figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
