#!/usr/bin/env python
"""Cross-run comparison for the joint (n_heads x p) sweep.

Loads metrics.json (and fourier_snapshots.npz) from all 32 cells and produces:
1. K(p, h) heatmap — intrinsic frequency count
2. K vs log2(p) per head count — tests O(log p) scaling
3. Grokking epoch heatmap (log colorscale)
4. Phase boundary plot — largest non-grokking p per h
5. Gini heatmap
6. Test accuracy small-multiples grid
7. Grokking speed interaction — grok_epoch vs h per prime

Also saves joint_sweep_metrics.json with all per-cell metrics.
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

from src.utils import setup_logging

SWEEP_PRIMES = [13, 23, 31, 43, 59, 67, 89, 113]
SWEEP_HEADS = [1, 2, 4, 8]
ENERGY_THRESHOLD = 0.90


def _run_id_for(p, n_heads):
    return f"p{p}_d128_h{n_heads}_mlp512_L1_wd1.0_s42"


def load_all_runs(results_root, primes, heads, logger):
    """Load metrics for all (p, n_heads) cells. Returns dict keyed by (p, h)."""
    runs = {}
    for p in primes:
        for h in heads:
            rid = _run_id_for(p, h)
            metrics_path = results_root / rid / "metrics.json"
            if not metrics_path.exists():
                continue
            with open(metrics_path) as f:
                metrics = json.load(f)

            # Try to load fourier snapshots
            snap_path = results_root / rid / "fourier_snapshots.npz"
            if snap_path.exists():
                metrics["_fourier_snapshots"] = dict(np.load(snap_path, allow_pickle=True))

            runs[(p, h)] = metrics
            logger.info(f"Loaded p={p} h={h}: {rid}")
    return runs


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def get_final_frequency_norms(metrics, p, logger):
    """Extract final frequency_norms. Returns (array or None, source)."""
    snaps = metrics.get("_fourier_snapshots")
    if snaps is not None and "frequency_norms" in snaps:
        fn = np.array(snaps["frequency_norms"])
        if fn.ndim == 2 and len(fn) > 0:
            return fn[-1], "fourier_snapshots (last)"
        elif fn.ndim == 1:
            return fn, "fourier_snapshots"

    history = metrics.get("history", {})
    freq_norms_history = history.get("frequency_norms", [])
    if freq_norms_history:
        fn = np.array(freq_norms_history[-1]) if isinstance(freq_norms_history[0], list) else np.array(freq_norms_history)
        return fn, "metrics history"

    return None, "none"


def compute_intrinsic_n_key_freq(frequency_norms, threshold=ENERGY_THRESHOLD):
    """Min K such that top-K non-DC freqs hold >= threshold of non-DC energy.

    Returns (K_energy, top10_indices).
    """
    norms = np.array(frequency_norms, dtype=float)
    norms[0] = 0.0  # zero out DC

    total_energy = norms.sum()
    if total_energy == 0:
        return 0, []

    sorted_indices = np.argsort(norms)[::-1]
    sorted_norms = norms[sorted_indices]
    cumulative = np.cumsum(sorted_norms)

    k_threshold = int(np.searchsorted(cumulative, threshold * total_energy)) + 1
    k_threshold = min(k_threshold, len(norms))

    top10 = sorted_indices[:10].tolist()
    return k_threshold, top10


def compute_k_unique(top10, p):
    """De-mirrored unique frequency count from top-10 list."""
    unique = set()
    for f in top10:
        f = int(f)
        if f != 0:
            unique.add(min(f, p - f))
    return len(unique)


def extract_cell_metrics(runs, primes, heads, logger):
    """Extract all metrics for each cell. Returns dict keyed by (p, h)."""
    cells = {}
    for p in primes:
        for h in heads:
            if (p, h) not in runs:
                continue

            metrics = runs[(p, h)]
            history = metrics.get("history", {})
            grok_epoch = find_grokking_epoch(history)
            grokked = grok_epoch is not None
            test_acc = metrics.get("final_test_acc", 0)
            train_acc = metrics.get("final_train_acc", 0)
            gini = metrics.get("final_gini", 0)

            freq_norms, source = get_final_frequency_norms(metrics, p, logger)
            if freq_norms is not None:
                k_energy, top10 = compute_intrinsic_n_key_freq(freq_norms)
                k_unique = compute_k_unique(top10, p)
            else:
                key_freqs = metrics.get("final_key_frequencies", [])
                k_energy = len(key_freqs)
                k_unique = compute_k_unique(key_freqs, p)
                top10 = list(key_freqs)[:10]

            max_k = p // 2  # ceiling for K at this p

            cells[(p, h)] = {
                "grok_epoch": grok_epoch,
                "grokked": grokked,
                "test_acc": test_acc,
                "train_acc": train_acc,
                "gini": gini,
                "k_energy": k_energy,
                "k_unique": k_unique,
                "max_k": max_k,
                "top10": top10,
            }

    return cells


# ---- Figure 1: K heatmap ----

def plot_K_heatmap(cells, primes, heads, fig_dir):
    """K(p, h) heatmap with annotations. Non-grokked cells hatched."""
    n_rows, n_cols = len(primes), len(heads)
    data = np.full((n_rows, n_cols), np.nan)
    annotations = [['' for _ in range(n_cols)] for _ in range(n_rows)]
    mask_not_grokked = np.zeros((n_rows, n_cols), dtype=bool)

    for i, p in enumerate(primes):
        for j, h in enumerate(heads):
            if (p, h) in cells:
                c = cells[(p, h)]
                data[i, j] = c["k_energy"]
                annotations[i][j] = str(c["k_energy"])
                if not c["grokked"]:
                    mask_not_grokked[i, j] = True
                    annotations[i][j] += "*"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=np.nanmax(data) + 1)

    # Annotate cells
    for i in range(n_rows):
        for j in range(n_cols):
            if not np.isnan(data[i, j]):
                color = "white" if data[i, j] > np.nanmax(data) * 0.6 else "black"
                ax.text(j, i, annotations[i][j], ha="center", va="center",
                        fontsize=11, fontweight="bold", color=color)
            else:
                ax.text(j, i, "?", ha="center", va="center", fontsize=10,
                        color="gray")

    # Hatch non-grokked cells
    for i in range(n_rows):
        for j in range(n_cols):
            if mask_not_grokked[i, j]:
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                             fill=False, hatch="///", edgecolor="gray",
                             linewidth=0.5))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"h={h}" for h in heads], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"p={p}" for p in primes], fontsize=11)
    ax.set_xlabel("Number of Attention Heads", fontsize=12)
    ax.set_ylabel("Prime Modulus p", fontsize=12)
    ax.set_title("Intrinsic Frequency Count K (90% energy)\n* = not grokked, hatched",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="K (90% energy)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(fig_dir / "joint_K_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_K_heatmap.png")


# ---- Figure 2: K vs log2(p) per head count ----

def plot_K_vs_log_p(cells, primes, heads, fig_dir):
    """K_energy vs log2(p), one line per h. Only grokked runs."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    fit_results = {}

    for j, h in enumerate(heads):
        ps_grok, ks_grok = [], []
        ps_nogrok, ks_nogrok = [], []
        for p in primes:
            if (p, h) not in cells:
                continue
            c = cells[(p, h)]
            if c["grokked"]:
                ps_grok.append(p)
                ks_grok.append(c["k_energy"])
            else:
                ps_nogrok.append(p)
                ks_nogrok.append(c["k_energy"])

        color = cmap(j)

        # Plot grokked points
        if ps_grok:
            ax.scatter(np.log2(ps_grok), ks_grok, s=80, color=color, zorder=5,
                       edgecolors="black", linewidth=0.5, label=f"h={h}")

            # Linear fit on grokked points
            if len(ps_grok) >= 2:
                log2_p = np.log2(ps_grok)
                coeffs = np.polyfit(log2_p, ks_grok, 1)
                ss_res = np.sum((np.array(ks_grok) - np.polyval(coeffs, log2_p)) ** 2)
                ss_tot = np.sum((np.array(ks_grok) - np.mean(ks_grok)) ** 2)
                r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
                fit_results[h] = {"slope": coeffs[0], "intercept": coeffs[1], "r2": r2}

                x_fit = np.linspace(min(log2_p) - 0.2, max(log2_p) + 0.2, 50)
                ax.plot(x_fit, np.polyval(coeffs, x_fit), color=color, linestyle="--",
                        alpha=0.6, linewidth=1.5)
                ax.annotate(f"slope={coeffs[0]:.2f}, R²={r2:.2f}",
                            (log2_p[-1], np.polyval(coeffs, log2_p[-1])),
                            textcoords="offset points", xytext=(5, -10),
                            fontsize=7, color=color)

        # Plot non-grokked as X markers
        if ps_nogrok:
            ax.scatter(np.log2(ps_nogrok), ks_nogrok, s=60, color=color, marker="x",
                       zorder=4, alpha=0.5)

    # McCracken reference: K = log2(p)
    log2_range = np.linspace(np.log2(min(primes)), np.log2(max(primes)), 50)
    ax.plot(log2_range, log2_range, "k:", alpha=0.4, linewidth=1.5, label="K = log₂(p)")

    ax.set_xlabel("log₂(p)", fontsize=12)
    ax.set_ylabel("K (90% energy threshold)", fontsize=12)
    ax.set_title("Frequency Count K vs log₂(p) by Head Count\n(X = not grokked, excluded from fit)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "joint_K_vs_log_p.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_K_vs_log_p.png")
    return fit_results


# ---- Figure 3: Grokking epoch heatmap ----

def plot_grokking_epoch_heatmap(cells, primes, heads, fig_dir):
    """Grokking epoch heatmap with log colorscale. Non-grokked marked X."""
    n_rows, n_cols = len(primes), len(heads)
    data = np.full((n_rows, n_cols), np.nan)

    for i, p in enumerate(primes):
        for j, h in enumerate(heads):
            if (p, h) in cells and cells[(p, h)]["grokked"]:
                data[i, j] = cells[(p, h)]["grok_epoch"]

    fig, ax = plt.subplots(figsize=(8, 7))

    # Use log scale for grokking epochs
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        plt.close(fig)
        return
    norm = mcolors.LogNorm(vmin=max(valid.min(), 1), vmax=valid.max())
    im = ax.imshow(data, cmap="viridis_r", aspect="auto", norm=norm)

    for i in range(n_rows):
        for j in range(n_cols):
            if (primes[i], heads[j]) not in cells:
                ax.text(j, i, "?", ha="center", va="center", fontsize=10, color="gray")
            elif np.isnan(data[i, j]):
                ax.text(j, i, "X", ha="center", va="center", fontsize=14,
                        fontweight="bold", color="#e74c3c")
            else:
                val = int(data[i, j])
                color = "white" if data[i, j] > np.nanmedian(valid) else "black"
                ax.text(j, i, f"{val:,}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"h={h}" for h in heads], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"p={p}" for p in primes], fontsize=11)
    ax.set_xlabel("Number of Attention Heads", fontsize=12)
    ax.set_ylabel("Prime Modulus p", fontsize=12)
    ax.set_title("Grokking Epoch (log scale)\nX = never grokked in 40K epochs",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Grokking Epoch", shrink=0.8)

    fig.tight_layout()
    fig.savefig(fig_dir / "joint_grokking_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_grokking_heatmap.png")


# ---- Figure 4: Phase boundary ----

def plot_phase_boundary(cells, primes, heads, fig_dir):
    """For each h, show the grokking phase boundary between primes."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")

    for j, h in enumerate(heads):
        grokked_ps = []
        not_grokked_ps = []
        for p in primes:
            if (p, h) not in cells:
                continue
            if cells[(p, h)]["grokked"]:
                grokked_ps.append(p)
            else:
                not_grokked_ps.append(p)

        color = cmap(j)

        # Plot grokked / not grokked
        if grokked_ps:
            ax.scatter(grokked_ps, [h] * len(grokked_ps), s=120, color=color,
                       marker="o", edgecolors="black", linewidth=0.8, zorder=5)
        if not_grokked_ps:
            ax.scatter(not_grokked_ps, [h] * len(not_grokked_ps), s=120, color=color,
                       marker="x", linewidth=2, zorder=5)

        # Draw phase boundary band
        if grokked_ps and not_grokked_ps:
            boundary_low = max(not_grokked_ps)
            boundary_high = min(grokked_ps)
            ax.axvspan(boundary_low, boundary_high, ymin=(j) / len(heads),
                       ymax=(j + 1) / len(heads), alpha=0.15, color=color)
            mid = (boundary_low + boundary_high) / 2
            ax.annotate(f"{boundary_low}-{boundary_high}", (mid, h),
                        textcoords="offset points", xytext=(0, 12),
                        fontsize=8, ha="center", color=color, fontweight="bold")
        elif not grokked_ps and not_grokked_ps:
            ax.annotate("never grokked", (max(not_grokked_ps), h),
                        textcoords="offset points", xytext=(10, 0),
                        fontsize=8, color=color)
        elif grokked_ps and not not_grokked_ps:
            ax.annotate("always grokked", (min(grokked_ps), h),
                        textcoords="offset points", xytext=(-40, 10),
                        fontsize=8, color=color)

    ax.set_xlabel("Prime Modulus p", fontsize=12)
    ax.set_ylabel("Number of Heads", fontsize=12)
    ax.set_yticks(heads)
    ax.set_yticklabels([f"h={h}" for h in heads])
    ax.set_title("Grokking Phase Boundary\n(circle = grokked, X = not grokked)",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "joint_phase_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_phase_boundary.png")


# ---- Figure 5: Gini heatmap ----

def plot_gini_heatmap(cells, primes, heads, fig_dir):
    """Gini(p, h) heatmap with annotations."""
    n_rows, n_cols = len(primes), len(heads)
    data = np.full((n_rows, n_cols), np.nan)

    for i, p in enumerate(primes):
        for j, h in enumerate(heads):
            if (p, h) in cells:
                data[i, j] = cells[(p, h)]["gini"]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(n_rows):
        for j in range(n_cols):
            if (primes[i], heads[j]) not in cells:
                ax.text(j, i, "?", ha="center", va="center", fontsize=10, color="gray")
            elif not np.isnan(data[i, j]):
                grokked = cells[(primes[i], heads[j])]["grokked"]
                val = data[i, j]
                color = "black" if val > 0.4 else "white"
                marker = "" if grokked else "*"
                ax.text(j, i, f"{val:.3f}{marker}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"h={h}" for h in heads], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"p={p}" for p in primes], fontsize=11)
    ax.set_xlabel("Number of Attention Heads", fontsize=12)
    ax.set_ylabel("Prime Modulus p", fontsize=12)
    ax.set_title("Fourier Gini Coefficient\n* = not grokked",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Gini", shrink=0.8)

    fig.tight_layout()
    fig.savefig(fig_dir / "joint_gini_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_gini_heatmap.png")


# ---- Figure 6: Test accuracy small-multiples ----

def plot_test_accuracy_grid(runs, cells, primes, heads, fig_dir):
    """8x4 small-multiples of test_acc vs epoch. Green/red border for grokked/not."""
    n_rows, n_cols = len(primes), len(heads)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.2 * n_rows),
                             squeeze=False, sharex=True, sharey=True)

    for i, p in enumerate(primes):
        for j, h in enumerate(heads):
            ax = axes[i][j]
            if (p, h) not in runs:
                ax.text(0.5, 0.5, "missing", transform=ax.transAxes,
                        ha="center", va="center", fontsize=9, color="gray")
                for spine in ax.spines.values():
                    spine.set_edgecolor("gray")
                    spine.set_linewidth(2)
            else:
                metrics = runs[(p, h)]
                history = metrics.get("history", {})
                eval_epochs = history.get("eval_epochs", [])
                test_acc = history.get("test_acc", [])
                grokked = (p, h) in cells and cells[(p, h)]["grokked"]

                if eval_epochs and test_acc:
                    ax.plot(eval_epochs, test_acc, linewidth=1, color="#1f77b4")
                    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.3, linewidth=0.5)

                border_color = "#2ca02c" if grokked else "#e74c3c"
                for spine in ax.spines.values():
                    spine.set_edgecolor(border_color)
                    spine.set_linewidth(2.5)

            ax.set_ylim(-0.05, 1.05)
            if i == 0:
                ax.set_title(f"h={h}", fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"p={p}", fontsize=9)
            if i == n_rows - 1:
                ax.set_xlabel("Epoch", fontsize=8)
            ax.tick_params(labelsize=6)

    fig.suptitle("Test Accuracy Curves (green border = grokked, red = not grokked)",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fig.savefig(fig_dir / "joint_test_accuracy_grid.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_test_accuracy_grid.png")


# ---- Figure 7: Grokking speed interaction ----

def plot_grokking_speed_interaction(cells, primes, heads, fig_dir):
    """Grokking epoch vs h, one line per grokked prime."""
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("viridis")

    # Only primes that grokked in at least one head count
    grokked_primes = []
    for p in primes:
        if any((p, h) in cells and cells[(p, h)]["grokked"] for h in heads):
            grokked_primes.append(p)

    for idx, p in enumerate(grokked_primes):
        hs, epochs = [], []
        for h in heads:
            if (p, h) in cells and cells[(p, h)]["grokked"]:
                hs.append(h)
                epochs.append(cells[(p, h)]["grok_epoch"])

        color = cmap(idx / max(len(grokked_primes) - 1, 1))
        ax.plot(hs, epochs, "o-", color=color, label=f"p={p}", linewidth=1.8,
                markersize=7, markeredgecolor="black", markeredgewidth=0.5)

    ax.set_xlabel("Number of Attention Heads", fontsize=12)
    ax.set_ylabel("Grokking Epoch", fontsize=12)
    ax.set_xticks(heads)
    ax.set_title("Grokking Speed: Epoch vs n_heads per Prime",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "joint_grokking_interaction.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: joint_grokking_interaction.png")


# ---- Summary table ----

def print_summary_table(cells, primes, heads):
    """Console 2D grid: rows=primes, cols=heads. Cell = grok_epoch (K_energy)."""
    print("\n" + "=" * 100)
    print("JOINT (n_heads x p) SWEEP -- METRICS SUMMARY")
    print("=" * 100)

    header = f"{'p':>5} | {'N_train':>7}"
    for h in heads:
        header += f" | {'h=' + str(h):>16}"
    print(header)
    print("-" * 100)

    for p in primes:
        n_train = int(p * p * 0.3)
        row = f"{p:>5} | {n_train:>7}"
        for h in heads:
            if (p, h) not in cells:
                row += f" | {'MISSING':>16}"
            else:
                c = cells[(p, h)]
                if c["grokked"]:
                    cell = f"{c['grok_epoch']} (K={c['k_energy']})"
                else:
                    cell = f"-- (K={c['k_energy']})"
                row += f" | {cell:>16}"
        print(row)

    print("=" * 100)
    print("Cell format: grok_epoch (K=K_energy)  |  -- = never grokked")
    print()


# ---- JSON output ----

def save_metrics_json(cells, primes, heads, fit_results, fig_dir):
    """Save all per-cell metrics + fit slopes to JSON."""
    output = {"cells": {}, "k_vs_logp_fits": {}}

    for (p, h), c in cells.items():
        key = f"p{p}_h{h}"
        output["cells"][key] = {
            "p": p,
            "n_heads": h,
            "grok_epoch": c["grok_epoch"],
            "grokked": c["grokked"],
            "test_acc": c["test_acc"],
            "train_acc": c["train_acc"],
            "gini": c["gini"],
            "k_energy": c["k_energy"],
            "k_unique": c["k_unique"],
            "max_k": c["max_k"],
        }

    for h, fit in fit_results.items():
        output["k_vs_logp_fits"][f"h={h}"] = {
            "slope": fit["slope"],
            "intercept": fit["intercept"],
            "r2": fit["r2"],
        }

    out_path = fig_dir / "joint_sweep_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare joint (n_heads x p) sweep results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/heads_x_primes_comparison/)")
    parser.add_argument("--primes", type=int, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES
    heads = args.heads if args.heads else SWEEP_HEADS
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "heads_x_primes_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, primes, heads, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    total_cells = len(primes) * len(heads)
    logger.info(f"Loaded {len(runs)}/{total_cells} cells")

    cells = extract_cell_metrics(runs, primes, heads, logger)

    logger.info("1/7: K heatmap")
    plot_K_heatmap(cells, primes, heads, fig_dir)

    logger.info("2/7: K vs log2(p)")
    fit_results = plot_K_vs_log_p(cells, primes, heads, fig_dir)

    logger.info("3/7: Grokking epoch heatmap")
    plot_grokking_epoch_heatmap(cells, primes, heads, fig_dir)

    logger.info("4/7: Phase boundary")
    plot_phase_boundary(cells, primes, heads, fig_dir)

    logger.info("5/7: Gini heatmap")
    plot_gini_heatmap(cells, primes, heads, fig_dir)

    logger.info("6/7: Test accuracy grid")
    plot_test_accuracy_grid(runs, cells, primes, heads, fig_dir)

    logger.info("7/7: Grokking speed interaction")
    plot_grokking_speed_interaction(cells, primes, heads, fig_dir)

    print_summary_table(cells, primes, heads)

    save_metrics_json(cells, primes, heads, fit_results, fig_dir)

    logger.info(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
