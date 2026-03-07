#!/usr/bin/env python
"""Cross-run comparison for the multi-seed validation sweep.

Loads metrics.json from all 48 cells (4 primes x 4 heads x 3 seeds) and produces:
1. Grokking epoch bar chart with error bars (h on x-axis, grouped by p)
2. Statistical test table (h=1 vs h=4 grokking epoch per prime)
3. Seed variability heatmap (CV of grokking epoch across seeds)
4. p=43 partial grokking analysis (test_acc distribution across seeds)
5. Phase transition consistency check across seeds
6. Gini comparison with error bars

Also saves multi_seed_metrics.json with all per-cell metrics.
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

from src.utils import setup_logging, run_id

SWEEP_PRIMES = [43, 59, 67, 113]
SWEEP_HEADS = [1, 2, 4, 8]
SWEEP_SEEDS = [42, 137, 256]
GROK_THRESHOLD = 0.95


def _make_run_id(p, h, seed):
    return run_id({"p": p, "d_model": 128, "n_heads": h, "d_mlp": 512,
                   "n_layers": 1, "weight_decay": 1.0, "seed": seed,
                   "lr": 1e-3, "operation": "addition", "train_fraction": 0.3})


def load_all_runs(results_root, primes, heads, seeds, logger):
    """Load metrics for all (p, h, seed) cells. Returns dict keyed by (p, h, seed)."""
    runs = {}
    for p in primes:
        for h in heads:
            for seed in seeds:
                rid = _make_run_id(p, h, seed)
                metrics_path = results_root / rid / "metrics.json"
                if not metrics_path.exists():
                    logger.warning(f"Missing: {rid}")
                    continue
                with open(metrics_path) as f:
                    metrics = json.load(f)
                runs[(p, h, seed)] = metrics
    logger.info(f"Loaded {len(runs)} / {len(primes)*len(heads)*len(seeds)} cells")
    return runs


def find_grokking_epoch(metrics, threshold=GROK_THRESHOLD):
    """Find first epoch where test acc >= threshold."""
    history = metrics.get("history", {})
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def extract_cell_data(runs, primes, heads, seeds):
    """Extract structured data for all cells.

    Returns dict keyed by (p, h) with lists across seeds:
      grok_epochs: list of int|None
      test_accs: list of float
      ginis: list of float
    """
    cells = {}
    for p in primes:
        for h in heads:
            grok_epochs = []
            test_accs = []
            ginis = []
            for seed in seeds:
                if (p, h, seed) not in runs:
                    grok_epochs.append(None)
                    test_accs.append(None)
                    ginis.append(None)
                    continue
                m = runs[(p, h, seed)]
                grok_epochs.append(find_grokking_epoch(m))
                test_accs.append(m.get("final_test_acc", 0))
                ginis.append(m.get("final_gini", 0))
            cells[(p, h)] = {
                "grok_epochs": grok_epochs,
                "test_accs": test_accs,
                "ginis": ginis,
            }
    return cells


# ---- Figure 1: Grokking epoch bar chart with error bars ----

def plot_grokking_bar_chart(cells, primes, heads, seeds, fig_dir):
    """Bar chart: h on x-axis, grouped by p, error bars = std across seeds."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")

    # Only primes that grok (p >= 59 expected)
    grokked_primes = []
    for p in primes:
        has_grok = any(
            any(e is not None for e in cells[(p, h)]["grok_epochs"])
            for h in heads
        )
        if has_grok:
            grokked_primes.append(p)

    n_groups = len(grokked_primes)
    n_bars = len(heads)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    for j, h in enumerate(heads):
        means = []
        stds = []
        for p in grokked_primes:
            epochs = [e for e in cells[(p, h)]["grok_epochs"] if e is not None]
            if epochs:
                means.append(np.mean(epochs))
                stds.append(np.std(epochs))
            else:
                means.append(0)
                stds.append(0)

        offset = (j - (n_bars - 1) / 2) * bar_width
        bars = ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
                      capsize=3, color=cmap(j), edgecolor="black",
                      linewidth=0.5, label=f"h={h}", alpha=0.85)

        # Annotate means
        for k, (m, s) in enumerate(zip(means, stds)):
            if m > 0:
                ax.text(x[k] + offset, m + s + 200, f"{int(m)}",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in grokked_primes], fontsize=11)
    ax.set_ylabel("Grokking Epoch", fontsize=12)
    ax.set_xlabel("Prime Modulus", fontsize=12)
    ax.set_title(f"Grokking Epoch by Head Count (mean +/- std, n={len(seeds)} seeds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_dir / "multi_seed_grokking_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: multi_seed_grokking_bar.png")


# ---- Figure 2: Statistical test table ----

def compute_statistical_tests(cells, primes, heads, seeds, fig_dir):
    """Paired comparison: h=1 vs each other h, per prime.

    Uses Wilcoxon signed-rank test where possible (n >= 5), otherwise
    reports descriptive stats since n=3 is too small for reliable p-values.
    """
    from scipy import stats

    results = []

    for p in primes:
        h1_epochs = [e for e in cells[(p, 1)]["grok_epochs"] if e is not None]
        if not h1_epochs:
            continue

        for h in heads:
            if h == 1:
                continue
            h_epochs = [e for e in cells[(p, h)]["grok_epochs"] if e is not None]
            if not h_epochs:
                continue

            h1_mean = np.mean(h1_epochs)
            h_mean = np.mean(h_epochs)
            speedup = h_mean / h1_mean if h1_mean > 0 else float("inf")

            # With only 3 seeds, use a paired t-test on matched seeds
            paired_h1 = []
            paired_h = []
            for seed_idx, seed in enumerate(seeds):
                e1 = cells[(p, 1)]["grok_epochs"][seed_idx]
                e_other = cells[(p, h)]["grok_epochs"][seed_idx]
                if e1 is not None and e_other is not None:
                    paired_h1.append(e1)
                    paired_h.append(e_other)

            if len(paired_h1) >= 2:
                t_stat, p_val = stats.ttest_rel(paired_h1, paired_h, alternative="less")
            else:
                t_stat, p_val = float("nan"), float("nan")

            results.append({
                "p": p,
                "h_compare": h,
                "h1_mean": h1_mean,
                "h1_std": np.std(h1_epochs),
                "h_mean": h_mean,
                "h_std": np.std(h_epochs),
                "speedup": speedup,
                "n_paired": len(paired_h1),
                "t_stat": t_stat,
                "p_value": p_val,
                "h1_faster": h1_mean < h_mean,
            })

    # Print table
    print("\n" + "=" * 100)
    print("STATISTICAL TESTS: h=1 vs h=k grokking epoch (paired t-test, one-sided)")
    print("=" * 100)
    print(f"{'Prime':>6} {'h=1 vs':>8} | {'h=1 mean':>10} {'h=k mean':>10} "
          f"{'Speedup':>8} {'n_paired':>9} {'t-stat':>8} {'p-value':>10} {'h=1 faster?':>12}")
    print("-" * 100)

    for r in results:
        sig = ""
        if not np.isnan(r["p_value"]):
            if r["p_value"] < 0.01:
                sig = "**"
            elif r["p_value"] < 0.05:
                sig = "*"
            elif r["p_value"] < 0.10:
                sig = "."

        print(f"  p={r['p']:>3} h={r['h_compare']:>2}   | "
              f"{r['h1_mean']:>10.0f} {r['h_mean']:>10.0f} "
              f"{r['speedup']:>7.2f}x {r['n_paired']:>9} "
              f"{r['t_stat']:>8.2f} {r['p_value']:>9.4f}{sig:>2} "
              f"{'YES' if r['h1_faster'] else 'no':>12}")

    print("-" * 100)
    # Count how many primes show h=1 faster
    grokked_primes = set(r["p"] for r in results)
    h1_wins = {}
    for h in heads:
        if h == 1:
            continue
        h1_wins[h] = sum(1 for r in results if r["h_compare"] == h and r["h1_faster"])
    print(f"h=1 faster count: " + ", ".join(
        f"vs h={h}: {w}/{len(grokked_primes)}" for h, w in h1_wins.items()))
    print("Significance: ** p<0.01, * p<0.05, . p<0.10")
    print("=" * 100)

    # Create figure version
    fig, ax = plt.subplots(figsize=(10, max(3, len(results) * 0.4 + 1.5)))
    ax.axis("off")

    col_labels = ["Prime", "h=1 vs", "h=1 mean", "h=k mean", "Speedup",
                  "n_paired", "t-stat", "p-value", "h=1 faster?"]
    table_data = []
    cell_colors = []
    for r in results:
        sig = ""
        if not np.isnan(r["p_value"]):
            if r["p_value"] < 0.05:
                sig = "*"
        row = [
            f"p={r['p']}",
            f"h={r['h_compare']}",
            f"{r['h1_mean']:.0f} +/- {r['h1_std']:.0f}",
            f"{r['h_mean']:.0f} +/- {r['h_std']:.0f}",
            f"{r['speedup']:.2f}x",
            str(r["n_paired"]),
            f"{r['t_stat']:.2f}" if not np.isnan(r["t_stat"]) else "N/A",
            f"{r['p_value']:.4f}{sig}" if not np.isnan(r["p_value"]) else "N/A",
            "YES" if r["h1_faster"] else "no",
        ]
        table_data.append(row)

        if r["h1_faster"] and not np.isnan(r["p_value"]) and r["p_value"] < 0.05:
            cell_colors.append(["#d4edda"] * len(col_labels))
        elif r["h1_faster"]:
            cell_colors.append(["#fff3cd"] * len(col_labels))
        else:
            cell_colors.append(["#f8d7da"] * len(col_labels))

    if table_data:
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellColours=cell_colors, loc="center",
                         cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 1.4)
        # Header styling
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#343a40")
            table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("Statistical Tests: h=1 vs h=k Grokking Epoch\n"
                 "(paired t-test, one-sided; green=sig, yellow=trend, red=h=1 not faster)",
                 fontsize=12, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(fig_dir / "multi_seed_stat_tests.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: multi_seed_stat_tests.png")

    return results


# ---- Figure 3: Seed variability heatmap ----

def plot_seed_variability_heatmap(cells, primes, heads, fig_dir):
    """Heatmap of CV(grok_epoch) across seeds for each (p, h) cell."""
    n_rows, n_cols = len(primes), len(heads)
    data = np.full((n_rows, n_cols), np.nan)
    annotations = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for i, p in enumerate(primes):
        for j, h in enumerate(heads):
            epochs = [e for e in cells[(p, h)]["grok_epochs"] if e is not None]
            if len(epochs) >= 2:
                mean = np.mean(epochs)
                std = np.std(epochs)
                cv = std / mean if mean > 0 else 0
                data[i, j] = cv
                annotations[i][j] = f"{cv:.2f}\n({len(epochs)})"
            elif len(epochs) == 1:
                annotations[i][j] = f"n=1"
            else:
                annotations[i][j] = "X"

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0,
                   vmax=max(0.5, np.nanmax(data) * 1.1) if not np.all(np.isnan(data)) else 0.5)

    for i in range(n_rows):
        for j in range(n_cols):
            color = "white" if not np.isnan(data[i, j]) and data[i, j] > 0.25 else "black"
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"h={h}" for h in heads], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"p={p}" for p in primes], fontsize=11)
    ax.set_xlabel("Number of Attention Heads", fontsize=12)
    ax.set_ylabel("Prime Modulus p", fontsize=12)
    ax.set_title("Seed Variability: CV(grok_epoch)\n"
                 "(lower = more reproducible, X = never grokked)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="CV (std/mean)", shrink=0.8)

    fig.tight_layout()
    fig.savefig(fig_dir / "multi_seed_cv_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: multi_seed_cv_heatmap.png")


# ---- Figure 4: p=43 partial grokking analysis ----

def plot_p43_partial_grokking(cells, runs, heads, seeds, fig_dir):
    """Test accuracy distribution at p=43 across seeds, comparing h values."""
    p = 43
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bar chart of test_acc per h, with individual seed points
    ax = axes[0]
    cmap = plt.get_cmap("tab10")

    for j, h in enumerate(heads):
        accs = [a for a in cells[(p, h)]["test_accs"] if a is not None]
        if not accs:
            continue
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        ax.bar(j, mean_acc, 0.6, color=cmap(j), alpha=0.7, edgecolor="black",
               linewidth=0.5, label=f"h={h}")
        ax.errorbar(j, mean_acc, yerr=std_acc, fmt="none", color="black", capsize=5)

        # Individual seed points
        for k, acc in enumerate(accs):
            ax.scatter(j + (k - 1) * 0.12, acc, s=40, color="black",
                       zorder=5, marker=["o", "s", "^"][k % 3])

    ax.set_xticks(range(len(heads)))
    ax.set_xticklabels([f"h={h}" for h in heads], fontsize=11)
    ax.set_ylabel("Final Test Accuracy", fontsize=12)
    ax.set_title(f"p=43 (N_train={int(43*43*0.3)}): Test Accuracy by Head Count\n"
                 f"(bars=mean, error=std, points=individual seeds)",
                 fontsize=11, fontweight="bold")
    ax.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="Grok threshold")
    ax.axhline(1/43, color="red", linestyle=":", alpha=0.5, label="Chance level")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: test_acc curves for p=43 across seeds (one panel per h)
    ax2 = axes[1]
    for j, h in enumerate(heads):
        for seed_idx, seed in enumerate(seeds):
            if (p, h, seed) not in runs:
                continue
            m = runs[(p, h, seed)]
            history = m.get("history", {})
            eval_epochs = history.get("eval_epochs", [])
            test_acc = history.get("test_acc", [])
            if eval_epochs and test_acc:
                linestyle = ["-", "--", ":"][seed_idx % 3]
                alpha = 0.8 if h == 1 else 0.4
                lw = 2.0 if h == 1 else 1.0
                label = f"h={h} s={seed}" if seed_idx == 0 or h == 1 else None
                ax2.plot(eval_epochs, test_acc, color=cmap(j),
                         linestyle=linestyle, alpha=alpha, linewidth=lw,
                         label=label)

    ax2.axhline(0.95, color="green", linestyle="--", alpha=0.3)
    ax2.axhline(1/43, color="red", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Test Accuracy", fontsize=12)
    ax2.set_title("p=43: Test Accuracy Curves\n(solid/dashed/dotted = seed 42/137/256)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "multi_seed_p43_partial_grokking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: multi_seed_p43_partial_grokking.png")


# ---- Figure 5: Phase transition consistency ----

def plot_phase_transition_consistency(cells, primes, heads, seeds, fig_dir):
    """For each (p, h), show fraction of seeds that grokked."""
    n_rows, n_cols = len(primes), len(heads)
    data = np.full((n_rows, n_cols), np.nan)
    annotations = [["" for _ in range(n_cols)] for _ in range(n_rows)]

    for i, p in enumerate(primes):
        for j, h in enumerate(heads):
            epochs = cells[(p, h)]["grok_epochs"]
            valid = [e for e in epochs if e is not None]
            total = sum(1 for e in epochs if e is not None or e is None)  # count non-missing
            n_present = sum(1 for e in cells[(p, h)]["test_accs"] if e is not None)
            if n_present > 0:
                frac = len(valid) / n_present
                data[i, j] = frac
                annotations[i][j] = f"{len(valid)}/{n_present}"

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(n_rows):
        for j in range(n_cols):
            ax.text(j, i, annotations[i][j], ha="center", va="center",
                    fontsize=12, fontweight="bold",
                    color="white" if not np.isnan(data[i, j]) and data[i, j] < 0.4 else "black")

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"h={h}" for h in heads], fontsize=11)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([f"p={p}" for p in primes], fontsize=11)
    ax.set_xlabel("Number of Attention Heads", fontsize=12)
    ax.set_ylabel("Prime Modulus p", fontsize=12)
    ax.set_title("Phase Transition Consistency\n"
                 "(fraction of seeds that grokked, n_grokked/n_total)",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Fraction Grokked", shrink=0.8)

    fig.tight_layout()
    fig.savefig(fig_dir / "multi_seed_phase_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: multi_seed_phase_consistency.png")


# ---- Figure 6: Gini comparison with error bars ----

def plot_gini_comparison(cells, primes, heads, fig_dir):
    """Gini by h, grouped by p, with error bars across seeds."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = plt.get_cmap("tab10")

    n_groups = len(primes)
    n_bars = len(heads)
    bar_width = 0.8 / n_bars
    x = np.arange(n_groups)

    for j, h in enumerate(heads):
        means = []
        stds = []
        for p in primes:
            ginis = [g for g in cells[(p, h)]["ginis"] if g is not None]
            if ginis:
                means.append(np.mean(ginis))
                stds.append(np.std(ginis))
            else:
                means.append(0)
                stds.append(0)

        offset = (j - (n_bars - 1) / 2) * bar_width
        ax.bar(x + offset, means, bar_width * 0.9, yerr=stds,
               capsize=3, color=cmap(j), edgecolor="black",
               linewidth=0.5, label=f"h={h}", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"p={p}" for p in primes], fontsize=11)
    ax.set_ylabel("Gini Coefficient", fontsize=12)
    ax.set_xlabel("Prime Modulus", fontsize=12)
    ax.set_title("Fourier Gini by Head Count (mean +/- std across seeds)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(fig_dir / "multi_seed_gini_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: multi_seed_gini_comparison.png")


# ---- Summary table ----

def print_summary_table(cells, primes, heads, seeds):
    """Console summary with mean +/- std across seeds."""
    print("\n" + "=" * 110)
    print("MULTI-SEED VALIDATION -- SUMMARY (mean +/- std across seeds)")
    print("=" * 110)

    header = f"{'p':>5} | {'N_train':>7}"
    for h in heads:
        header += f" | {'h=' + str(h):>22}"
    print(header)
    print("-" * 110)

    for p in primes:
        n_train = int(p * p * 0.3)
        row = f"{p:>5} | {n_train:>7}"
        for h in heads:
            epochs = [e for e in cells[(p, h)]["grok_epochs"] if e is not None]
            n_seeds = sum(1 for a in cells[(p, h)]["test_accs"] if a is not None)
            if epochs:
                mean_e = np.mean(epochs)
                std_e = np.std(epochs)
                cell = f"{mean_e:.0f}+/-{std_e:.0f} ({len(epochs)}/{n_seeds})"
            else:
                accs = [a for a in cells[(p, h)]["test_accs"] if a is not None]
                mean_a = np.mean(accs) if accs else 0
                cell = f"-- (acc={mean_a:.2f}, 0/{n_seeds})"
            row += f" | {cell:>22}"
        print(row)

    print("=" * 110)
    print("Format: mean_epoch +/- std (n_grokked/n_seeds) | -- = never grokked")


# ---- JSON output ----

def save_metrics_json(cells, primes, heads, seeds, stat_results, fig_dir):
    """Save all per-cell metrics to JSON."""
    output = {
        "sweep_config": {
            "primes": primes,
            "heads": heads,
            "seeds": seeds,
        },
        "cells": {},
        "statistical_tests": [],
    }

    for (p, h), c in cells.items():
        key = f"p{p}_h{h}"
        epochs = [e for e in c["grok_epochs"] if e is not None]
        accs = [a for a in c["test_accs"] if a is not None]
        ginis = [g for g in c["ginis"] if g is not None]
        output["cells"][key] = {
            "p": p,
            "n_heads": h,
            "grok_epochs": c["grok_epochs"],
            "grok_epoch_mean": float(np.mean(epochs)) if epochs else None,
            "grok_epoch_std": float(np.std(epochs)) if len(epochs) >= 2 else None,
            "n_grokked": len(epochs),
            "n_seeds": len(accs),
            "test_acc_mean": float(np.mean(accs)) if accs else None,
            "test_acc_std": float(np.std(accs)) if len(accs) >= 2 else None,
            "test_accs": c["test_accs"],
            "gini_mean": float(np.mean(ginis)) if ginis else None,
            "gini_std": float(np.std(ginis)) if len(ginis) >= 2 else None,
        }

    for r in stat_results:
        output["statistical_tests"].append({
            "p": r["p"],
            "h_compare": r["h_compare"],
            "h1_mean": r["h1_mean"],
            "h_mean": r["h_mean"],
            "speedup": r["speedup"],
            "t_stat": r["t_stat"] if not np.isnan(r["t_stat"]) else None,
            "p_value": r["p_value"] if not np.isnan(r["p_value"]) else None,
            "h1_faster": bool(r["h1_faster"]),
        })

    out_path = fig_dir / "multi_seed_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare multi-seed validation sweep results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/multi_seed_comparison/)")
    parser.add_argument("--primes", type=int, nargs="+", default=None)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES
    heads = args.heads if args.heads else SWEEP_HEADS
    seeds = args.seeds if args.seeds else SWEEP_SEEDS
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "multi_seed_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, primes, heads, seeds, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    cells = extract_cell_data(runs, primes, heads, seeds)

    logger.info("1/6: Grokking epoch bar chart with error bars")
    plot_grokking_bar_chart(cells, primes, heads, seeds, fig_dir)

    logger.info("2/6: Statistical tests (h=1 vs h=k)")
    stat_results = compute_statistical_tests(cells, primes, heads, seeds, fig_dir)

    logger.info("3/6: Seed variability heatmap")
    plot_seed_variability_heatmap(cells, primes, heads, fig_dir)

    logger.info("4/6: p=43 partial grokking analysis")
    if 43 in primes:
        plot_p43_partial_grokking(cells, runs, heads, seeds, fig_dir)
    else:
        logger.info("  Skipping (p=43 not in sweep)")

    logger.info("5/6: Phase transition consistency")
    plot_phase_transition_consistency(cells, primes, heads, seeds, fig_dir)

    logger.info("6/6: Gini comparison")
    plot_gini_comparison(cells, primes, heads, fig_dir)

    print_summary_table(cells, primes, heads, seeds)

    save_metrics_json(cells, primes, heads, seeds, stat_results, fig_dir)

    # Print success criteria evaluation
    print("\n" + "=" * 80)
    print("SUCCESS CRITERIA EVALUATION")
    print("=" * 80)

    # Criterion 1: h=1 mean grok_epoch < h=4 mean grok_epoch for >= 3/4 primes
    grokked_primes = [p for p in primes if any(
        e is not None for e in cells[(p, 1)]["grok_epochs"])]
    h1_faster_count = 0
    sig_count = 0
    for p in grokked_primes:
        h1_epochs = [e for e in cells[(p, 1)]["grok_epochs"] if e is not None]
        h4_epochs = [e for e in cells[(p, 4)]["grok_epochs"] if e is not None]
        if h1_epochs and h4_epochs:
            if np.mean(h1_epochs) < np.mean(h4_epochs):
                h1_faster_count += 1
            # Check significance from stat_results
            for r in stat_results:
                if r["p"] == p and r["h_compare"] == 4:
                    if r["h1_faster"] and not np.isnan(r["p_value"]) and r["p_value"] < 0.05:
                        sig_count += 1

    print(f"\n1. h=1 faster than h=4: {h1_faster_count}/{len(grokked_primes)} primes "
          f"({'PASS' if h1_faster_count >= 3 else 'FAIL'}, need >= 3/4)")
    print(f"   Statistically significant (p<0.05): {sig_count}/{len(grokked_primes)}")

    # Criterion 2: p=43/h=1 partial grokking reproduces in >= 2/3 seeds
    if 43 in primes:
        p43_h1_accs = [a for a in cells[(43, 1)]["test_accs"] if a is not None]
        partial_count = sum(1 for a in p43_h1_accs if a > 0.5)
        total_p43 = len(p43_h1_accs)
        print(f"\n2. p=43/h=1 partial grokking (test_acc > 0.5): {partial_count}/{total_p43} seeds "
              f"({'PASS' if partial_count >= 2 else 'FAIL'}, need >= 2/3)")
        for i, a in enumerate(p43_h1_accs):
            print(f"   Seed {seeds[i] if i < len(seeds) else '?'}: test_acc = {a:.4f}")

    print("=" * 80)

    logger.info(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
