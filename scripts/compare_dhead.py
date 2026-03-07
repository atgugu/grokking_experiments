#!/usr/bin/env python
"""Cross-run comparison for the d_head control experiment.

Loads metrics from all cells and produces:
1. Grokking epoch bar chart: (h,d_model) configs grouped by p, with error bars
2. Param-efficiency scatter: grok_epoch vs total params
3. Statistical tests: h=1/d=32 vs h=4/d=128 (matched d_head), h=1/d=32 vs h=1/d=64
4. Phase boundary analysis at p=43: which configs grok?
5. d_head vs head-count decomposition summary

Also saves dhead_control_metrics.json.
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

# (n_heads, d_model, d_mlp) configurations
SWEEP_CONFIGS = [
    (1, 32, 128),
    (2, 64, 256),
    (4, 128, 512),
    (1, 64, 256),
]
CONFIG_LABELS = {
    (1, 32): "h=1,d=32\n(d_head=32)",
    (2, 64): "h=2,d=64\n(d_head=32)",
    (4, 128): "h=4,d=128\n(d_head=32)",
    (1, 64): "h=1,d=64\n(d_head=64)",
}
SWEEP_PRIMES = [43, 113]
SWEEP_SEEDS = [42, 137, 256]
GROK_THRESHOLD = 0.95


def _make_run_id(h, d_model, d_mlp, p, seed):
    return run_id({"p": p, "d_model": d_model, "n_heads": h, "d_mlp": d_mlp,
                   "n_layers": 1, "weight_decay": 1.0, "seed": seed,
                   "lr": 1e-3, "operation": "addition", "train_fraction": 0.3})


def load_all_runs(results_root, configs, primes, seeds, logger):
    """Load metrics for all cells. Returns dict keyed by (h, d_model, p, seed)."""
    runs = {}
    for h, d_model, d_mlp in configs:
        for p in primes:
            for seed in seeds:
                rid = _make_run_id(h, d_model, d_mlp, p, seed)
                metrics_path = results_root / rid / "metrics.json"
                if not metrics_path.exists():
                    logger.warning(f"Missing: {rid}")
                    continue
                with open(metrics_path) as f:
                    metrics = json.load(f)
                runs[(h, d_model, p, seed)] = metrics
    expected = len(configs) * len(primes) * len(seeds)
    logger.info(f"Loaded {len(runs)} / {expected} cells")
    return runs


def find_grokking_epoch(metrics, threshold=GROK_THRESHOLD):
    history = metrics.get("history", {})
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def extract_cell_data(runs, configs, primes, seeds):
    """Extract structured data. Returns dict keyed by (h, d_model, p)."""
    cells = {}
    for h, d_model, _ in configs:
        for p in primes:
            grok_epochs = []
            test_accs = []
            ginis = []
            for seed in seeds:
                if (h, d_model, p, seed) not in runs:
                    grok_epochs.append(None)
                    test_accs.append(None)
                    ginis.append(None)
                    continue
                m = runs[(h, d_model, p, seed)]
                grok_epochs.append(find_grokking_epoch(m))
                test_accs.append(m.get("final_test_acc", 0))
                ginis.append(m.get("final_gini", 0))
            cells[(h, d_model, p)] = {
                "grok_epochs": grok_epochs,
                "test_accs": test_accs,
                "ginis": ginis,
            }
    return cells


def estimate_params(h, d_model, d_mlp, p=113):
    """Rough parameter count estimate."""
    # W_E: (p+1)*d_model, W_pos: 3*d_model, W_U: p*d_model
    # Attention: in_proj 3*d_model*d_model + out_proj d_model*d_model = 4*d_model^2
    # MLP: d_model*d_mlp + d_mlp + d_mlp*d_model + d_model = 2*d_model*d_mlp + d_mlp + d_model
    embed = (p + 1 + 3) * d_model + p * d_model
    attn = 4 * d_model * d_model
    mlp = 2 * d_model * d_mlp + d_mlp + d_model
    return embed + attn + mlp


# ---- Figure 1: Grokking epoch bar chart ----

def plot_grokking_bar_chart(cells, configs, primes, seeds, fig_dir):
    fig, axes = plt.subplots(1, len(primes), figsize=(7 * len(primes), 6), sharey=True)
    if len(primes) == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    config_keys = [(h, d) for h, d, _ in configs]

    for ax, p in zip(axes, primes):
        n_bars = len(config_keys)
        bar_width = 0.8 / n_bars
        x = np.arange(1)  # single group per panel

        for j, (h, d_model) in enumerate(config_keys):
            key = (h, d_model, p)
            if key not in cells:
                continue
            epochs = [e for e in cells[key]["grok_epochs"] if e is not None]
            mean_e = np.mean(epochs) if epochs else 0
            std_e = np.std(epochs) if len(epochs) >= 2 else 0

            offset = (j - (n_bars - 1) / 2) * bar_width
            label = CONFIG_LABELS.get((h, d_model), f"h={h},d={d_model}")
            bar = ax.bar(x + offset, [mean_e], bar_width * 0.9, yerr=[std_e],
                         capsize=4, color=cmap(j), edgecolor="black",
                         linewidth=0.5, label=label.replace('\n', ' '), alpha=0.85)

            n_grok = len(epochs)
            n_total = sum(1 for a in cells[key]["test_accs"] if a is not None)
            if mean_e > 0:
                ax.text(x[0] + offset, mean_e + std_e + 200,
                        f"{int(mean_e)}\n({n_grok}/{n_total})",
                        ha="center", va="bottom", fontsize=7, fontweight="bold")
            else:
                ax.text(x[0] + offset, 500, f"0/{n_total}\ngrok",
                        ha="center", va="bottom", fontsize=7, color="red")

        n_train = int(p * p * 0.3)
        ax.set_title(f"p={p} (N_train={n_train})", fontsize=12, fontweight="bold")
        ax.set_xticks([])
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=8, loc="upper right")

    axes[0].set_ylabel("Grokking Epoch", fontsize=12)
    fig.suptitle(f"d_head Control: Grokking Epoch (mean +/- std, n={len(seeds)} seeds)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "dhead_grokking_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dhead_grokking_bar.png")


# ---- Figure 2: Parameter efficiency scatter ----

def plot_param_efficiency(cells, configs, primes, seeds, fig_dir):
    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.get_cmap("tab10")
    markers = {"43": "^", "113": "o"}

    for j, (h, d_model, d_mlp) in enumerate(configs):
        for p in primes:
            key = (h, d_model, p)
            if key not in cells:
                continue
            epochs = [e for e in cells[key]["grok_epochs"] if e is not None]
            if not epochs:
                continue
            mean_e = np.mean(epochs)
            std_e = np.std(epochs) if len(epochs) >= 2 else 0
            params = estimate_params(h, d_model, d_mlp, p)

            label = CONFIG_LABELS.get((h, d_model), f"h={h},d={d_model}").replace('\n', ' ')
            ax.errorbar(params, mean_e, yerr=std_e, fmt=markers[str(p)],
                        color=cmap(j), markersize=10, capsize=5,
                        label=f"{label}, p={p}")

    ax.set_xlabel("Total Parameters", fontsize=12)
    ax.set_ylabel("Grokking Epoch", fontsize=12)
    ax.set_xscale("log")
    ax.set_title("Parameter Efficiency: Grokking Speed vs Model Size\n"
                 "(circles=p=113, triangles=p=43)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "dhead_param_efficiency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dhead_param_efficiency.png")


# ---- Figure 3: Statistical tests ----

def compute_statistical_tests(cells, configs, primes, seeds, fig_dir):
    from scipy import stats

    config_keys = [(h, d) for h, d, _ in configs]
    results = []

    # Comparisons of interest:
    # 1. h=1/d=32 vs h=4/d=128 (same d_head=32, different h): is it head count?
    # 2. h=1/d=32 vs h=2/d=64 (same d_head=32): gradation
    # 3. h=1/d=32 vs h=1/d=64 (same h=1, different d_head): is it d_head capacity?
    # 4. h=1/d=64 vs h=2/d=64 (same d_model, same d_mlp): pure head count effect
    comparisons = [
        ((1, 32), (4, 128), "h=1/d=32 vs h=4/d=128 (head count, d_head=32)"),
        ((1, 32), (2, 64),  "h=1/d=32 vs h=2/d=64 (head count, d_head=32)"),
        ((1, 32), (1, 64),  "h=1/d=32 vs h=1/d=64 (d_head capacity, h=1)"),
        ((1, 64), (2, 64),  "h=1/d=64 vs h=2/d=64 (head count, same d_model)"),
    ]

    for p in primes:
        for (h_a, d_a), (h_b, d_b), desc in comparisons:
            key_a = (h_a, d_a, p)
            key_b = (h_b, d_b, p)
            if key_a not in cells or key_b not in cells:
                continue

            # Paired comparison across seeds
            paired_a, paired_b = [], []
            for seed_idx in range(len(seeds)):
                ea = cells[key_a]["grok_epochs"][seed_idx]
                eb = cells[key_b]["grok_epochs"][seed_idx]
                if ea is not None and eb is not None:
                    paired_a.append(ea)
                    paired_b.append(eb)

            mean_a = np.mean(paired_a) if paired_a else None
            mean_b = np.mean(paired_b) if paired_b else None

            if len(paired_a) >= 2:
                t_stat, p_val = stats.ttest_rel(paired_a, paired_b, alternative="less")
            else:
                t_stat, p_val = float("nan"), float("nan")

            speedup = mean_b / mean_a if mean_a and mean_b and mean_a > 0 else None

            results.append({
                "p": p,
                "config_a": f"h={h_a}/d={d_a}",
                "config_b": f"h={h_b}/d={d_b}",
                "desc": desc,
                "mean_a": mean_a,
                "mean_b": mean_b,
                "speedup": speedup,
                "n_paired": len(paired_a),
                "t_stat": t_stat,
                "p_value": p_val,
                "a_faster": mean_a < mean_b if mean_a and mean_b else None,
            })

    # Print table
    print("\n" + "=" * 120)
    print("STATISTICAL TESTS: d_head CONTROL (paired t-test, one-sided: A < B)")
    print("=" * 120)
    print(f"{'Prime':>6} {'Comparison':>45} | {'A mean':>8} {'B mean':>8} "
          f"{'Speedup':>8} {'n':>3} {'t-stat':>8} {'p-value':>10} {'A faster?':>10}")
    print("-" * 120)

    for r in results:
        sig = ""
        if not np.isnan(r.get("p_value", float("nan"))):
            if r["p_value"] < 0.01:
                sig = "**"
            elif r["p_value"] < 0.05:
                sig = "*"
            elif r["p_value"] < 0.10:
                sig = "."

        mean_a_str = f"{r['mean_a']:.0f}" if r["mean_a"] is not None else "N/A"
        mean_b_str = f"{r['mean_b']:.0f}" if r["mean_b"] is not None else "N/A"
        speedup_str = f"{r['speedup']:.2f}x" if r["speedup"] else "N/A"
        t_str = f"{r['t_stat']:.2f}" if not np.isnan(r.get("t_stat", float("nan"))) else "N/A"
        p_str = f"{r['p_value']:.4f}{sig}" if not np.isnan(r.get("p_value", float("nan"))) else "N/A"
        a_faster = "YES" if r["a_faster"] else ("no" if r["a_faster"] is not None else "N/A")

        print(f"  p={r['p']:>3} {r['desc']:>45} | "
              f"{mean_a_str:>8} {mean_b_str:>8} "
              f"{speedup_str:>8} {r['n_paired']:>3} "
              f"{t_str:>8} {p_str:>10} {a_faster:>10}")

    print("=" * 120)

    # Save as figure
    fig, ax = plt.subplots(figsize=(14, max(3, len(results) * 0.5 + 1.5)))
    ax.axis("off")

    col_labels = ["Prime", "Comparison", "A mean", "B mean", "Speedup",
                  "n", "t-stat", "p-value", "A faster?"]
    table_data = []
    cell_colors = []
    for r in results:
        sig = ""
        pv = r.get("p_value", float("nan"))
        if not np.isnan(pv) and pv < 0.05:
            sig = "*"
        row = [
            f"p={r['p']}",
            r["desc"][:50],
            f"{r['mean_a']:.0f}" if r["mean_a"] else "N/A",
            f"{r['mean_b']:.0f}" if r["mean_b"] else "N/A",
            f"{r['speedup']:.2f}x" if r["speedup"] else "N/A",
            str(r["n_paired"]),
            f"{r['t_stat']:.2f}" if not np.isnan(r.get("t_stat", float("nan"))) else "N/A",
            f"{pv:.4f}{sig}" if not np.isnan(pv) else "N/A",
            "YES" if r["a_faster"] else ("no" if r["a_faster"] is not None else "N/A"),
        ]
        table_data.append(row)

        if r["a_faster"] and not np.isnan(pv) and pv < 0.05:
            cell_colors.append(["#d4edda"] * len(col_labels))
        elif r["a_faster"]:
            cell_colors.append(["#fff3cd"] * len(col_labels))
        else:
            cell_colors.append(["#f8d7da"] * len(col_labels))

    if table_data:
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellColours=cell_colors, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(7)
        table.scale(1.0, 1.4)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#343a40")
            table[0, j].set_text_props(color="white", fontweight="bold")

    ax.set_title("d_head Control: Statistical Tests\n"
                 "(paired t-test, one-sided; green=sig, yellow=trend, red=A not faster)",
                 fontsize=11, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(fig_dir / "dhead_stat_tests.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dhead_stat_tests.png")

    return results


# ---- Figure 4: p=43 phase boundary analysis ----

def plot_p43_phase_boundary(cells, configs, runs, seeds, fig_dir):
    """Test accuracy at p=43 for each config: does h=1/d=32 grok at the phase boundary?"""
    p = 43
    config_keys = [(h, d) for h, d, _ in configs]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    cmap = plt.get_cmap("tab10")

    # Left: bar chart of final test_acc
    ax = axes[0]
    for j, (h, d_model) in enumerate(config_keys):
        key = (h, d_model, p)
        if key not in cells:
            continue
        accs = [a for a in cells[key]["test_accs"] if a is not None]
        if not accs:
            continue
        mean_acc = np.mean(accs)
        std_acc = np.std(accs) if len(accs) >= 2 else 0

        label = CONFIG_LABELS.get((h, d_model), f"h={h},d={d_model}").replace('\n', ' ')
        ax.bar(j, mean_acc, 0.6, color=cmap(j), alpha=0.7, edgecolor="black",
               linewidth=0.5, label=label)
        ax.errorbar(j, mean_acc, yerr=std_acc, fmt="none", color="black", capsize=5)

        for k, acc in enumerate(accs):
            ax.scatter(j + (k - 1) * 0.12, acc, s=40, color="black",
                       zorder=5, marker=["o", "s", "^"][k % 3])

    ax.set_xticks(range(len(config_keys)))
    ax.set_xticklabels([CONFIG_LABELS.get(ck, "").replace('\n', '\n') for ck in config_keys],
                       fontsize=8)
    ax.set_ylabel("Final Test Accuracy", fontsize=11)
    ax.set_title(f"p=43: Test Accuracy by Architecture Config\n"
                 f"(bars=mean, points=individual seeds)", fontsize=11, fontweight="bold")
    ax.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="Grok threshold")
    ax.axhline(1/43, color="red", linestyle=":", alpha=0.5, label="Chance")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3, axis="y")

    # Right: test_acc curves
    ax2 = axes[1]
    for j, (h, d_model, d_mlp) in enumerate(configs):
        for seed_idx, seed in enumerate(seeds):
            if (h, d_model, p, seed) not in runs:
                continue
            m = runs[(h, d_model, p, seed)]
            history = m.get("history", {})
            eval_epochs = history.get("eval_epochs", [])
            test_acc = history.get("test_acc", [])
            if eval_epochs and test_acc:
                linestyle = ["-", "--", ":"][seed_idx % 3]
                label_text = CONFIG_LABELS.get((h, d_model), "").replace('\n', ' ')
                label = f"{label_text} s={seed}" if seed_idx == 0 else None
                ax2.plot(eval_epochs, test_acc, color=cmap(j),
                         linestyle=linestyle, alpha=0.7, linewidth=1.2,
                         label=label)

    ax2.axhline(0.95, color="green", linestyle="--", alpha=0.3)
    ax2.axhline(1/43, color="red", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Test Accuracy", fontsize=11)
    ax2.set_title("p=43: Test Accuracy Curves\n(solid/dashed/dotted = seed 42/137/256)",
                  fontsize=11, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "dhead_p43_phase_boundary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dhead_p43_phase_boundary.png")


# ---- Figure 5: Decomposition summary ----

def plot_decomposition_summary(cells, configs, primes, seeds, fig_dir):
    """Summary figure: bottleneck vs capacity decomposition."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    cmap = plt.get_cmap("tab10")
    config_keys = [(h, d) for h, d, _ in configs]

    # Left: for p=113, grokking epoch by config (clear grokking expected)
    p = 113
    ax = axes[0]
    means, stds, labels, colors_list = [], [], [], []
    for j, (h, d_model) in enumerate(config_keys):
        key = (h, d_model, p)
        if key not in cells:
            continue
        epochs = [e for e in cells[key]["grok_epochs"] if e is not None]
        if epochs:
            means.append(np.mean(epochs))
            stds.append(np.std(epochs) if len(epochs) >= 2 else 0)
        else:
            means.append(0)
            stds.append(0)
        labels.append(CONFIG_LABELS.get((h, d_model), f"h={h},d={d_model}").replace('\n', '\n'))
        colors_list.append(cmap(j))

    x = np.arange(len(means))
    ax.bar(x, means, 0.6, yerr=stds, capsize=5, color=colors_list,
           edgecolor="black", linewidth=0.8)
    for i, (m, s) in enumerate(zip(means, stds)):
        if m > 0:
            ax.text(i, m + s + 200, f"{int(m)}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Grokking Epoch", fontsize=11)
    ax.set_title(f"p=113: Grokking Speed by Architecture\n(d_head=32 controlled, +d_head=64 comparison)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # Right: conceptual decomposition
    ax2 = axes[1]
    # Show: head_count effect (d_head=32 series) and d_head effect (h=1 series)
    dhead32 = [(h, d) for h, d in config_keys if d // h == 32]
    h1_series = [(h, d) for h, d in config_keys if h == 1]

    for series, label_prefix, marker, color_offset in [
        (dhead32, "d_head=32", "o", 0),
        (h1_series, "h=1", "s", 4),
    ]:
        heads_vals = []
        epoch_vals = []
        err_vals = []
        for h, d_model in series:
            key = (h, d_model, p)
            if key not in cells:
                continue
            epochs = [e for e in cells[key]["grok_epochs"] if e is not None]
            if epochs:
                heads_vals.append(h)
                epoch_vals.append(np.mean(epochs))
                err_vals.append(np.std(epochs) if len(epochs) >= 2 else 0)

        if heads_vals:
            ax2.errorbar(heads_vals, epoch_vals, yerr=err_vals,
                         fmt=f"-{marker}", markersize=8, capsize=5,
                         label=f"{label_prefix} series", linewidth=2)

    ax2.set_xlabel("Number of Heads (h)", fontsize=11)
    ax2.set_ylabel("Grokking Epoch", fontsize=11)
    ax2.set_title("Decomposition: Head Count vs Per-Head Capacity\n"
                  "(p=113)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks([1, 2, 4])

    fig.tight_layout()
    fig.savefig(fig_dir / "dhead_decomposition.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: dhead_decomposition.png")


# ---- Summary table ----

def print_summary_table(cells, configs, primes, seeds):
    config_keys = [(h, d) for h, d, _ in configs]
    print("\n" + "=" * 110)
    print("d_head CONTROL -- SUMMARY (mean +/- std across seeds)")
    print("=" * 110)

    for p in primes:
        n_train = int(p * p * 0.3)
        print(f"\n--- p={p} (N_train={n_train}) ---")

        for h, d_model in config_keys:
            d_head = d_model // h
            d_mlp = 4 * d_model
            params = estimate_params(h, d_model, d_mlp, p)
            key = (h, d_model, p)
            if key not in cells:
                print(f"  h={h} d={d_model} d_head={d_head} d_mlp={d_mlp} params={params:,}: MISSING")
                continue

            epochs = [e for e in cells[key]["grok_epochs"] if e is not None]
            accs = [a for a in cells[key]["test_accs"] if a is not None]
            ginis = [g for g in cells[key]["ginis"] if g is not None]
            n_total = len(accs)

            epoch_str = (f"{np.mean(epochs):.0f} +/- {np.std(epochs):.0f} ({len(epochs)}/{n_total} grok)"
                         if epochs else f"-- (0/{n_total} grok)")
            acc_str = f"acc={np.mean(accs):.3f}" if accs else "acc=N/A"
            gini_str = f"gini={np.mean(ginis):.3f}" if ginis else "gini=N/A"

            print(f"  h={h} d={d_model} d_head={d_head} d_mlp={d_mlp} params={params:,}: "
                  f"grok_epoch={epoch_str}, {acc_str}, {gini_str}")

    print("=" * 110)


# ---- JSON output ----

def save_metrics_json(cells, configs, primes, seeds, stat_results, fig_dir):
    output = {
        "experiment": "d_head_control",
        "sweep_config": {
            "configs": [{"n_heads": h, "d_model": d, "d_mlp": m} for h, d, m in configs],
            "primes": primes,
            "seeds": seeds,
        },
        "cells": {},
        "statistical_tests": [],
    }

    for (h, d_model, p), c in cells.items():
        key = f"h{h}_d{d_model}_p{p}"
        epochs = [e for e in c["grok_epochs"] if e is not None]
        accs = [a for a in c["test_accs"] if a is not None]
        ginis = [g for g in c["ginis"] if g is not None]
        d_head = d_model // h
        d_mlp = 4 * d_model
        output["cells"][key] = {
            "n_heads": h,
            "d_model": d_model,
            "d_head": d_head,
            "d_mlp": d_mlp,
            "p": p,
            "params_est": estimate_params(h, d_model, d_mlp, p),
            "grok_epochs": c["grok_epochs"],
            "grok_epoch_mean": float(np.mean(epochs)) if epochs else None,
            "grok_epoch_std": float(np.std(epochs)) if len(epochs) >= 2 else None,
            "n_grokked": len(epochs),
            "n_seeds": len(accs),
            "test_acc_mean": float(np.mean(accs)) if accs else None,
            "gini_mean": float(np.mean(ginis)) if ginis else None,
        }

    for r in stat_results:
        output["statistical_tests"].append({
            "p": r["p"],
            "config_a": r["config_a"],
            "config_b": r["config_b"],
            "desc": r["desc"],
            "mean_a": r["mean_a"],
            "mean_b": r["mean_b"],
            "speedup": r["speedup"],
            "t_stat": r["t_stat"] if not np.isnan(r.get("t_stat", float("nan"))) else None,
            "p_value": r["p_value"] if not np.isnan(r.get("p_value", float("nan"))) else None,
            "a_faster": bool(r["a_faster"]) if r["a_faster"] is not None else None,
        })

    out_path = fig_dir / "dhead_control_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare d_head control experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: results/dhead_comparison/)")
    parser.add_argument("--primes", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES
    seeds = args.seeds if args.seeds else SWEEP_SEEDS
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "dhead_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, SWEEP_CONFIGS, primes, seeds, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    cells = extract_cell_data(runs, SWEEP_CONFIGS, primes, seeds)

    logger.info("1/5: Grokking epoch bar chart")
    plot_grokking_bar_chart(cells, SWEEP_CONFIGS, primes, seeds, fig_dir)

    logger.info("2/5: Parameter efficiency scatter")
    plot_param_efficiency(cells, SWEEP_CONFIGS, primes, seeds, fig_dir)

    logger.info("3/5: Statistical tests")
    stat_results = compute_statistical_tests(cells, SWEEP_CONFIGS, primes, seeds, fig_dir)

    logger.info("4/5: p=43 phase boundary analysis")
    if 43 in primes:
        plot_p43_phase_boundary(cells, SWEEP_CONFIGS, runs, seeds, fig_dir)
    else:
        logger.info("  Skipping (p=43 not in sweep)")

    logger.info("5/5: Decomposition summary")
    plot_decomposition_summary(cells, SWEEP_CONFIGS, primes, seeds, fig_dir)

    print_summary_table(cells, SWEEP_CONFIGS, primes, seeds)
    save_metrics_json(cells, SWEEP_CONFIGS, primes, seeds, stat_results, fig_dir)

    # Verdict
    print("\n" + "=" * 80)
    print("VERDICT: BOTTLENECK vs CAPACITY")
    print("=" * 80)

    for p in primes:
        print(f"\np={p}:")
        key_h1d32 = (1, 32, p)
        key_h4d128 = (4, 128, p)
        key_h1d64 = (1, 64, p)

        for key, label in [(key_h1d32, "h=1/d=32"), (key_h4d128, "h=4/d=128"), (key_h1d64, "h=1/d=64")]:
            if key in cells:
                epochs = [e for e in cells[key]["grok_epochs"] if e is not None]
                accs = [a for a in cells[key]["test_accs"] if a is not None]
                n_grok = len(epochs)
                n_total = len(accs)
                if epochs:
                    print(f"  {label}: grok {np.mean(epochs):.0f} +/- {np.std(epochs):.0f} ({n_grok}/{n_total})")
                else:
                    mean_acc = np.mean(accs) if accs else 0
                    print(f"  {label}: NO GROK ({n_grok}/{n_total}), mean_acc={mean_acc:.3f}")

        if key_h1d32 in cells and key_h4d128 in cells:
            e_h1 = [e for e in cells[key_h1d32]["grok_epochs"] if e is not None]
            e_h4 = [e for e in cells[key_h4d128]["grok_epochs"] if e is not None]
            if e_h1 and e_h4:
                if np.mean(e_h1) < np.mean(e_h4):
                    print(f"  -> h=1/d=32 FASTER despite 50x fewer params: BOTTLENECK STORY WINS")
                else:
                    print(f"  -> h=4/d=128 faster: per-head CAPACITY matters")
            elif e_h1 and not e_h4:
                print(f"  -> h=1/d=32 groks but h=4/d=128 does NOT: BOTTLENECK STORY WINS")
            elif not e_h1 and e_h4:
                print(f"  -> h=4/d=128 groks but h=1/d=32 does NOT: CAPACITY matters")
            else:
                print(f"  -> Neither groks: inconclusive")

    print("=" * 80)
    logger.info(f"\nAll figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
