#!/usr/bin/env python
"""Cross-run comparison plots for the effective weight-decay (wd × lr) unification experiment.

Tests whether grokking dynamics are controlled purely by eff_wd = wd × lr,
or whether wd and lr have independent effects beyond their product.

Loads all relevant existing runs (WD sweep, LR sweep) plus 4 new runs, and
produces 5 figures testing the unification hypothesis.
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

from src.utils import setup_logging


# All (wd, lr) pairs to analyse, grouped by eff_wd = wd × lr
ALL_RUNS = [
    # eff_wd = 1e-4
    (0.1,  1e-3),   # WD sweep
    (1.0,  1e-4),   # LR sweep
    # eff_wd = 3e-4
    (0.3,  1e-3),   # WD sweep
    (1.0,  3e-4),   # LR sweep
    # eff_wd = 5e-4
    (0.5,  1e-3),   # WD sweep
    # eff_wd = 1e-3  ← KEY TEST (3 decompositions)
    (1.0,  1e-3),   # baseline
    (2.0,  5e-4),   # NEW
    (0.5,  2e-3),   # NEW
    # eff_wd = 2e-3  ← KEY TEST
    (2.0,  1e-3),   # WD sweep
    (1.0,  2e-3),   # NEW
    # eff_wd = 3e-3  ← KEY TEST
    (1.0,  3e-3),   # LR sweep
    (3.0,  1e-3),   # NEW
    # eff_wd = 5e-3
    (5.0,  1e-3),   # WD sweep
]

# eff_wd values with multiple decompositions (key test groups)
KEY_EFF_WD_GROUPS = {
    1e-3: [(1.0, 1e-3), (2.0, 5e-4), (0.5, 2e-3)],
    2e-3: [(2.0, 1e-3), (1.0, 2e-3)],
    3e-3: [(1.0, 3e-3), (3.0, 1e-3)],
}

NEVER_SENTINEL = 50_000


def _run_id_for(wd: float, lr: float) -> str:
    base = f"p113_d128_h4_mlp512_L1_wd{wd}_s42"
    if abs(lr - 1e-3) > 1e-9:
        return f"{base}_lr{lr}"
    return base


def load_all_runs(results_root, run_pairs, logger):
    """Load metrics for all (wd, lr) pairs. Returns {(wd, lr): metrics}."""
    runs = {}
    for wd, lr in run_pairs:
        rid = _run_id_for(wd, lr)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning(f"Missing: {metrics_path}")
            continue
        with open(metrics_path) as f:
            runs[(wd, lr)] = json.load(f)
        logger.info(f"Loaded (wd={wd}, lr={lr:.0e}): {rid}")
    return runs


def find_grokking_epoch(history, threshold=0.95):
    """Find first epoch where test acc >= threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def _get_run_style(wd: float, lr: float) -> dict:
    """Return plot style dict for a (wd, lr) pair."""
    # Color by wd value
    wd_palette = {
        0.1: "#aec7e8",
        0.3: "#1f77b4",
        0.5: "#9467bd",
        1.0: "#2ca02c",
        2.0: "#ff7f0e",
        3.0: "#d62728",
        5.0: "#8c564b",
    }
    # Marker by lr
    lr_markers = {
        1e-4:  "v",
        3e-4:  "<",
        5e-4:  "D",
        1e-3:  "o",
        2e-3:  "s",
        3e-3:  "^",
        1e-2:  ">",
    }
    color = wd_palette.get(wd, "#777777")
    marker = lr_markers.get(lr, "x")
    label = f"wd={wd}, lr={lr:.0e}"
    return {"color": color, "marker": marker, "label": label}


def plot_unification_curve(runs, fig_dir):
    """Figure 1: Grokking epoch vs eff_wd (log-log), colored by wd, marker by lr."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for (wd, lr), metrics in runs.items():
        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        eff_wd = wd * lr

        y = grok_epoch if grok_epoch is not None else NEVER_SENTINEL
        style = _get_run_style(wd, lr)
        never = grok_epoch is None

        ax.scatter(
            eff_wd, y,
            color=style["color"],
            marker=style["marker"],
            s=120,
            zorder=3,
            edgecolors="black",
            linewidths=0.8,
            alpha=0.7 if never else 1.0,
            label=style["label"],
        )
        if never:
            ax.annotate(
                "Never", (eff_wd, y),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, color=style["color"],
            )

    # Dashed line at NEVER_SENTINEL
    ax.axhline(NEVER_SENTINEL, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(2e-5, NEVER_SENTINEL * 1.05, "Never grokked sentinel", fontsize=8, color="gray")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Effective weight decay  (eff_wd = wd × lr)", fontsize=12)
    ax.set_ylabel("Grokking epoch (test acc ≥ 95%)", fontsize=12)
    ax.set_title(
        "Effective WD Unification — Does eff_wd = wd × lr control grokking?\n"
        "(color = wd value, marker shape = lr value; same x → same eff_wd)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, which="both")

    # Custom legend: avoid duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    unique = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    if unique:
        ax.legend(*zip(*unique), fontsize=8, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(fig_dir / "effwd_unification_curve.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: effwd_unification_curve.png")


def plot_grokking_bar_chart(runs, fig_dir):
    """Figure 2: Grouped bar chart of grokking epoch by eff_wd, bars colored by (wd, lr)."""
    # Collect data per eff_wd level
    from collections import defaultdict
    data = defaultdict(list)
    for (wd, lr), metrics in runs.items():
        eff_wd = wd * lr
        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        data[eff_wd].append((wd, lr, grok_epoch))

    eff_wd_levels = sorted(data.keys())
    fig, ax = plt.subplots(figsize=(13, 6))

    x_pos = 0
    x_ticks = []
    x_tick_labels = []
    group_gap = 1.0
    bar_width = 0.6

    for eff_wd in eff_wd_levels:
        entries = data[eff_wd]
        group_center = x_pos + (len(entries) - 1) * bar_width / 2

        for wd, lr, grok_epoch in entries:
            style = _get_run_style(wd, lr)
            y = grok_epoch if grok_epoch is not None else NEVER_SENTINEL
            color = style["color"]
            hatch = "///" if grok_epoch is None else None

            bar = ax.bar(x_pos, y, bar_width * 0.85, color=color,
                         edgecolor="black", linewidth=0.8, alpha=0.85, hatch=hatch)
            label_text = f"wd={wd}\nlr={lr:.0e}"
            ax.text(x_pos, y + NEVER_SENTINEL * 0.01, label_text,
                    ha="center", va="bottom", fontsize=6.5, fontweight="bold")
            x_pos += bar_width

        x_ticks.append(group_center)
        x_tick_labels.append(f"eff={eff_wd:.0e}")
        x_pos += group_gap

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_tick_labels, fontsize=9)
    ax.set_ylabel("Grokking epoch (test acc ≥ 95%)", fontsize=11)
    ax.set_xlabel("Effective weight decay (wd × lr)", fontsize=11)
    ax.axhline(NEVER_SENTINEL, color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.text(0, NEVER_SENTINEL * 1.02, f"Never (sentinel={NEVER_SENTINEL})", fontsize=8, color="gray")
    ax.set_title(
        "Grokking Epoch by eff_wd Group\n"
        "(bars at same x-group share eff_wd; spread within group = unification gap)",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "effwd_grokking_bar_chart.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: effwd_grokking_bar_chart.png")


def plot_test_acc_per_group(runs, fig_dir):
    """Figure 3: Test accuracy curves overlaid per eff_wd group (3 subplots)."""
    n_groups = len(KEY_EFF_WD_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), sharey=True)

    cmap = plt.get_cmap("tab10")

    for ax, (eff_wd, pairs) in zip(axes, sorted(KEY_EFF_WD_GROUPS.items())):
        for idx, (wd, lr) in enumerate(pairs):
            key = (wd, lr)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            epochs = history.get("eval_epochs", [])
            test_acc = history.get("test_acc", [])
            if epochs and test_acc:
                style = _get_run_style(wd, lr)
                ax.plot(epochs, test_acc, color=cmap(idx), linewidth=1.8,
                        label=f"wd={wd}, lr={lr:.0e}")

        ax.set_title(f"eff_wd = {eff_wd:.0e}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    fig.suptitle(
        "Test Accuracy Curves per eff_wd Group\n"
        "(if unification holds, curves should overlap)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "effwd_test_acc_per_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: effwd_test_acc_per_group.png")


def plot_gini_per_group(runs, fig_dir):
    """Figure 4: Gini evolution overlaid per eff_wd group (3 subplots)."""
    n_groups = len(KEY_EFF_WD_GROUPS)
    fig, axes = plt.subplots(1, n_groups, figsize=(6 * n_groups, 5), sharey=True)

    cmap = plt.get_cmap("tab10")

    for ax, (eff_wd, pairs) in zip(axes, sorted(KEY_EFF_WD_GROUPS.items())):
        for idx, (wd, lr) in enumerate(pairs):
            key = (wd, lr)
            if key not in runs:
                continue
            history = runs[key].get("history", {})
            fourier_epochs = history.get("fourier_epochs", [])
            gini = history.get("gini", [])
            if fourier_epochs and gini:
                cmap_idx = idx
                ax.plot(fourier_epochs, gini, color=cmap(cmap_idx), linewidth=1.8,
                        label=f"wd={wd}, lr={lr:.0e}")

        ax.set_title(f"eff_wd = {eff_wd:.0e}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")

    axes[0].set_ylabel("Gini Coefficient (Fourier Sparsity)", fontsize=11)
    fig.suptitle(
        "Gini Evolution per eff_wd Group\n"
        "(if unification holds, curves should overlap)",
        fontsize=13, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "effwd_gini_per_group.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: effwd_gini_per_group.png")


def plot_theory_vs_observation(runs, fig_dir):
    """Figure 5: Predicted (from baseline lr=1e-3 curve) vs actual grokking epoch.

    Baseline: grokking epoch as a function of eff_wd from lr=1e-3 runs only.
    For non-baseline runs, we predict by interpolating the baseline curve at eff_wd = wd*lr.
    Points on diagonal → unification; deviations reveal wd/lr interplay.
    """
    # Build baseline curve from lr=1e-3 runs
    baseline_eff_wds = []
    baseline_grok = []
    for (wd, lr), metrics in runs.items():
        if abs(lr - 1e-3) < 1e-9:
            eff_wd = wd * lr
            history = metrics.get("history", {})
            grok_epoch = find_grokking_epoch(history)
            y = grok_epoch if grok_epoch is not None else NEVER_SENTINEL
            baseline_eff_wds.append(eff_wd)
            baseline_grok.append(y)

    if len(baseline_eff_wds) < 2:
        print("  Skipping theory vs observation (too few baseline runs)")
        return

    # Sort baseline
    order = np.argsort(baseline_eff_wds)
    bx = np.array(baseline_eff_wds)[order]
    by = np.array(baseline_grok)[order]

    fig, ax = plt.subplots(figsize=(7, 7))

    # Plot non-baseline runs
    plotted_any = False
    for (wd, lr), metrics in runs.items():
        if abs(lr - 1e-3) < 1e-9:
            continue  # skip baseline runs themselves
        eff_wd = wd * lr
        history = metrics.get("history", {})
        actual_grok = find_grokking_epoch(history)
        actual = actual_grok if actual_grok is not None else NEVER_SENTINEL

        # Predict by interpolating baseline
        if eff_wd < bx[0] or eff_wd > bx[-1]:
            predicted = float(np.interp(np.log(eff_wd), np.log(bx), by))
        else:
            predicted = float(np.interp(eff_wd, bx, by))

        style = _get_run_style(wd, lr)
        ax.scatter(predicted, actual, color=style["color"], marker=style["marker"],
                   s=150, zorder=3, edgecolors="black", linewidths=0.8,
                   label=style["label"])
        plotted_any = True

    if not plotted_any:
        print("  Skipping theory vs observation (no non-baseline runs found)")
        plt.close(fig)
        return

    # Diagonal line
    all_vals = []
    for ax_obj in [ax]:
        for (wd, lr), metrics in runs.items():
            if abs(lr - 1e-3) < 1e-9:
                continue
            history = metrics.get("history", {})
            grok_epoch = find_grokking_epoch(history)
            all_vals.append(grok_epoch if grok_epoch is not None else NEVER_SENTINEL)
    if all_vals:
        vmin = min(all_vals) * 0.5
        vmax = max(all_vals) * 2.0
        diag = np.array([max(100, vmin), min(NEVER_SENTINEL * 1.5, vmax)])
        ax.plot(diag, diag, "k--", linewidth=1, alpha=0.5, label="y = x (perfect unification)")

    ax.set_xlabel("Predicted grokking epoch (from baseline lr=1e-3 curve)", fontsize=11)
    ax.set_ylabel("Actual grokking epoch", fontsize=11)
    ax.set_title(
        "Effective WD Unification: Theory vs Observation\n"
        "(points on diagonal = grokking controlled purely by eff_wd = wd × lr)",
        fontsize=11,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(fig_dir / "effwd_theory_vs_observation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: effwd_theory_vs_observation.png")


def print_summary(runs, logger):
    """Print console summary table."""
    print("\n" + "=" * 100)
    print("EFFECTIVE WD UNIFICATION — SUMMARY")
    print("=" * 100)
    header = (f"{'eff_wd':>8} | {'wd':>5} | {'lr':>8} | {'Grok Epoch':>10} | "
              f"{'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | {'W Norm':>8}")
    print(header)
    print("-" * 100)

    from collections import defaultdict
    groups = defaultdict(list)
    for (wd, lr), metrics in runs.items():
        eff_wd = wd * lr
        groups[eff_wd].append((wd, lr, metrics))

    for eff_wd in sorted(groups.keys()):
        for wd, lr, metrics in groups[eff_wd]:
            history = metrics.get("history", {})
            grok_epoch = find_grokking_epoch(history)
            grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
            train_acc = metrics.get("final_train_acc", 0)
            test_acc = metrics.get("final_test_acc", 0)
            gini = metrics.get("final_gini", 0)
            w_norm = metrics.get("final_weight_norm", 0)
            print(f"{eff_wd:>8.0e} | {wd:>5} | {lr:>8.0e} | {grok_str:>10} | "
                  f"{train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {w_norm:>8.1f}")
        print("-" * 100)

    print("=" * 100)

    # Unification gap per key group
    print("\n| UNIFICATION CHECK — spread of grokking epochs within each eff_wd group:")
    for eff_wd, pairs in sorted(KEY_EFF_WD_GROUPS.items()):
        grok_epochs = []
        for wd, lr in pairs:
            key = (wd, lr)
            if key in runs:
                history = runs[key].get("history", {})
                ge = find_grokking_epoch(history)
                if ge is not None:
                    grok_epochs.append((wd, lr, ge))
        if len(grok_epochs) >= 2:
            epochs_only = [ge for _, _, ge in grok_epochs]
            spread = max(epochs_only) - min(epochs_only)
            rel_spread = spread / np.mean(epochs_only) * 100
            print(f"|   eff_wd={eff_wd:.0e}: epochs={[ge for _,_,ge in grok_epochs]}, "
                  f"spread={spread} ({rel_spread:.0f}%)")
        else:
            present = [(wd, lr) for wd, lr in pairs if (wd, lr) in runs]
            missing = [(wd, lr) for wd, lr in pairs if (wd, lr) not in runs]
            print(f"|   eff_wd={eff_wd:.0e}: insufficient data (present={present}, missing={missing})")
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare effective WD unification results")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    fig_dir = Path(args.output_dir) if args.output_dir else results_root / "effective_wd_comparison"
    fig_dir.mkdir(parents=True, exist_ok=True)

    runs = load_all_runs(results_root, ALL_RUNS, logger)
    if not runs:
        logger.error("No runs found! Run the sweep first.")
        return

    logger.info(f"Loaded {len(runs)}/{len(ALL_RUNS)} runs")

    logger.info("1/5: Unification curve (key plot)")
    plot_unification_curve(runs, fig_dir)

    logger.info("2/5: Grokking epoch bar chart by eff_wd group")
    plot_grokking_bar_chart(runs, fig_dir)

    logger.info("3/5: Test accuracy per key eff_wd group")
    plot_test_acc_per_group(runs, fig_dir)

    logger.info("4/5: Gini evolution per key eff_wd group")
    plot_gini_per_group(runs, fig_dir)

    logger.info("5/5: Theory vs observation scatter")
    plot_theory_vs_observation(runs, fig_dir)

    print_summary(runs, logger)
    logger.info(f"\nAll comparison figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
