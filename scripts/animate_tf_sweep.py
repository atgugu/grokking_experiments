#!/usr/bin/env python3
"""Animated GIF comparing train fraction sweep runs on modular addition.

2x2 panels:
  A) Test accuracy (all runs overlaid, 0-1 y-axis, 95% threshold)
  B) Train loss (log-y, all runs overlaid)
  C) Gini coefficient (Fourier sparsity, overlaid)
  D) Key frequency energy fraction (top-5 energy / total excl DC)

Usage:
    python scripts/animate_tf_sweep.py
    python scripts/animate_tf_sweep.py --frame-step 200 --fps 15
    python scripts/animate_tf_sweep.py --tf-values 0.3 0.4 0.5 0.7
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

logger = logging.getLogger(__name__)

DEFAULT_TF_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUTPUT = "results/tf_sweep_comparison/tf_sweep_animation.gif"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _run_id(tf: float) -> str:
    """Return the run directory name for a given train fraction."""
    if tf == 0.3:
        return "p113_d128_h4_mlp512_L1_wd1.0_s42"
    return f"p113_d128_h4_mlp512_L1_wd1.0_s42_tf{tf}"


def load_all_runs(results_root: Path, tf_values: list[float]) -> dict:
    """Load metrics.json for each train fraction run."""
    runs = {}
    for tf in tf_values:
        rid = _run_id(tf)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Missing: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            runs[tf] = json.load(f)
        logger.info("Loaded tf=%s: %s", tf, rid)
    return runs


def load_fourier_data(results_root: Path, tf_values: list[float]) -> dict:
    """Load fourier_snapshots.npz for each train fraction run."""
    fourier = {}
    for tf in tf_values:
        rid = _run_id(tf)
        snap_path = results_root / rid / "fourier_snapshots.npz"
        if not snap_path.exists():
            logger.warning("Missing Fourier data: %s", snap_path)
            continue
        fourier[tf] = dict(np.load(snap_path))
        logger.info("Loaded Fourier tf=%s (%d snapshots)", tf, len(fourier[tf]["fourier_epochs"]))
    return fourier


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def compute_key_energy_fraction(frequency_norms: np.ndarray, n_top: int = 5) -> np.ndarray:
    """Per-snapshot fraction of energy in top-n frequencies (excluding DC).

    Args:
        frequency_norms: (n_snapshots, p) array of per-frequency norms.
        n_top: number of top frequencies to sum.

    Returns:
        (n_snapshots,) array of energy fractions in [0, 1].
    """
    # Exclude DC component (index 0)
    norms = frequency_norms[:, 1:].copy()
    total = norms.sum(axis=1)
    # Sort descending per snapshot, take top-n
    sorted_norms = np.sort(norms, axis=1)[:, ::-1]
    top_energy = sorted_norms[:, :n_top].sum(axis=1)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, top_energy / total, 0.0)
    return frac


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def get_tf_style(tf_values: list[float], reference_tf: float = 0.3) -> dict:
    """Return {tf: {"color": ..., "linewidth": ..., "zorder": ...}} using tab10."""
    cmap = plt.get_cmap("tab10")
    styles = {}
    for i, tf in enumerate(tf_values):
        is_ref = abs(tf - reference_tf) < 1e-9
        styles[tf] = {
            "color": cmap(i % 10),
            "linewidth": 3.0 if is_ref else 1.5,
            "zorder": 10 if is_ref else 2,
            "label": f"tf={tf}",
        }
    return styles


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_run_data(runs: dict, fourier_data: dict) -> dict:
    """Convert loaded data to numpy arrays and compute derived quantities.

    Returns dict keyed by tf with sub-dicts:
        eval_epochs, test_acc, train_loss, fourier_epochs, gini, key_energy_frac
    """
    processed = {}
    for tf, metrics in runs.items():
        h = metrics["history"]
        entry = {
            "eval_epochs": np.array(h["eval_epochs"], dtype=np.float64),
            "test_acc": np.array(h["test_acc"], dtype=np.float64),
            "train_loss": np.array(h["train_loss"], dtype=np.float64),
            "fourier_epochs": np.array(h["fourier_epochs"], dtype=np.float64),
            "gini": np.array(h["gini"], dtype=np.float64),
        }
        if tf in fourier_data:
            fn = fourier_data[tf]["frequency_norms"]
            entry["key_energy_frac"] = compute_key_energy_fraction(fn)
            entry["key_energy_epochs"] = np.array(
                fourier_data[tf]["fourier_epochs"], dtype=np.float64
            )
        processed[tf] = entry
    return processed


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------

def build_common_timeline(run_data: dict, frame_step: int = 100) -> np.ndarray:
    """Build epoch array spanning all runs."""
    max_epoch = max(
        rd["eval_epochs"][-1] for rd in run_data.values()
    )
    return np.arange(0, max_epoch + 1, frame_step)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def create_tf_sweep_animation(
    run_data: dict,
    styles: dict,
    common_epochs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 100,
) -> FuncAnimation:
    """Build and optionally save the 2x2 train fraction sweep animation."""

    sorted_tfs = sorted(run_data.keys())
    n_frames = len(common_epochs)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax_acc, ax_loss = axes[0]
    ax_gini, ax_energy = axes[1]

    # --- Panel A: Test Accuracy ---
    ax_acc.set_title("Test Accuracy", fontsize=12, fontweight="bold")
    ax_acc.set_xlabel("Epoch", fontsize=10)
    ax_acc.set_ylabel("Accuracy", fontsize=10)
    ax_acc.set_xlim(0, common_epochs[-1])
    ax_acc.set_ylim(-0.05, 1.05)
    ax_acc.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)
    ax_acc.grid(True, alpha=0.3)

    # --- Panel B: Train Loss ---
    ax_loss.set_title("Train Loss", fontsize=12, fontweight="bold")
    ax_loss.set_xlabel("Epoch", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    ax_loss.set_xlim(0, common_epochs[-1])
    ax_loss.set_yscale("log")
    ax_loss.grid(True, alpha=0.3, which="both")

    # --- Panel C: Gini Coefficient ---
    ax_gini.set_title("Gini Coefficient (Fourier Sparsity)", fontsize=12, fontweight="bold")
    ax_gini.set_xlabel("Epoch", fontsize=10)
    ax_gini.set_ylabel("Gini", fontsize=10)
    ax_gini.set_xlim(0, common_epochs[-1])
    ax_gini.set_ylim(-0.05, 1.05)
    ax_gini.grid(True, alpha=0.3)

    # --- Panel D: Key Frequency Energy ---
    ax_energy.set_title("Key Freq Energy Fraction (top-5)", fontsize=12, fontweight="bold")
    ax_energy.set_xlabel("Epoch", fontsize=10)
    ax_energy.set_ylabel("Energy Fraction", fontsize=10)
    ax_energy.set_xlim(0, common_epochs[-1])
    ax_energy.set_ylim(-0.05, 1.05)
    ax_energy.grid(True, alpha=0.3)

    # Create line artists (sorted so reference tf drawn last = on top)
    draw_order = sorted(sorted_tfs, key=lambda t: styles[t]["zorder"])

    acc_lines = {}
    loss_lines = {}
    gini_lines = {}
    energy_lines = {}
    for tf in draw_order:
        s = styles[tf]
        kw = dict(color=s["color"], linewidth=s["linewidth"], zorder=s["zorder"], label=s["label"])
        (acc_lines[tf],) = ax_acc.plot([], [], **kw)
        (loss_lines[tf],) = ax_loss.plot([], [], **kw)
        (gini_lines[tf],) = ax_gini.plot([], [], **kw)
        (energy_lines[tf],) = ax_energy.plot([], [], **kw)

    # Vertical epoch markers
    marker_kw = dict(color="gray", linestyle="--", alpha=0.6, linewidth=1)
    markers = [ax.axvline(0, **marker_kw) for ax in [ax_acc, ax_loss, ax_gini, ax_energy]]

    # Legends
    for ax in [ax_acc, ax_loss, ax_gini, ax_energy]:
        ax.legend(fontsize=7, ncol=2, loc="best")

    suptitle = fig.suptitle("Train Fraction Sweep — Epoch 0", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame_idx):
        current_epoch = common_epochs[frame_idx]
        suptitle.set_text(f"Train Fraction Sweep — Epoch {int(current_epoch)}")

        for tf in sorted_tfs:
            rd = run_data[tf]

            # Panels A & B: mask on eval_epochs
            mask_eval = rd["eval_epochs"] <= current_epoch
            if mask_eval.any():
                ep = rd["eval_epochs"][mask_eval]
                acc_lines[tf].set_data(ep, rd["test_acc"][mask_eval])
                loss_lines[tf].set_data(ep, rd["train_loss"][mask_eval])
            else:
                acc_lines[tf].set_data([], [])
                loss_lines[tf].set_data([], [])

            # Panel C: mask on fourier_epochs
            mask_fourier = rd["fourier_epochs"] <= current_epoch
            if mask_fourier.any():
                gini_lines[tf].set_data(
                    rd["fourier_epochs"][mask_fourier],
                    rd["gini"][mask_fourier],
                )
            else:
                gini_lines[tf].set_data([], [])

            # Panel D: key energy fraction
            if "key_energy_frac" in rd:
                mask_ke = rd["key_energy_epochs"] <= current_epoch
                if mask_ke.any():
                    energy_lines[tf].set_data(
                        rd["key_energy_epochs"][mask_ke],
                        rd["key_energy_frac"][mask_ke],
                    )
                else:
                    energy_lines[tf].set_data([], [])

        # Move epoch markers
        for m in markers:
            m.set_xdata([current_epoch, current_epoch])

        # Auto-scale loss y-axis based on visible data
        all_loss = []
        for tf in sorted_tfs:
            rd = run_data[tf]
            mask = rd["eval_epochs"] <= current_epoch
            if mask.any():
                all_loss.append(rd["train_loss"][mask])
        if all_loss:
            concat = np.concatenate(all_loss)
            lo, hi = concat.min(), concat.max()
            if lo > 0 and hi > 0:
                ax_loss.set_ylim(lo * 0.5, hi * 2.0)

        return []

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000 // fps, blit=False
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Saving animation to %s (%d frames, %d fps) ...", out, n_frames, fps)
        anim.save(str(out), writer="pillow", fps=fps, dpi=dpi)
        logger.info("Saved: %s", out)

    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate train fraction sweep comparison (2x2 panels)"
    )
    parser.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS_DIR,
        help="Root results directory containing per-run folders",
    )
    parser.add_argument(
        "--output", type=str, default=DEFAULT_OUTPUT,
        help="Output GIF path",
    )
    parser.add_argument(
        "--tf-values", type=float, nargs="+", default=DEFAULT_TF_VALUES,
        help="Train fraction values to include",
    )
    parser.add_argument(
        "--frame-step", type=int, default=100,
        help="Epoch step between frames (lower = more frames)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=[14, 10],
        help="Figure size (width height)",
    )
    parser.add_argument(
        "--reference-tf", type=float, default=0.3,
        help="Train fraction value to highlight (thicker line)",
    )
    parser.add_argument("--dpi", type=int, default=100, help="Output DPI")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results_root = args.results_dir
    tf_values = sorted(args.tf_values)

    logger.info("Loading runs for tf=%s from %s", tf_values, results_root)
    runs = load_all_runs(results_root, tf_values)
    fourier_data = load_fourier_data(results_root, list(runs.keys()))

    if not runs:
        logger.error("No runs found. Check --results-dir and --tf-values.")
        return

    available_tfs = sorted(runs.keys())
    logger.info("Available runs: %s", available_tfs)

    styles = get_tf_style(available_tfs, reference_tf=args.reference_tf)
    run_data = preprocess_run_data(runs, fourier_data)
    common_epochs = build_common_timeline(run_data, frame_step=args.frame_step)
    logger.info("Timeline: %d frames (step=%d, max_epoch=%d)", len(common_epochs), args.frame_step, int(common_epochs[-1]))

    create_tf_sweep_animation(
        run_data=run_data,
        styles=styles,
        common_epochs=common_epochs,
        fps=args.fps,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
