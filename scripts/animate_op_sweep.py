#!/usr/bin/env python3
"""Animated GIF comparing operation sweep runs on modular arithmetic.

2x2 panels:
  A) Test accuracy (all runs overlaid, 0-1 y-axis, 95% threshold)
  B) Train loss (log-y, all runs overlaid)
  C) Gini coefficient (Fourier sparsity, overlaid)
  D) Key frequency energy fraction (top-5 energy / total excl DC)

Usage:
    python scripts/animate_op_sweep.py
    python scripts/animate_op_sweep.py --frame-step 200 --fps 15
    python scripts/animate_op_sweep.py --operations addition multiplication x2_plus_y2
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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import _OP_SUFFIXES

logger = logging.getLogger(__name__)

SWEEP_OPERATIONS = ["addition", "subtraction", "multiplication", "x2_plus_y2", "x3_plus_xy"]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUTPUT = "results/op_sweep_comparison/op_sweep_animation.gif"

OP_LABELS = {
    "addition": "a + b",
    "subtraction": "a \u2212 b",
    "multiplication": "a \u00d7 b",
    "x2_plus_y2": "a\u00b2 + b\u00b2",
    "x3_plus_xy": "a\u00b3 + ab",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _run_id(op: str) -> str:
    """Return the run directory name for a given operation."""
    suffix = _OP_SUFFIXES.get(op)
    base = "p113_d128_h4_mlp512_L1_wd1.0_s42"
    if suffix is not None:
        return f"{base}_{suffix}"
    return base


def load_all_runs(results_root: Path, operations: list[str]) -> dict:
    """Load metrics.json for each operation run."""
    runs = {}
    for op in operations:
        rid = _run_id(op)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Missing: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            runs[op] = json.load(f)
        logger.info("Loaded %s: %s", op, rid)
    return runs


def load_fourier_data(results_root: Path, operations: list[str]) -> dict:
    """Load fourier_snapshots.npz for each operation run."""
    fourier = {}
    for op in operations:
        rid = _run_id(op)
        snap_path = results_root / rid / "fourier_snapshots.npz"
        if not snap_path.exists():
            logger.warning("Missing Fourier data: %s", snap_path)
            continue
        fourier[op] = dict(np.load(snap_path))
        logger.info("Loaded Fourier %s (%d snapshots)", op, len(fourier[op]["fourier_epochs"]))
    return fourier


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def compute_key_energy_fraction(frequency_norms: np.ndarray, n_top: int = 5) -> np.ndarray:
    """Per-snapshot fraction of energy in top-n frequencies (excluding DC)."""
    norms = frequency_norms[:, 1:].copy()
    total = norms.sum(axis=1)
    sorted_norms = np.sort(norms, axis=1)[:, ::-1]
    top_energy = sorted_norms[:, :n_top].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(total > 0, top_energy / total, 0.0)
    return frac


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def get_op_style(operations: list[str], reference_op: str = "addition") -> dict:
    """Return {op: {"color": ..., "linewidth": ..., "zorder": ...}} using tab10."""
    cmap = plt.get_cmap("tab10")
    # Use consistent color indices from SWEEP_OPERATIONS
    op_to_idx = {op: i for i, op in enumerate(SWEEP_OPERATIONS)}
    styles = {}
    for op in operations:
        is_ref = (op == reference_op)
        idx = op_to_idx.get(op, len(op_to_idx))
        styles[op] = {
            "color": cmap(idx % 10),
            "linewidth": 3.0 if is_ref else 1.5,
            "zorder": 10 if is_ref else 2,
            "label": OP_LABELS.get(op, op),
        }
    return styles


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_run_data(runs: dict, fourier_data: dict) -> dict:
    """Convert loaded data to numpy arrays and compute derived quantities."""
    processed = {}
    for op, metrics in runs.items():
        h = metrics["history"]
        entry = {
            "eval_epochs": np.array(h["eval_epochs"], dtype=np.float64),
            "test_acc": np.array(h["test_acc"], dtype=np.float64),
            "train_loss": np.array(h["train_loss"], dtype=np.float64),
            "fourier_epochs": np.array(h["fourier_epochs"], dtype=np.float64),
            "gini": np.array(h["gini"], dtype=np.float64),
        }
        if op in fourier_data:
            fn = fourier_data[op]["frequency_norms"]
            entry["key_energy_frac"] = compute_key_energy_fraction(fn)
            entry["key_energy_epochs"] = np.array(
                fourier_data[op]["fourier_epochs"], dtype=np.float64
            )
        processed[op] = entry
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

def create_op_sweep_animation(
    run_data: dict,
    styles: dict,
    common_epochs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 10),
    dpi: int = 100,
) -> FuncAnimation:
    """Build and optionally save the 2x2 operation sweep animation."""

    sorted_ops = [op for op in SWEEP_OPERATIONS if op in run_data]
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

    # Create line artists (sorted so reference op drawn last = on top)
    draw_order = sorted(sorted_ops, key=lambda o: styles[o]["zorder"])

    acc_lines = {}
    loss_lines = {}
    gini_lines = {}
    energy_lines = {}
    for op in draw_order:
        s = styles[op]
        kw = dict(color=s["color"], linewidth=s["linewidth"], zorder=s["zorder"], label=s["label"])
        (acc_lines[op],) = ax_acc.plot([], [], **kw)
        (loss_lines[op],) = ax_loss.plot([], [], **kw)
        (gini_lines[op],) = ax_gini.plot([], [], **kw)
        (energy_lines[op],) = ax_energy.plot([], [], **kw)

    # Vertical epoch markers
    marker_kw = dict(color="gray", linestyle="--", alpha=0.6, linewidth=1)
    markers = [ax.axvline(0, **marker_kw) for ax in [ax_acc, ax_loss, ax_gini, ax_energy]]

    # Legends
    for ax in [ax_acc, ax_loss, ax_gini, ax_energy]:
        ax.legend(fontsize=7, ncol=2, loc="best")

    suptitle = fig.suptitle("Operation Sweep \u2014 Epoch 0", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame_idx):
        current_epoch = common_epochs[frame_idx]
        suptitle.set_text(f"Operation Sweep \u2014 Epoch {int(current_epoch)}")

        for op in sorted_ops:
            rd = run_data[op]

            # Panels A & B: mask on eval_epochs
            mask_eval = rd["eval_epochs"] <= current_epoch
            if mask_eval.any():
                ep = rd["eval_epochs"][mask_eval]
                acc_lines[op].set_data(ep, rd["test_acc"][mask_eval])
                loss_lines[op].set_data(ep, rd["train_loss"][mask_eval])
            else:
                acc_lines[op].set_data([], [])
                loss_lines[op].set_data([], [])

            # Panel C: mask on fourier_epochs
            mask_fourier = rd["fourier_epochs"] <= current_epoch
            if mask_fourier.any():
                gini_lines[op].set_data(
                    rd["fourier_epochs"][mask_fourier],
                    rd["gini"][mask_fourier],
                )
            else:
                gini_lines[op].set_data([], [])

            # Panel D: key energy fraction
            if "key_energy_frac" in rd:
                mask_ke = rd["key_energy_epochs"] <= current_epoch
                if mask_ke.any():
                    energy_lines[op].set_data(
                        rd["key_energy_epochs"][mask_ke],
                        rd["key_energy_frac"][mask_ke],
                    )
                else:
                    energy_lines[op].set_data([], [])

        # Move epoch markers
        for m in markers:
            m.set_xdata([current_epoch, current_epoch])

        # Auto-scale loss y-axis based on visible data
        all_loss = []
        for op in sorted_ops:
            rd = run_data[op]
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

        # Also copy to docs/figures
        docs_fig = Path(__file__).resolve().parent.parent / "docs" / "figures" / "op_sweep_animation.gif"
        docs_fig.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(str(out), str(docs_fig))
        logger.info("Copied to: %s", docs_fig)

    plt.close(fig)
    return anim


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate operation sweep comparison (2x2 panels)"
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
        "--operations", type=str, nargs="+", default=SWEEP_OPERATIONS,
        help="Operations to include",
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
        "--reference-op", type=str, default="addition",
        help="Operation to highlight (thicker line)",
    )
    parser.add_argument("--dpi", type=int, default=100, help="Output DPI")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    results_root = args.results_dir
    operations = args.operations

    logger.info("Loading runs for operations=%s from %s", operations, results_root)
    runs = load_all_runs(results_root, operations)
    fourier_data = load_fourier_data(results_root, list(runs.keys()))

    if not runs:
        logger.error("No runs found. Check --results-dir and --operations.")
        return

    available_ops = [op for op in SWEEP_OPERATIONS if op in runs]
    logger.info("Available runs: %s", available_ops)

    styles = get_op_style(available_ops, reference_op=args.reference_op)
    run_data = preprocess_run_data(runs, fourier_data)
    common_epochs = build_common_timeline(run_data, frame_step=args.frame_step)
    logger.info("Timeline: %d frames (step=%d, max_epoch=%d)", len(common_epochs), args.frame_step, int(common_epochs[-1]))

    create_op_sweep_animation(
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
