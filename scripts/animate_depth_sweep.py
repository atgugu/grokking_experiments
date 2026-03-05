#!/usr/bin/env python3
"""Animated GIF comparing depth sweep runs on modular arithmetic.

2x3 panels:
  Row 1 (Test Accuracy): L=1 | L=2 | L=3  (all 5 ops per panel)
  Row 2 (Gini):          L=1 | L=2 | L=3  (all 5 ops per panel)

Usage:
    python scripts/animate_depth_sweep.py
    python scripts/animate_depth_sweep.py --frame-step 200 --fps 15
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
SWEEP_LAYERS = [1, 2, 3]
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUTPUT = "results/depth_sweep_comparison/depth_sweep_animation.gif"

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

def _run_id(op: str, n_layers: int) -> str:
    suffix = _OP_SUFFIXES.get(op)
    base = f"p113_d128_h4_mlp512_L{n_layers}_wd1.0_s42"
    if suffix is not None:
        return f"{base}_{suffix}"
    return base


def load_all_runs(results_root: Path, operations: list[str], layers: list[int]) -> dict:
    """Load metrics.json for each (op, n_layers) pair. Returns {(op, n_layers): metrics}."""
    runs = {}
    for n_layers in layers:
        for op in operations:
            rid = _run_id(op, n_layers)
            metrics_path = results_root / rid / "metrics.json"
            if not metrics_path.exists():
                logger.warning("Missing: %s", metrics_path)
                continue
            with open(metrics_path) as f:
                runs[(op, n_layers)] = json.load(f)
            logger.info("Loaded (%s, L=%d): %s", op, n_layers, rid)
    return runs


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def get_op_styles(operations: list[str]) -> dict:
    """Return {op: {"color": ..., "linewidth": ..., "label": ...}} using tab10."""
    cmap = plt.get_cmap("tab10")
    op_to_idx = {op: i for i, op in enumerate(SWEEP_OPERATIONS)}
    return {
        op: {
            "color": cmap(op_to_idx.get(op, len(op_to_idx)) % 10),
            "linewidth": 1.8,
            "label": OP_LABELS.get(op, op),
        }
        for op in operations
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_runs(runs: dict) -> dict:
    """Convert to numpy arrays. Returns {(op, n_layers): entry}."""
    processed = {}
    for (op, n_layers), metrics in runs.items():
        h = metrics["history"]
        processed[(op, n_layers)] = {
            "eval_epochs": np.array(h["eval_epochs"], dtype=np.float64),
            "test_acc": np.array(h["test_acc"], dtype=np.float64),
            "fourier_epochs": np.array(h["fourier_epochs"], dtype=np.float64),
            "gini": np.array(h["gini"], dtype=np.float64),
        }
    return processed


def build_common_timeline(run_data: dict, frame_step: int = 100) -> np.ndarray:
    max_epoch = max(rd["eval_epochs"][-1] for rd in run_data.values())
    return np.arange(0, max_epoch + 1, frame_step)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def create_depth_sweep_animation(
    run_data: dict,
    styles: dict,
    operations: list[str],
    layers: list[int],
    common_epochs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
    figsize: tuple[float, float] = (18, 10),
    dpi: int = 100,
) -> FuncAnimation:
    """Build and optionally save the 2x3 depth sweep animation."""

    n_frames = len(common_epochs)
    n_cols = len(layers)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    # axes[0, j] = test accuracy for layer j
    # axes[1, j] = gini for layer j

    for j, n_layers in enumerate(layers):
        axes[0, j].set_title(f"Test Accuracy  (L={n_layers})", fontsize=11, fontweight="bold")
        axes[0, j].set_xlabel("Epoch", fontsize=9)
        axes[0, j].set_xlim(0, common_epochs[-1])
        axes[0, j].set_ylim(-0.05, 1.05)
        axes[0, j].axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        axes[0, j].grid(True, alpha=0.3)

        axes[1, j].set_title(f"Gini (Fourier Sparsity)  (L={n_layers})", fontsize=11, fontweight="bold")
        axes[1, j].set_xlabel("Epoch", fontsize=9)
        axes[1, j].set_xlim(0, common_epochs[-1])
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Test Accuracy", fontsize=9)
    axes[1, 0].set_ylabel("Gini Coefficient", fontsize=9)

    # Create line artists: acc_lines[(op, n_layers)], gini_lines[(op, n_layers)]
    acc_lines = {}
    gini_lines = {}
    for j, n_layers in enumerate(layers):
        for op in operations:
            s = styles[op]
            kw = dict(color=s["color"], linewidth=s["linewidth"], label=s["label"])
            (acc_lines[(op, n_layers)],) = axes[0, j].plot([], [], **kw)
            (gini_lines[(op, n_layers)],) = axes[1, j].plot([], [], **kw)

    # Vertical epoch markers
    all_axes = [axes[r, c] for r in range(2) for c in range(n_cols)]
    marker_kw = dict(color="gray", linestyle="--", alpha=0.6, linewidth=1)
    markers = [ax.axvline(0, **marker_kw) for ax in all_axes]

    # Legends (only first row to avoid duplication)
    for j in range(n_cols):
        axes[0, j].legend(fontsize=7, ncol=1, loc="lower right")

    suptitle = fig.suptitle("Depth Sweep \u2014 Epoch 0", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame_idx):
        current_epoch = common_epochs[frame_idx]
        suptitle.set_text(f"Depth Sweep \u2014 Epoch {int(current_epoch)}")

        for j, n_layers in enumerate(layers):
            for op in operations:
                key = (op, n_layers)
                if key not in run_data:
                    continue
                rd = run_data[key]

                mask_eval = rd["eval_epochs"] <= current_epoch
                if mask_eval.any():
                    ep = rd["eval_epochs"][mask_eval]
                    acc_lines[key].set_data(ep, rd["test_acc"][mask_eval])
                else:
                    acc_lines[key].set_data([], [])

                mask_fourier = rd["fourier_epochs"] <= current_epoch
                if mask_fourier.any():
                    gini_lines[key].set_data(
                        rd["fourier_epochs"][mask_fourier],
                        rd["gini"][mask_fourier],
                    )
                else:
                    gini_lines[key].set_data([], [])

        for m in markers:
            m.set_xdata([current_epoch, current_epoch])

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

        docs_fig = Path(__file__).resolve().parent.parent / "docs" / "figures" / "depth_sweep_animation.gif"
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
        description="Animate depth sweep comparison (2x3 panels: test_acc + Gini, per layer)"
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
        "--layers", type=int, nargs="+", default=SWEEP_LAYERS,
        help="n_layers values to include",
    )
    parser.add_argument(
        "--frame-step", type=int, default=100,
        help="Epoch step between frames",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=[18, 10],
        help="Figure size (width height)",
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
    layers = args.layers

    logger.info("Loading runs for operations=%s layers=%s from %s", operations, layers, results_root)
    runs = load_all_runs(results_root, operations, layers)

    if not runs:
        logger.error("No runs found. Check --results-dir.")
        return

    logger.info("Loaded %d/%d runs", len(runs), len(operations) * len(layers))

    styles = get_op_styles(operations)
    run_data = preprocess_runs(runs)
    common_epochs = build_common_timeline(run_data, frame_step=args.frame_step)
    logger.info(
        "Timeline: %d frames (step=%d, max_epoch=%d)",
        len(common_epochs), args.frame_step, int(common_epochs[-1])
    )

    create_depth_sweep_animation(
        run_data=run_data,
        styles=styles,
        operations=operations,
        layers=layers,
        common_epochs=common_epochs,
        fps=args.fps,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
