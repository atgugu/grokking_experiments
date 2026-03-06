#!/usr/bin/env python3
"""Animated GIF for the prime p sweep.

2×1 panels:
  Left:  Test accuracy for all primes
  Right: Gini coefficient (Fourier sparsity) for all primes

Each prime gets its own colored line (viridis colormap, small → large p).
A vertical dashed line sweeps through epochs.

Usage:
    python scripts/animate_primes.py
    python scripts/animate_primes.py --frame-step 200 --fps 12
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from matplotlib.animation import FuncAnimation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

SWEEP_PRIMES = [7, 11, 13, 17, 23, 31, 43, 59, 67, 89, 97, 113]

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUTPUT = "results/primes_comparison/primes_animation.gif"


def _run_id_for(p: int) -> str:
    return f"p{p}_d128_h4_mlp512_L1_wd1.0_s42"


def load_all_runs(results_root: Path, primes: list) -> dict:
    """Load metrics.json for all primes. Returns {p: metrics_dict}."""
    runs = {}
    for p in primes:
        rid = _run_id_for(p)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Missing: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            runs[p] = json.load(f)
        logger.info("Loaded p=%d: %s", p, rid)
    return runs


def preprocess_runs(runs: dict) -> dict:
    """Convert history arrays to numpy. Returns {p: entry}."""
    processed = {}
    for p, metrics in runs.items():
        h = metrics["history"]
        processed[p] = {
            "eval_epochs": np.array(h["eval_epochs"], dtype=np.float64),
            "test_acc": np.array(h["test_acc"], dtype=np.float64),
            "fourier_epochs": np.array(h["fourier_epochs"], dtype=np.float64),
            "gini": np.array(h["gini"], dtype=np.float64),
        }
    return processed


def build_common_timeline(run_data: dict, frame_step: int = 200) -> np.ndarray:
    max_epoch = max(rd["eval_epochs"][-1] for rd in run_data.values())
    return np.arange(0, max_epoch + 1, frame_step)


def create_primes_animation(
    run_data: dict,
    primes: list,
    common_epochs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 6),
    dpi: int = 100,
) -> FuncAnimation:
    """Build and optionally save the prime sweep animation."""
    available = [p for p in primes if p in run_data]
    n_primes = len(available)
    n_frames = len(common_epochs)

    # Assign colors via viridis (small p = dark, large p = bright)
    cmap = cm.get_cmap("viridis")
    prime_colors = {p: cmap(i / max(n_primes - 1, 1)) for i, p in enumerate(available)}

    fig, (ax_acc, ax_gini) = plt.subplots(1, 2, figsize=figsize)

    # Setup axes
    max_epoch = common_epochs[-1]
    for ax in (ax_acc, ax_gini):
        ax.set_xlim(0, max_epoch)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch", fontsize=10)

    ax_acc.set_title("Test Accuracy per Prime", fontsize=12, fontweight="bold")
    ax_acc.set_ylabel("Test Accuracy", fontsize=10)
    ax_acc.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax_gini.set_title("Fourier Sparsity (Gini) per Prime", fontsize=12, fontweight="bold")
    ax_gini.set_ylabel("Gini Coefficient", fontsize=10)

    # Create line artists
    acc_lines = {}
    gini_lines = {}
    for p in available:
        color = prime_colors[p]
        lw = 2.0 if p == 113 else 1.5
        label = f"p={p}"
        (acc_lines[p],) = ax_acc.plot([], [], color=color, linewidth=lw, label=label)
        (gini_lines[p],) = ax_gini.plot([], [], color=color, linewidth=lw, label=label)

    # Vertical epoch markers
    marker_kw = dict(color="gray", linestyle="--", alpha=0.6, linewidth=1)
    acc_marker = ax_acc.axvline(0, **marker_kw)
    gini_marker = ax_gini.axvline(0, **marker_kw)

    # Legends
    ax_acc.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)
    ax_gini.legend(fontsize=7, loc="lower right", ncol=2, framealpha=0.9)

    suptitle = fig.suptitle(
        "Prime p Sweep — Epoch 0", fontsize=14, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame_idx):
        current_epoch = common_epochs[frame_idx]
        suptitle.set_text(f"Prime p Sweep — Epoch {int(current_epoch)}")

        for p in available:
            rd = run_data[p]

            mask_eval = rd["eval_epochs"] <= current_epoch
            if mask_eval.any():
                ep = rd["eval_epochs"][mask_eval]
                acc_lines[p].set_data(ep, rd["test_acc"][mask_eval])
            else:
                acc_lines[p].set_data([], [])

            mask_fourier = rd["fourier_epochs"] <= current_epoch
            if mask_fourier.any():
                gini_lines[p].set_data(
                    rd["fourier_epochs"][mask_fourier],
                    rd["gini"][mask_fourier],
                )
            else:
                gini_lines[p].set_data([], [])

        acc_marker.set_xdata([current_epoch, current_epoch])
        gini_marker.set_xdata([current_epoch, current_epoch])
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


def main():
    parser = argparse.ArgumentParser(
        description="Animate prime p sweep (test_acc + Gini for all primes)"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-step", type=int, default=200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--figsize", type=float, nargs=2, default=[14, 6])
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--primes", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    primes = args.primes if args.primes else SWEEP_PRIMES
    results_root = args.results_dir

    logger.info("Loading runs from %s", results_root)
    runs = load_all_runs(results_root, primes)

    if not runs:
        logger.error("No runs found. Check --results-dir.")
        return

    logger.info("Loaded %d/%d runs", len(runs), len(primes))
    run_data = preprocess_runs(runs)
    common_epochs = build_common_timeline(run_data, frame_step=args.frame_step)
    logger.info(
        "Timeline: %d frames (step=%d, max_epoch=%d)",
        len(common_epochs), args.frame_step, int(common_epochs[-1]),
    )

    create_primes_animation(
        run_data=run_data,
        primes=primes,
        common_epochs=common_epochs,
        fps=args.fps,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
