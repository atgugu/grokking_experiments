#!/usr/bin/env python3
"""Animated GIF for the attention head count sweep.

3-panel layout per n_heads value:
  Panel 1 (left):   Test accuracy over time for all n_heads
  Panel 2 (middle): Gini coefficient (Fourier sparsity) for all n_heads
  Panel 3 (right):  Frequency spectrum at current epoch (last available snapshot)

A vertical dashed line sweeps through epochs.

Usage:
    python scripts/animate_heads.py
    python scripts/animate_heads.py --frame-step 200 --fps 12
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

SWEEP_HEADS = [1, 2, 4, 8, 16]

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUTPUT = "results/heads_comparison/heads_animation.gif"


def _run_id_for(n_heads: int) -> str:
    return f"p113_d128_h{n_heads}_mlp512_L1_wd1.0_s42"


def load_all_runs(results_root: Path, heads: list) -> dict:
    """Load metrics.json for all n_heads values. Returns {n_heads: metrics_dict}."""
    runs = {}
    for n_heads in heads:
        rid = _run_id_for(n_heads)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            logger.warning("Missing: %s", metrics_path)
            continue
        with open(metrics_path) as f:
            data = json.load(f)

        # Try to load fourier snapshots for richer spectrum evolution
        snap_path = results_root / rid / "fourier_snapshots.npz"
        if snap_path.exists():
            data["_fourier_snapshots"] = dict(np.load(snap_path, allow_pickle=True))
            logger.info("Loaded n_heads=%d: %s (+ fourier_snapshots)", n_heads, rid)
        else:
            logger.info("Loaded n_heads=%d: %s", n_heads, rid)

        runs[n_heads] = data
    return runs


def preprocess_runs(runs: dict) -> dict:
    """Convert history arrays to numpy. Returns {n_heads: entry}."""
    processed = {}
    for n_heads, metrics in runs.items():
        h = metrics["history"]
        entry = {
            "eval_epochs": np.array(h["eval_epochs"], dtype=np.float64),
            "test_acc": np.array(h["test_acc"], dtype=np.float64),
            "fourier_epochs": np.array(h["fourier_epochs"], dtype=np.float64),
            "gini": np.array(h["gini"], dtype=np.float64),
        }

        # Load per-snapshot frequency norms for spectrum animation
        snaps = metrics.get("_fourier_snapshots")
        if snaps is not None and "frequency_norms" in snaps:
            fn = np.array(snaps["frequency_norms"])
            if fn.ndim == 2:
                entry["freq_norms_history"] = fn  # (n_snapshots, p)
                # fourier_epochs key in snapshots aligns with the frequency_norms rows
                if "fourier_epochs" in snaps:
                    entry["snap_epochs"] = np.array(snaps["fourier_epochs"], dtype=np.float64)
                else:
                    entry["snap_epochs"] = entry["fourier_epochs"][:len(fn)]

        processed[n_heads] = entry
    return processed


def build_common_timeline(run_data: dict, frame_step: int = 200) -> np.ndarray:
    max_epoch = max(rd["eval_epochs"][-1] for rd in run_data.values())
    return np.arange(0, max_epoch + 1, frame_step)


def create_heads_animation(
    run_data: dict,
    heads: list,
    common_epochs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
    figsize: tuple[float, float] = (18, 6),
    dpi: int = 100,
) -> FuncAnimation:
    """Build and optionally save the head count sweep animation."""
    available = [h for h in heads if h in run_data]
    n_heads_vals = len(available)
    n_frames = len(common_epochs)
    p = 113

    cmap = cm.get_cmap("viridis")
    head_colors = {h: cmap(i / max(n_heads_vals - 1, 1)) for i, h in enumerate(available)}

    fig, (ax_acc, ax_gini, ax_spec) = plt.subplots(1, 3, figsize=figsize)

    max_epoch = common_epochs[-1]
    for ax in (ax_acc, ax_gini):
        ax.set_xlim(0, max_epoch)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("Epoch", fontsize=10)

    ax_acc.set_title("Test Accuracy by n_heads", fontsize=11, fontweight="bold")
    ax_acc.set_ylabel("Test Accuracy", fontsize=10)
    ax_acc.axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax_gini.set_title("Fourier Sparsity (Gini) by n_heads", fontsize=11, fontweight="bold")
    ax_gini.set_ylabel("Gini Coefficient", fontsize=10)

    ax_spec.set_title("Frequency Spectrum (latest snapshot)", fontsize=11, fontweight="bold")
    ax_spec.set_xlabel("Frequency k", fontsize=10)
    ax_spec.set_ylabel("Normalized energy", fontsize=10)
    ax_spec.set_xlim(0, p)
    ax_spec.set_ylim(0, 1.1)
    ax_spec.grid(True, axis="y", alpha=0.2)

    # Create line artists for acc and gini
    acc_lines = {}
    gini_lines = {}
    for h in available:
        color = head_colors[h]
        lw = 2.0 if h == 4 else 1.5  # highlight baseline
        d_head = 128 // h
        label = f"h={h} (d_h={d_head})"
        (acc_lines[h],) = ax_acc.plot([], [], color=color, linewidth=lw, label=label)
        (gini_lines[h],) = ax_gini.plot([], [], color=color, linewidth=lw, label=label)

    # Spectrum bars (one set of bars per n_heads, stacked offset or overlaid)
    # We overlay all spectra with transparency; use bar containers per n_heads
    freq_bar_collections = {}
    bar_width = 1.0 / n_heads_vals * 0.8
    for i, h in enumerate(available):
        color = head_colors[h]
        offset = (i - n_heads_vals / 2) * bar_width + bar_width / 2
        bars = ax_spec.bar(
            np.arange(p) + offset, np.zeros(p),
            width=bar_width, color=color, alpha=0.6,
            label=f"h={h}"
        )
        freq_bar_collections[h] = bars

    # Vertical epoch markers
    marker_kw = dict(color="gray", linestyle="--", alpha=0.6, linewidth=1)
    acc_marker = ax_acc.axvline(0, **marker_kw)
    gini_marker = ax_gini.axvline(0, **marker_kw)

    ax_acc.legend(fontsize=7, loc="lower right", ncol=1, framealpha=0.9)
    ax_gini.legend(fontsize=7, loc="lower right", ncol=1, framealpha=0.9)
    ax_spec.legend(fontsize=7, loc="upper right", framealpha=0.9)

    suptitle = fig.suptitle("Attention Head Sweep — Epoch 0", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    def update(frame_idx):
        current_epoch = common_epochs[frame_idx]
        suptitle.set_text(f"Attention Head Sweep — Epoch {int(current_epoch)}")

        max_spec = 0.0
        for h in available:
            rd = run_data[h]

            # Test accuracy
            mask_eval = rd["eval_epochs"] <= current_epoch
            if mask_eval.any():
                acc_lines[h].set_data(rd["eval_epochs"][mask_eval], rd["test_acc"][mask_eval])
            else:
                acc_lines[h].set_data([], [])

            # Gini
            mask_fourier = rd["fourier_epochs"] <= current_epoch
            if mask_fourier.any():
                gini_lines[h].set_data(
                    rd["fourier_epochs"][mask_fourier],
                    rd["gini"][mask_fourier],
                )
            else:
                gini_lines[h].set_data([], [])

            # Frequency spectrum: find most recent snapshot up to current_epoch
            if "freq_norms_history" in rd and "snap_epochs" in rd:
                snap_epochs = rd["snap_epochs"]
                mask_snap = snap_epochs <= current_epoch
                if mask_snap.any():
                    snap_idx = np.where(mask_snap)[0][-1]
                    fn = rd["freq_norms_history"][snap_idx].copy()
                    fn[0] = 0.0
                    fn_norm = fn / fn.max() if fn.max() > 0 else fn
                    max_spec = max(max_spec, fn_norm.max())
                    for bar, h_val in zip(freq_bar_collections[h], fn_norm):
                        bar.set_height(h_val)
                else:
                    for bar in freq_bar_collections[h]:
                        bar.set_height(0)
            else:
                for bar in freq_bar_collections[h]:
                    bar.set_height(0)

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
        description="Animate attention head sweep (test_acc + Gini + spectrum)"
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--frame-step", type=int, default=200)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--figsize", type=float, nargs=2, default=[18, 6])
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--heads", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    heads = args.heads if args.heads else SWEEP_HEADS
    results_root = args.results_dir

    logger.info("Loading runs from %s", results_root)
    runs = load_all_runs(results_root, heads)

    if not runs:
        logger.error("No runs found. Check --results-dir.")
        return

    logger.info("Loaded %d/%d runs", len(runs), len(heads))
    run_data = preprocess_runs(runs)
    common_epochs = build_common_timeline(run_data, frame_step=args.frame_step)
    logger.info(
        "Timeline: %d frames (step=%d, max_epoch=%d)",
        len(common_epochs), args.frame_step, int(common_epochs[-1]),
    )

    create_heads_animation(
        run_data=run_data,
        heads=heads,
        common_epochs=common_epochs,
        fps=args.fps,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
