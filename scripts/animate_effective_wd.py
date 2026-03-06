#!/usr/bin/env python3
"""Animated GIF for effective weight-decay (wd × lr) unification experiment.

2×2 panels:
  Row 1: Test accuracy for eff_wd=1e-3 group | Test accuracy for eff_wd=3e-3 group
  Row 2: Gini coefficient for eff_wd=1e-3 group | Gini coefficient for eff_wd=3e-3 group

Each panel overlays all (wd, lr) decompositions with the same eff_wd value.

Usage:
    python scripts/animate_effective_wd.py
    python scripts/animate_effective_wd.py --frame-step 200 --fps 15
"""

import argparse
import json
import logging
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUTPUT = "results/effective_wd_comparison/effective_wd_animation.gif"

# The two eff_wd groups shown in the animation (left column = eff=1e-3, right = eff=3e-3)
EFF_WD_GROUPS = {
    1e-3: [(1.0, 1e-3), (2.0, 5e-4), (0.5, 2e-3)],
    3e-3: [(1.0, 3e-3), (3.0, 1e-3)],
}

# Colors per (wd, lr) pair — consistent across panels
PAIR_COLORS = {
    (1.0, 1e-3): "#2ca02c",   # green — baseline
    (2.0, 5e-4): "#ff7f0e",   # orange — new
    (0.5, 2e-3): "#9467bd",   # purple — new
    (1.0, 3e-3): "#1f77b4",   # blue — LR sweep
    (3.0, 1e-3): "#d62728",   # red — new
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _run_id_for(wd: float, lr: float) -> str:
    base = f"p113_d128_h4_mlp512_L1_wd{wd}_s42"
    if abs(lr - 1e-3) > 1e-9:
        return f"{base}_lr{lr}"
    return base


def load_all_runs(results_root: Path, groups: dict) -> dict:
    """Load metrics.json for all (wd, lr) pairs in groups.

    Returns {(wd, lr): metrics_dict}
    """
    runs = {}
    for eff_wd, pairs in groups.items():
        for wd, lr in pairs:
            rid = _run_id_for(wd, lr)
            metrics_path = results_root / rid / "metrics.json"
            if not metrics_path.exists():
                logger.warning("Missing: %s", metrics_path)
                continue
            with open(metrics_path) as f:
                runs[(wd, lr)] = json.load(f)
            logger.info("Loaded (wd=%s, lr=%.0e, eff=%.0e): %s", wd, lr, wd * lr, rid)
    return runs


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess_runs(runs: dict) -> dict:
    """Convert to numpy arrays. Returns {(wd, lr): entry}."""
    processed = {}
    for (wd, lr), metrics in runs.items():
        h = metrics["history"]
        processed[(wd, lr)] = {
            "eval_epochs": np.array(h["eval_epochs"], dtype=np.float64),
            "test_acc": np.array(h["test_acc"], dtype=np.float64),
            "fourier_epochs": np.array(h["fourier_epochs"], dtype=np.float64),
            "gini": np.array(h["gini"], dtype=np.float64),
        }
    return processed


def build_common_timeline(run_data: dict, frame_step: int = 200) -> np.ndarray:
    max_epoch = max(rd["eval_epochs"][-1] for rd in run_data.values())
    return np.arange(0, max_epoch + 1, frame_step)


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def create_effective_wd_animation(
    run_data: dict,
    groups: dict,
    common_epochs: np.ndarray,
    fps: int = 10,
    output_path: str | None = None,
    figsize: tuple[float, float] = (14, 8),
    dpi: int = 100,
) -> FuncAnimation:
    """Build and optionally save the 2×2 effective WD animation."""
    sorted_eff_wds = sorted(groups.keys())
    n_frames = len(common_epochs)
    n_cols = len(sorted_eff_wds)

    fig, axes = plt.subplots(2, n_cols, figsize=figsize)
    # axes[0, j] = test accuracy for eff_wd group j
    # axes[1, j] = gini for eff_wd group j

    for j, eff_wd in enumerate(sorted_eff_wds):
        axes[0, j].set_title(
            f"Test Accuracy  (eff_wd = {eff_wd:.0e})", fontsize=11, fontweight="bold"
        )
        axes[0, j].set_xlabel("Epoch", fontsize=9)
        axes[0, j].set_xlim(0, common_epochs[-1])
        axes[0, j].set_ylim(-0.05, 1.05)
        axes[0, j].axhline(0.95, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        axes[0, j].grid(True, alpha=0.3)

        axes[1, j].set_title(
            f"Gini (Fourier Sparsity)  (eff_wd = {eff_wd:.0e})", fontsize=11, fontweight="bold"
        )
        axes[1, j].set_xlabel("Epoch", fontsize=9)
        axes[1, j].set_xlim(0, common_epochs[-1])
        axes[1, j].set_ylim(-0.05, 1.05)
        axes[1, j].grid(True, alpha=0.3)

    axes[0, 0].set_ylabel("Test Accuracy", fontsize=9)
    axes[1, 0].set_ylabel("Gini Coefficient", fontsize=9)

    # Create line artists for each (eff_wd, wd, lr) combination
    acc_lines = {}
    gini_lines = {}
    for j, eff_wd in enumerate(sorted_eff_wds):
        for wd, lr in groups[eff_wd]:
            key = (wd, lr)
            color = PAIR_COLORS.get(key, "#777777")
            label = f"wd={wd}, lr={lr:.0e}"
            is_baseline = abs(wd - 1.0) < 1e-9 and abs(lr - 1e-3) < 1e-9
            lw = 2.5 if is_baseline else 1.8
            kw = dict(color=color, linewidth=lw, label=label)
            (acc_lines[key],) = axes[0, j].plot([], [], **kw)
            (gini_lines[key],) = axes[1, j].plot([], [], **kw)

    # Vertical epoch markers
    all_axes = [axes[r, c] for r in range(2) for c in range(n_cols)]
    marker_kw = dict(color="gray", linestyle="--", alpha=0.6, linewidth=1)
    markers = [ax.axvline(0, **marker_kw) for ax in all_axes]

    # Legends
    for j in range(n_cols):
        axes[0, j].legend(fontsize=8, loc="lower right", framealpha=0.9)
        axes[1, j].legend(fontsize=8, loc="lower right", framealpha=0.9)

    suptitle = fig.suptitle(
        "Effective WD Unification — Epoch 0", fontsize=14, fontweight="bold", y=1.01
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    def update(frame_idx):
        current_epoch = common_epochs[frame_idx]
        suptitle.set_text(f"Effective WD Unification — Epoch {int(current_epoch)}")

        for j, eff_wd in enumerate(sorted_eff_wds):
            for wd, lr in groups[eff_wd]:
                key = (wd, lr)
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

        # Downscale and copy to docs/figures/
        _copy_to_docs(out)

    plt.close(fig)
    return anim


def _copy_to_docs(gif_path: Path, target_size: tuple[int, int] = (900, 500)):
    """Downscale GIF to target_size and copy to docs/figures/."""
    try:
        from PIL import Image
        img = Image.open(gif_path)

        frames = []
        durations = []
        try:
            while True:
                frame = img.copy().convert("RGBA").resize(target_size, Image.LANCZOS)
                frames.append(frame)
                durations.append(img.info.get("duration", 100))
                img.seek(img.tell() + 1)
        except EOFError:
            pass

        docs_fig = Path(__file__).resolve().parent.parent / "docs" / "figures" / "effective_wd_animation.gif"
        docs_fig.parent.mkdir(parents=True, exist_ok=True)

        if frames:
            frames[0].save(
                str(docs_fig),
                save_all=True,
                append_images=frames[1:],
                loop=0,
                duration=durations,
                optimize=True,
            )
            logger.info("Downscaled and saved to: %s", docs_fig)
        else:
            shutil.copy2(str(gif_path), str(docs_fig))
            logger.info("Copied to: %s", docs_fig)

    except ImportError:
        # Pillow not available — plain copy
        docs_fig = Path(__file__).resolve().parent.parent / "docs" / "figures" / "effective_wd_animation.gif"
        docs_fig.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(gif_path), str(docs_fig))
        logger.info("Copied to: %s", docs_fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Animate effective WD unification (2x2 panels: test_acc + Gini per eff_wd group)"
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
        "--frame-step", type=int, default=200,
        help="Epoch step between frames",
    )
    parser.add_argument("--fps", type=int, default=10, help="Frames per second")
    parser.add_argument(
        "--figsize", type=float, nargs=2, default=[14, 8],
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

    logger.info("Loading runs from %s", results_root)
    runs = load_all_runs(results_root, EFF_WD_GROUPS)

    if not runs:
        logger.error("No runs found. Check --results-dir.")
        return

    logger.info("Loaded %d runs", len(runs))

    run_data = preprocess_runs(runs)
    common_epochs = build_common_timeline(run_data, frame_step=args.frame_step)
    logger.info(
        "Timeline: %d frames (step=%d, max_epoch=%d)",
        len(common_epochs), args.frame_step, int(common_epochs[-1]),
    )

    create_effective_wd_animation(
        run_data=run_data,
        groups=EFF_WD_GROUPS,
        common_epochs=common_epochs,
        fps=args.fps,
        output_path=args.output,
        figsize=tuple(args.figsize),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
