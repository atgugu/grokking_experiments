"""Save/load utilities for grokking experiment results."""

import json
from pathlib import Path

import numpy as np
import torch

from ..utils import run_id


def save_run_result(result: dict, config: dict, output_dir: Path, model=None):
    """Save a single run's results.

    Saves:
        - config.json: Full configuration
        - metrics.json: Training history and final metrics
        - model.pt: Model weights (optional)
        - fourier_snapshots.npz: Fourier analysis snapshots

    Args:
        result: Training result dict from Trainer.train().
        config: Configuration dict.
        output_dir: Root output directory.
        model: Optional nn.Module to save model weights.
    """
    output_dir = Path(output_dir)
    rid = run_id(config)
    run_dir = output_dir / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(run_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Save metrics (everything except large arrays)
    metrics = {}
    for k, v in result.items():
        if k == "checkpoints":
            continue
        if k == "history":
            history = {}
            for hk, hv in v.items():
                if isinstance(hv, list) and hv:
                    if isinstance(hv[0], (list, np.ndarray)):
                        continue  # Save separately in npz
                    elif isinstance(hv[0], (int, float, np.floating)):
                        history[hk] = [float(x) for x in hv]
                    else:
                        history[hk] = hv
                else:
                    history[hk] = hv
            metrics["history"] = history
        elif isinstance(v, (list, np.ndarray)):
            if isinstance(v, np.ndarray):
                metrics[k] = v.tolist()
            elif v and isinstance(v[0], (int, float, np.integer, np.floating)):
                metrics[k] = [float(x) if isinstance(x, (np.floating, float)) else int(x) for x in v]
            else:
                metrics[k] = v
        elif isinstance(v, (int, float, np.floating, np.integer)):
            metrics[k] = float(v) if isinstance(v, (np.floating, float)) else int(v)
        else:
            metrics[k] = v

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save model weights
    if model is not None:
        save_model(model, run_dir / "model.pt")

    # Save Fourier snapshots as npz
    history = result.get("history", {})
    freq_norms = history.get("frequency_norms_snapshots", [])
    key_freqs = history.get("key_frequencies_snapshots", [])
    fourier_epochs = history.get("fourier_epochs", [])

    if freq_norms:
        snap_data = {
            "frequency_norms": np.array(freq_norms),
            "key_frequencies": np.array(key_freqs),
            "fourier_epochs": np.array(fourier_epochs),
        }
        np.savez_compressed(run_dir / "fourier_snapshots.npz", **snap_data)

    return run_dir


def load_run_result(run_dir: Path) -> dict:
    """Load a run's metrics and config."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        config = json.load(f)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    return {"config": config, "metrics": metrics}


def load_fourier_snapshots(run_dir: Path) -> dict:
    """Load Fourier analysis snapshots."""
    snap_path = Path(run_dir) / "fourier_snapshots.npz"
    if not snap_path.exists():
        return {}
    return dict(np.load(snap_path))


def save_model(model, path: Path):
    """Save model state dict."""
    torch.save(model.state_dict(), path)


def load_model(model, path: Path, device: torch.device):
    """Load model state dict."""
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model


def save_checkpoint(model, optimizer, epoch: int, path: Path):
    """Save full training checkpoint (model + optimizer + epoch)."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path: Path, device: torch.device) -> int:
    """Load full training checkpoint. Returns epoch number."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt["epoch"]
