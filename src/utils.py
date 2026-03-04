"""Shared utilities: seeding, config loading, logging, device selection."""

import hashlib
import logging
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict:
    """Load YAML config, merging with default.yaml for missing keys."""
    path = Path(path)
    default_path = path.parent / "default.yaml"

    config = {}
    if default_path.exists() and path.name != "default.yaml":
        with open(default_path) as f:
            config = yaml.safe_load(f) or {}

    with open(path) as f:
        override = yaml.safe_load(f) or {}
    config.update(override)
    return config


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return project logger."""
    logger = logging.getLogger("grokking")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper()))
    return logger


def run_id(config: dict) -> str:
    """Generate a unique run ID from config parameters."""
    parts = [
        f"p{config['p']}",
        f"d{config['d_model']}",
        f"h{config['n_heads']}",
        f"mlp{config['d_mlp']}",
        f"L{config['n_layers']}",
        f"wd{config['weight_decay']}",
        f"s{config['seed']}",
    ]
    if config.get("train_fraction", 0.3) != 0.3:
        parts.append(f"tf{config['train_fraction']}")
    if config.get("lr", 1e-3) != 1e-3:
        parts.append(f"lr{config['lr']}")
    if not config.get("mlp_bias", True):
        parts.append("nobias")
    return "_".join(parts)
