#!/usr/bin/env python
"""Post-hoc Fourier and neuron analysis on saved checkpoints."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.modular_arithmetic import ModularArithmeticEnvironment
from src.models.transformer import GrokkingTransformer
from src.models.hooks import ActivationCache
from src.analysis.fourier import (
    compute_fourier_component_norms,
    compute_gini_coefficient,
    compute_restricted_logits,
    compute_excluded_logits,
    identify_key_frequencies,
    fourier_embed_analysis,
)
from src.analysis.progress_measures import compute_all_progress_measures
from src.analysis.neuron_analysis import (
    classify_neuron_frequencies,
    compute_neuron_frequency_spectrum,
)
from src.training.checkpointing import load_run_result
from src.utils import get_device, setup_logging


def analyze_checkpoint(model, config, env, device, logger):
    """Run full analysis on a single model state."""
    p = config["p"]
    model.eval()

    # Logit table
    logit_table = model.get_logit_table(device).cpu().numpy()

    # Fourier analysis
    fourier_result = compute_fourier_component_norms(logit_table, p)
    freq_norms = fourier_result["frequency_norms"]
    key_freqs = identify_key_frequencies(freq_norms, n_top=5)
    gini = compute_gini_coefficient(freq_norms)

    # Restricted/excluded loss
    train_data = env.get_train_dataset()
    progress = compute_all_progress_measures(
        model, train_data.inputs.to(device), train_data.target_tensor.to(device),
        p, key_freqs, device,
    )

    # Embedding Fourier analysis
    W_E = model.W_E.weight.detach().cpu().numpy()
    embed_fourier = fourier_embed_analysis(W_E, p)

    # Neuron analysis via activation cache
    full_data = env.get_full_dataset()
    full_inputs = full_data.inputs.to(device)

    with ActivationCache(model) as cache:
        _ = model(full_inputs)
        if cache.neuron_activations:
            # Get neuron activations at = position (position 2)
            neuron_acts = cache.neuron_activations[0][:, 2, :].cpu().numpy()
            # Reshape to (p, p, d_mlp)
            neuron_acts_grid = neuron_acts.reshape(p, p, -1)
            neuron_class = classify_neuron_frequencies(neuron_acts_grid, p)
            neuron_spectrum = compute_neuron_frequency_spectrum(neuron_acts_grid, p)
        else:
            neuron_class = {"dominant_freq": [], "r_squared": [], "clusters": {}}
            neuron_spectrum = np.array([])

    return {
        "key_frequencies": key_freqs.tolist(),
        "gini": gini,
        "frequency_norms": freq_norms.tolist(),
        "restricted_loss": progress["restricted_loss"],
        "excluded_loss": progress["excluded_loss"],
        "weight_norm": progress["weight_norm"],
        "embed_frequency_energy": embed_fourier["frequency_energy"].tolist(),
        "neuron_clusters": {str(k): v for k, v in neuron_class["clusters"].items()},
        "neuron_r_squared_mean": float(np.mean(neuron_class["r_squared"])) if len(neuron_class["r_squared"]) > 0 else 0.0,
        "component_norms": fourier_result["component_norms"].tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze grokking run checkpoints")
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: <run-dir>/analysis.json)")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    run_dir = Path(args.run_dir)

    # Load config
    run_result = load_run_result(run_dir)
    config = run_result["config"]

    # Setup environment and model
    env = ModularArithmeticEnvironment(
        p=config["p"],
        operation=config.get("operation", "addition"),
        train_fraction=config.get("train_fraction", 0.3),
        seed=config["seed"],
    )

    # Find checkpoints
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        logger.error(f"No checkpoints directory found in {run_dir}")
        return

    ckpt_files = sorted(ckpt_dir.glob("epoch_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    logger.info(f"Found {len(ckpt_files)} checkpoints")

    analyses = {}
    for ckpt_path in ckpt_files:
        epoch = int(ckpt_path.stem.split("_")[1])
        logger.info(f"Analyzing epoch {epoch}...")

        model = GrokkingTransformer(
            p=config["p"],
            d_model=config["d_model"],
            n_heads=config["n_heads"],
            d_mlp=config["d_mlp"],
            n_layers=config["n_layers"],
            activation=config.get("activation", "relu"),
            use_layernorm=config.get("use_layernorm", False),
            tie_embeddings=config.get("tie_embeddings", False),
        )
        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
        model.to(device)

        result = analyze_checkpoint(model, config, env, device, logger)
        analyses[str(epoch)] = result

    # Save
    output_path = Path(args.output) if args.output else run_dir / "analysis.json"
    # Convert component_norms to avoid huge JSON; just save summary
    for epoch_key in analyses:
        if "component_norms" in analyses[epoch_key]:
            del analyses[epoch_key]["component_norms"]

    with open(output_path, "w") as f:
        json.dump(analyses, f, indent=2)
    logger.info(f"Analysis saved to {output_path}")


if __name__ == "__main__":
    main()
