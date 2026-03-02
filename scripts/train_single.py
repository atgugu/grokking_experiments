#!/usr/bin/env python
"""Train a single grokking model with given config."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.modular_arithmetic import ModularArithmeticEnvironment
from src.models.transformer import GrokkingTransformer
from src.training.trainer import Trainer
from src.training.checkpointing import save_run_result
from src.utils import set_seed, load_config, get_device, setup_logging, run_id


def main():
    parser = argparse.ArgumentParser(description="Train a single grokking model")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--operation", type=str, default=None)
    parser.add_argument("--train-fraction", type=float, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--n-heads", type=int, default=None)
    parser.add_argument("--d-mlp", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "gelu"])
    parser.add_argument("--use-layernorm", type=lambda v: v.lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--tie-embeddings", type=lambda v: v.lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--mlp-bias", type=lambda v: v.lower() in ("true", "1", "yes"), default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None, choices=["adamw", "adam"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--fourier-interval", type=int, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config(args.config)

    # CLI overrides
    override_keys = [
        "p", "operation", "train_fraction", "d_model", "n_heads", "d_mlp",
        "n_layers", "activation", "use_layernorm", "tie_embeddings", "mlp_bias",
        "max_epochs", "optimizer", "lr", "weight_decay", "seed",
        "eval_interval", "fourier_interval", "checkpoint_interval",
    ]
    for key in override_keys:
        val = getattr(args, key.replace("-", "_"), None)
        if val is not None:
            config[key] = val

    set_seed(config["seed"])
    device = get_device()
    logger.info(f"Device: {device}")
    logger.info(f"Config: p={config['p']}, d_model={config['d_model']}, "
                f"n_heads={config['n_heads']}, d_mlp={config['d_mlp']}, "
                f"L={config['n_layers']}, wd={config['weight_decay']}")

    env = ModularArithmeticEnvironment(
        p=config["p"],
        operation=config.get("operation", "addition"),
        train_fraction=config.get("train_fraction", 0.3),
        seed=config["seed"],
    )
    logger.info(f"Train: {len(env.train_indices)}, Test: {len(env.test_indices)}")

    model = GrokkingTransformer(
        p=config["p"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_mlp=config["d_mlp"],
        n_layers=config["n_layers"],
        activation=config.get("activation", "relu"),
        use_layernorm=config.get("use_layernorm", False),
        tie_embeddings=config.get("tie_embeddings", False),
        mlp_bias=config.get("mlp_bias", True),
    )
    logger.info(f"Parameters: {model.count_parameters():,}")

    rid = run_id(config)
    output_dir = Path(args.output_dir) / rid
    trainer = Trainer(model, env, config, device, output_dir=output_dir)
    result = trainer.train(quiet=args.quiet)

    logger.info(f"Final train acc: {result['final_train_acc']:.4f}")
    logger.info(f"Final test acc: {result['final_test_acc']:.4f}")
    logger.info(f"Final Gini: {result['final_gini']:.4f}")
    logger.info(f"Key frequencies: {result['final_key_frequencies']}")

    saved_dir = save_run_result(result, config, Path(args.output_dir), model=model)
    logger.info(f"Saved to: {saved_dir}")


if __name__ == "__main__":
    main()
