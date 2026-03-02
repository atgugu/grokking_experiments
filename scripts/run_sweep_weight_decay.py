#!/usr/bin/env python
"""Run weight decay sweep: train models for each wd value, generate figures, print summary."""

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.modular_arithmetic import ModularArithmeticEnvironment
from src.models.transformer import GrokkingTransformer
from src.training.trainer import Trainer
from src.training.checkpointing import save_run_result
from src.utils import set_seed, load_config, get_device, setup_logging, run_id


SWEEP_WD_VALUES = [0.01, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
SKIP_TRAINING = {1.0}  # Already trained — only include in summary


def train_single_run(config, output_root, device, logger):
    """Train one model and save results. Returns the run directory."""
    set_seed(config["seed"])

    env = ModularArithmeticEnvironment(
        p=config["p"],
        operation=config.get("operation", "addition"),
        train_fraction=config.get("train_fraction", 0.3),
        seed=config["seed"],
    )

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
    output_dir = output_root / rid
    trainer = Trainer(model, env, config, device, output_dir=output_dir)
    result = trainer.train(quiet=False)

    logger.info(f"Final train acc: {result['final_train_acc']:.4f}")
    logger.info(f"Final test acc:  {result['final_test_acc']:.4f}")
    logger.info(f"Final Gini:      {result['final_gini']:.4f}")

    saved_dir = save_run_result(result, config, output_root, model=model)
    return saved_dir


def generate_figures_for_run(run_dir, logger):
    """Call generate_figures.py as a subprocess."""
    script = Path(__file__).parent / "generate_figures.py"
    cmd = [sys.executable, str(script), "--run-dir", str(run_dir)]
    logger.info(f"Generating figures: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Figure generation failed for {run_dir.name}:\n{result.stderr[-500:]}")
    else:
        logger.info(f"Figures saved to {run_dir / 'figures'}")


def find_grokking_epoch(history, threshold=0.95):
    """Find the first epoch where test accuracy exceeds threshold."""
    test_accs = history.get("test_acc", [])
    eval_epochs = history.get("eval_epochs", [])
    for epoch, acc in zip(eval_epochs, test_accs):
        if acc >= threshold:
            return epoch
    return None


def print_summary(results_root, wd_values, logger):
    """Load metrics from all runs and print a comparison table."""
    print("\n" + "=" * 90)
    print("WEIGHT DECAY SWEEP — SUMMARY")
    print("=" * 90)
    header = f"{'WD':>6} | {'Grok Epoch':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | {'W Norm':>8} | Key Freqs"
    print(header)
    print("-" * 90)

    for wd in wd_values:
        # Build expected run_id
        rid = f"p113_d128_h4_mlp512_L1_wd{wd}_s42"
        run_dir = results_root / rid
        metrics_path = run_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"{wd:>6} | {'MISSING':>10} | {'—':>9} | {'—':>8} | {'—':>6} | {'—':>8} | —")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"

        train_acc = metrics.get("final_train_acc", 0)
        test_acc = metrics.get("final_test_acc", 0)
        gini = metrics.get("final_gini", 0)
        w_norm = metrics.get("final_weight_norm", 0)
        key_freqs = metrics.get("final_key_frequencies", [])

        print(f"{wd:>6} | {grok_str:>10} | {train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {w_norm:>8.1f} | {key_freqs}")

    print("=" * 90)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Weight decay sweep for grokking experiments")
    parser.add_argument("--config", type=str, default="configs/sweep_weight_decay.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary of existing runs")
    parser.add_argument("--wd-values", type=float, nargs="+", default=None,
                        help="Override wd values to sweep (default: 0.01 0.1 0.3 0.5 1.0 2.0 5.0)")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    output_root = Path(args.output_dir)
    wd_values = args.wd_values if args.wd_values else SWEEP_WD_VALUES

    if args.summary_only:
        print_summary(output_root, wd_values, logger)
        return

    base_config = load_config(args.config)
    logger.info(f"Base config: p={base_config['p']}, d_model={base_config['d_model']}, "
                f"beta2={base_config.get('beta2', 0.999)}")
    logger.info(f"Sweep values: {wd_values}")
    logger.info(f"Device: {device}")

    completed = []
    skipped = []
    failed = []

    for wd in wd_values:
        config = dict(base_config)
        config["weight_decay"] = wd
        rid = run_id(config)
        run_dir = output_root / rid

        print(f"\n{'='*60}")
        logger.info(f"Weight decay = {wd} — run_id: {rid}")

        # Check if already exists
        if (run_dir / "metrics.json").exists():
            logger.info(f"SKIP: {run_dir} already exists")
            skipped.append(wd)
            continue

        # Skip training for values we already have (but generate figures if missing)
        if wd in SKIP_TRAINING:
            logger.info(f"SKIP training: wd={wd} is in SKIP_TRAINING set")
            skipped.append(wd)
            continue

        # Train
        logger.info(f"Training wd={wd} for {config['max_epochs']} epochs...")
        t0 = time.time()
        try:
            saved_dir = train_single_run(config, output_root, device, logger)
            elapsed = time.time() - t0
            logger.info(f"Training complete in {elapsed:.0f}s")
            completed.append(wd)

            # Generate figures
            if not args.skip_figures:
                generate_figures_for_run(saved_dir, logger)
        except Exception as e:
            logger.error(f"FAILED wd={wd}: {e}")
            failed.append(wd)
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(completed)} trained, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed runs: {failed}")

    print_summary(output_root, wd_values, logger)


if __name__ == "__main__":
    main()
