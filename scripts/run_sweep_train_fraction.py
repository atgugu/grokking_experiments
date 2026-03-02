#!/usr/bin/env python
"""Run train fraction sweep: train models for each fraction, generate figures, print summary."""

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


SWEEP_TF_VALUES = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
SKIP_TRAINING = {0.3}  # Already trained as default WD=1.0 run


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


def print_summary(results_root, tf_values, p, logger):
    """Load metrics from all runs and print a comparison table."""
    print("\n" + "=" * 100)
    print("TRAIN FRACTION SWEEP — SUMMARY")
    print("=" * 100)
    header = (f"{'Frac':>6} | {'# Train':>7} | {'# Test':>7} | {'Grok Epoch':>10} | "
              f"{'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | Key Freqs")
    print(header)
    print("-" * 100)

    total_pairs = p * p

    for tf in tf_values:
        # Build expected run_id
        n_train = int(total_pairs * tf)
        n_test = total_pairs - n_train
        if tf == 0.3:
            rid = f"p{p}_d128_h4_mlp512_L1_wd1.0_s42"
        else:
            rid = f"p{p}_d128_h4_mlp512_L1_wd1.0_s42_tf{tf}"
        run_dir = results_root / rid
        metrics_path = run_dir / "metrics.json"

        if not metrics_path.exists():
            print(f"{tf:>6} | {n_train:>7} | {n_test:>7} | {'MISSING':>10} | "
                  f"{'—':>9} | {'—':>8} | {'—':>6} | —")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"

        train_acc = metrics.get("final_train_acc", 0)
        test_acc = metrics.get("final_test_acc", 0)
        gini = metrics.get("final_gini", 0)
        key_freqs = metrics.get("final_key_frequencies", [])

        print(f"{tf:>6} | {n_train:>7} | {n_test:>7} | {grok_str:>10} | "
              f"{train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {key_freqs}")

    print("=" * 100)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train fraction sweep for grokking experiments")
    parser.add_argument("--config", type=str, default="configs/sweep_train_fraction.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary of existing runs")
    parser.add_argument("--tf-values", type=float, nargs="+", default=None,
                        help="Override train fraction values to sweep (default: 0.05 0.1 0.2 0.3 0.4 0.5 0.7)")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    output_root = Path(args.output_dir)
    tf_values = args.tf_values if args.tf_values else SWEEP_TF_VALUES

    base_config = load_config(args.config)
    p = base_config["p"]

    if args.summary_only:
        print_summary(output_root, tf_values, p, logger)
        return

    logger.info(f"Base config: p={p}, d_model={base_config['d_model']}, "
                f"wd={base_config.get('weight_decay', 1.0)}")
    logger.info(f"Sweep values: {tf_values}")
    logger.info(f"Device: {device}")

    completed = []
    skipped = []
    failed = []

    for tf in tf_values:
        config = dict(base_config)
        config["train_fraction"] = tf
        rid = run_id(config)
        run_dir = output_root / rid

        n_train = int(p * p * tf)
        n_test = p * p - n_train

        print(f"\n{'='*60}")
        logger.info(f"Train fraction = {tf} ({n_train} train / {n_test} test) — run_id: {rid}")

        # Check if already exists
        if (run_dir / "metrics.json").exists():
            logger.info(f"SKIP: {run_dir} already exists")
            skipped.append(tf)
            continue

        # Skip training for values we already have (but generate figures if missing)
        if tf in SKIP_TRAINING:
            logger.info(f"SKIP training: tf={tf} is in SKIP_TRAINING set (reuse existing wd=1.0 run)")
            skipped.append(tf)
            continue

        # Train
        logger.info(f"Training tf={tf} for {config['max_epochs']} epochs...")
        t0 = time.time()
        try:
            saved_dir = train_single_run(config, output_root, device, logger)
            elapsed = time.time() - t0
            logger.info(f"Training complete in {elapsed:.0f}s")
            completed.append(tf)

            # Generate figures
            if not args.skip_figures:
                generate_figures_for_run(saved_dir, logger)
        except Exception as e:
            logger.error(f"FAILED tf={tf}: {e}")
            failed.append(tf)
            import traceback
            traceback.print_exc()

    # Final summary
    print(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(completed)} trained, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed runs: {failed}")

    print_summary(output_root, tf_values, p, logger)


if __name__ == "__main__":
    main()
