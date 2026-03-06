#!/usr/bin/env python
"""Run prime p sweep: train models for each prime modulus.

Tests p ∈ {7, 11, 13, 17, 23, 31, 43, 59, 67, 89, 97, 113} — 12 runs total.
p=113 (baseline) is skipped if metrics.json already exists.

Key scientific question: does grokking scale with p, and is the trigonometric
algorithm universal across all prime moduli?
"""

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


SWEEP_PRIMES = [7, 11, 13, 17, 23, 31, 43, 59, 67, 89, 97, 113]


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


def print_summary(results_root, primes, logger):
    """Load metrics from all runs and print a summary table."""
    print("\n" + "=" * 90)
    print("PRIME p SWEEP — SUMMARY")
    print("=" * 90)
    header = f"{'p':>5} | {'Pairs':>5} | {'Train N':>7} | {'Grok Epoch':>10} | {'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | Key Freqs"
    print(header)
    print("-" * 90)

    for p in primes:
        rid = f"p{p}_d128_h4_mlp512_L1_wd1.0_s42"
        run_dir = results_root / rid
        metrics_path = run_dir / "metrics.json"
        n_pairs = p * p
        n_train = int(n_pairs * 0.3)

        if not metrics_path.exists():
            print(f"{p:>5} | {n_pairs:>5} | {n_train:>7} | {'MISSING':>10} | {'—':>9} | {'—':>8} | {'—':>6} | —")
            continue

        with open(metrics_path) as f:
            metrics = json.load(f)

        history = metrics.get("history", {})
        test_accs = history.get("test_acc", [])
        eval_epochs = history.get("eval_epochs", [])
        grok_epoch = None
        for epoch, acc in zip(eval_epochs, test_accs):
            if acc >= 0.95:
                grok_epoch = epoch
                break

        grok_str = str(grok_epoch) if grok_epoch is not None else "Never"
        train_acc = metrics.get("final_train_acc", 0)
        test_acc = metrics.get("final_test_acc", 0)
        gini = metrics.get("final_gini", 0)
        key_freqs = metrics.get("final_key_frequencies", [])

        print(f"{p:>5} | {n_pairs:>5} | {n_train:>7} | {grok_str:>10} | {train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {key_freqs}")

    print("=" * 90)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prime p sweep for grokking experiments")
    parser.add_argument("--config", type=str, default="configs/sweep_primes.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary of existing runs")
    parser.add_argument("--primes", type=int, nargs="+", default=None,
                        help="Override primes to sweep (default: all 12)")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    output_root = Path(args.output_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES

    if args.summary_only:
        print_summary(output_root, primes, logger)
        return

    base_config = load_config(args.config)
    logger.info(f"Base config: d_model={base_config['d_model']}, wd={base_config.get('weight_decay', 1.0)}, "
                f"lr={base_config.get('lr', 1e-3)}, beta2={base_config.get('beta2', 0.999)}")
    logger.info(f"Sweep primes: {primes}")
    logger.info(f"Device: {device}")

    total = len(primes)
    completed = []
    skipped = []
    failed = []

    for run_num, p in enumerate(primes, 1):
        config = dict(base_config)
        config["p"] = p
        rid = run_id(config)
        run_dir = output_root / rid

        print(f"\n{'='*60}")
        n_pairs = p * p
        n_train = int(n_pairs * 0.3)
        logger.info(f"[{run_num}/{total}] p={p} ({n_pairs} pairs, {n_train} training) — run_id: {rid}")

        if (run_dir / "metrics.json").exists():
            logger.info(f"SKIP: {run_dir} already exists")
            skipped.append(p)
            continue

        logger.info(f"Training for {config['max_epochs']} epochs...")
        t0 = time.time()
        try:
            saved_dir = train_single_run(config, output_root, device, logger)
            elapsed = time.time() - t0
            logger.info(f"Training complete in {elapsed:.0f}s")
            completed.append(p)

            if not args.skip_figures:
                generate_figures_for_run(saved_dir, logger)
        except Exception as e:
            logger.error(f"FAILED p={p}: {e}")
            failed.append(p)
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(completed)} trained, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed runs: {failed}")

    print_summary(output_root, primes, logger)


if __name__ == "__main__":
    main()
