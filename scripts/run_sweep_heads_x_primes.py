#!/usr/bin/env python
"""Run joint (n_heads x p) sweep: train models for each (p, n_heads) cell.

Tests p in {13, 23, 31, 43, 59, 67, 89, 113} x n_heads in {1, 2, 4, 8}.
Existing runs (from heads sweep at p=113 and primes sweep at h=4) are skipped.

Key scientific questions:
  1. Does K scale as O(log p) regardless of n_heads?
  2. Does the grokking phase transition (p=43->59 at h=4) shift with n_heads?
  3. Does the h=1 speed advantage hold across all primes?
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


SWEEP_PRIMES = [13, 23, 31, 43, 59, 67, 89, 113]
SWEEP_HEADS = [1, 2, 4, 8]


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
    d_head = config["d_model"] // config["n_heads"]
    logger.info(f"p={config['p']}, n_heads={config['n_heads']}, d_head={d_head}, "
                f"Parameters: {model.count_parameters():,}")

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


def print_summary(results_root, primes, heads, logger):
    """Load metrics from all runs and print a 2D summary table."""
    print("\n" + "=" * 100)
    print("JOINT (n_heads x p) SWEEP -- SUMMARY")
    print("=" * 100)

    # Header row
    header = f"{'p':>5} | {'N_train':>7}"
    for h in heads:
        header += f" | {'h=' + str(h):>18}"
    print(header)
    print("-" * 100)

    for p in primes:
        n_train = int(p * p * 0.3)
        row = f"{p:>5} | {n_train:>7}"
        for h in heads:
            rid = f"p{p}_d128_h{h}_mlp512_L1_wd1.0_s42"
            metrics_path = results_root / rid / "metrics.json"

            if not metrics_path.exists():
                row += f" | {'MISSING':>18}"
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

            test_acc = metrics.get("final_test_acc", 0)
            if grok_epoch is not None:
                cell = f"{grok_epoch} ({test_acc:.2f})"
            else:
                cell = f"-- ({test_acc:.2f})"
            row += f" | {cell:>18}"

        print(row)

    print("=" * 100)
    print("Cell format: grok_epoch (final_test_acc)  |  -- = never grokked")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Joint (n_heads x p) sweep for grokking experiments")
    parser.add_argument("--config", type=str, default="configs/sweep_heads_x_primes.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary of existing runs")
    parser.add_argument("--primes", type=int, nargs="+", default=None,
                        help="Override primes to sweep (default: all 8)")
    parser.add_argument("--heads", type=int, nargs="+", default=None,
                        help="Override head counts to sweep (default: all 4)")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    output_root = Path(args.output_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES
    heads = args.heads if args.heads else SWEEP_HEADS

    if args.summary_only:
        print_summary(output_root, primes, heads, logger)
        return

    base_config = load_config(args.config)
    logger.info(f"Base config: d_model={base_config['d_model']}, wd={base_config.get('weight_decay', 1.0)}, "
                f"lr={base_config.get('lr', 1e-3)}")
    logger.info(f"Sweep primes: {primes}")
    logger.info(f"Sweep heads:  {heads}")
    logger.info(f"Device: {device}")

    # Build list of all (p, n_heads) cells
    cells = [(p, h) for p in primes for h in heads]
    total = len(cells)

    completed = []
    skipped = []
    failed = []

    for run_num, (p, n_heads) in enumerate(cells, 1):
        config = dict(base_config)
        config["p"] = p
        config["n_heads"] = n_heads
        rid = run_id(config)
        run_dir = output_root / rid
        d_head = config["d_model"] // n_heads
        n_train = int(p * p * 0.3)

        print(f"\n{'='*60}")
        logger.info(f"[{run_num}/{total}] p={p} h={n_heads} (d_head={d_head}, N_train={n_train}) -- {rid}")

        if (run_dir / "metrics.json").exists():
            logger.info(f"SKIP: {run_dir} already exists")
            skipped.append((p, n_heads))
            continue

        logger.info(f"Training for {config['max_epochs']} epochs...")
        t0 = time.time()
        try:
            saved_dir = train_single_run(config, output_root, device, logger)
            elapsed = time.time() - t0
            logger.info(f"Training complete in {elapsed:.0f}s")
            completed.append((p, n_heads))

            if not args.skip_figures:
                generate_figures_for_run(saved_dir, logger)
        except Exception as e:
            logger.error(f"FAILED p={p} h={n_heads}: {e}")
            failed.append((p, n_heads))
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(completed)} trained, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed runs: {failed}")

    print_summary(output_root, primes, heads, logger)


if __name__ == "__main__":
    main()
