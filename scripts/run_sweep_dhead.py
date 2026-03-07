#!/usr/bin/env python
"""Run d_head control experiment to resolve head-count vs per-head-capacity confound.

Design: Hold d_head constant, vary n_heads (and therefore d_model).

Configs (d_head=32):
  h=1, d_model=32   (~4K params)
  h=2, d_model=64   (~17K params)
  h=4, d_model=128  (~200K params, matches baseline)

Extra (d_head=64):
  h=1, d_model=64   (~17K params, same total params as h=2/d=64 but single head)

Grid: configs above x p in {43, 113} x seed in {42, 137, 256} = 24 cells.
Existing runs are skipped.
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


# (n_heads, d_model) configurations; d_mlp scales as 4*d_model
SWEEP_CONFIGS = [
    (1, 32),    # d_head=32, d_mlp=128
    (2, 64),    # d_head=32, d_mlp=256
    (4, 128),   # d_head=32, d_mlp=512  (baseline architecture)
    (1, 64),    # d_head=64, d_mlp=256  (capacity control)
]

SWEEP_PRIMES = [43, 113]
SWEEP_SEEDS = [42, 137, 256]


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
    logger.info(f"p={config['p']}, h={config['n_heads']}, d_model={config['d_model']}, "
                f"d_head={d_head}, d_mlp={config['d_mlp']}, "
                f"seed={config['seed']}, Parameters: {model.count_parameters():,}")

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


def print_summary(results_root, configs, primes, seeds, logger):
    """Load metrics from all runs and print summary table."""
    print("\n" + "=" * 110)
    print("d_head CONTROL EXPERIMENT -- SUMMARY")
    print("=" * 110)

    for p in primes:
        n_train = int(p * p * 0.3)
        print(f"\n--- p={p} (N_train={n_train}) ---")
        header = f"{'Config':>20} | {'d_head':>6}"
        for seed in seeds:
            header += f" | {'s=' + str(seed):>18}"
        header += f" | {'mean +/- std':>18}"
        print(header)
        print("-" * 110)

        for h, d_model in configs:
            d_head = d_model // h
            d_mlp = 4 * d_model
            label = f"h={h} d={d_model} mlp={d_mlp}"
            row = f"{label:>20} | {d_head:>6}"

            grok_epochs = []
            for seed in seeds:
                rid = run_id({"p": p, "d_model": d_model, "n_heads": h, "d_mlp": d_mlp,
                              "n_layers": 1, "weight_decay": 1.0, "seed": seed,
                              "lr": 1e-3, "operation": "addition",
                              "train_fraction": 0.3})
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
                    grok_epochs.append(grok_epoch)
                else:
                    cell = f"-- ({test_acc:.2f})"
                row += f" | {cell:>18}"

            # Add mean +/- std
            if grok_epochs:
                import numpy as np
                mean_e = np.mean(grok_epochs)
                std_e = np.std(grok_epochs)
                row += f" | {mean_e:.0f}+/-{std_e:.0f} ({len(grok_epochs)}/{len(seeds)})"
            else:
                row += f" | {'no grokking':>18}"
            print(row)

    print("=" * 110)
    print("Cell format: grok_epoch (final_test_acc)  |  -- = never grokked")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="d_head control experiment for grokking")
    parser.add_argument("--config", type=str, default="configs/sweep_dhead_control.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary of existing runs")
    parser.add_argument("--primes", type=int, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    output_root = Path(args.output_dir)
    primes = args.primes if args.primes else SWEEP_PRIMES
    seeds = args.seeds if args.seeds else SWEEP_SEEDS

    if args.summary_only:
        print_summary(output_root, SWEEP_CONFIGS, primes, seeds, logger)
        return

    base_config = load_config(args.config)
    logger.info(f"d_head control experiment")
    logger.info(f"Configs: {SWEEP_CONFIGS}")
    logger.info(f"Primes:  {primes}")
    logger.info(f"Seeds:   {seeds}")
    logger.info(f"Device:  {device}")

    # Build list of all cells
    cells = [(h, d_model, p, s) for h, d_model in SWEEP_CONFIGS
             for p in primes for s in seeds]
    total = len(cells)

    completed = []
    skipped = []
    failed = []

    for run_num, (n_heads, d_model, p, seed) in enumerate(cells, 1):
        config = dict(base_config)
        config["p"] = p
        config["n_heads"] = n_heads
        config["d_model"] = d_model
        config["d_mlp"] = 4 * d_model  # Scale MLP proportionally
        config["seed"] = seed
        rid = run_id(config)
        run_dir = output_root / rid
        d_head = d_model // n_heads
        n_train = int(p * p * 0.3)

        print(f"\n{'='*60}")
        logger.info(f"[{run_num}/{total}] p={p} h={n_heads} d_model={d_model} d_head={d_head} "
                     f"d_mlp={config['d_mlp']} seed={seed} (N_train={n_train}) -- {rid}")

        if (run_dir / "metrics.json").exists():
            logger.info(f"SKIP: {run_dir} already exists")
            skipped.append((n_heads, d_model, p, seed))
            continue

        logger.info(f"Training for {config['max_epochs']} epochs...")
        t0 = time.time()
        try:
            saved_dir = train_single_run(config, output_root, device, logger)
            elapsed = time.time() - t0
            logger.info(f"Training complete in {elapsed:.0f}s")
            completed.append((n_heads, d_model, p, seed))

            if not args.skip_figures:
                generate_figures_for_run(saved_dir, logger)
        except Exception as e:
            logger.error(f"FAILED h={n_heads} d={d_model} p={p} seed={seed}: {e}")
            failed.append((n_heads, d_model, p, seed))
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(completed)} trained, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed runs: {failed}")

    print_summary(output_root, SWEEP_CONFIGS, primes, seeds, logger)


if __name__ == "__main__":
    main()
