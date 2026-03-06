#!/usr/bin/env python
"""Run effective weight-decay (wd × lr) unification sweep.

Tests 4 new (wd, lr) pairs designed so that eff_wd = wd × lr matches
existing baseline values, enabling a direct test of whether grokking
dynamics are controlled purely by the product eff_wd.

New runs:
    wd=2.0, lr=5e-4  → eff_wd=1e-3  (compare to baseline wd=1.0, lr=1e-3)
    wd=0.5, lr=2e-3  → eff_wd=1e-3  (compare to baseline wd=1.0, lr=1e-3)
    wd=3.0, lr=1e-3  → eff_wd=3e-3  (compare to LR sweep wd=1.0, lr=3e-3)
    wd=1.0, lr=2e-3  → eff_wd=2e-3  (compare to WD sweep wd=2.0, lr=1e-3)
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


# (wd, lr) pairs — all yield a key eff_wd value to test unification
SWEEP_CONFIGS = [
    (2.0, 5e-4),   # eff_wd = 1e-3
    (0.5, 2e-3),   # eff_wd = 1e-3
    (3.0, 1e-3),   # eff_wd = 3e-3
    (1.0, 2e-3),   # eff_wd = 2e-3
]


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


def _run_id_for(wd, lr):
    """Build expected run_id for a given (wd, lr) pair."""
    base = f"p113_d128_h4_mlp512_L1_wd{wd}_s42"
    if abs(lr - 1e-3) > 1e-9:
        return f"{base}_lr{lr}"
    return base


def print_summary(results_root, sweep_configs, logger):
    """Load metrics from all runs and print comparison table grouped by eff_wd."""
    print("\n" + "=" * 100)
    print("EFFECTIVE WD SWEEP — SUMMARY")
    print("=" * 100)
    header = (f"{'eff_wd':>8} | {'wd':>5} | {'lr':>8} | {'Grok Epoch':>10} | "
              f"{'Train Acc':>9} | {'Test Acc':>8} | {'Gini':>6} | {'W Norm':>8}")
    print(header)
    print("-" * 100)

    # Group by eff_wd for display
    from collections import defaultdict
    groups = defaultdict(list)
    for wd, lr in sweep_configs:
        eff_wd = wd * lr
        groups[eff_wd].append((wd, lr))

    for eff_wd in sorted(groups.keys()):
        for wd, lr in groups[eff_wd]:
            rid = _run_id_for(wd, lr)
            run_dir = results_root / rid
            metrics_path = run_dir / "metrics.json"

            if not metrics_path.exists():
                print(f"{eff_wd:>8.0e} | {wd:>5} | {lr:>8.0e} | {'MISSING':>10} | "
                      f"{'—':>9} | {'—':>8} | {'—':>6} | {'—':>8}")
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
            print(f"{eff_wd:>8.0e} | {wd:>5} | {lr:>8.0e} | {grok_str:>10} | "
                  f"{train_acc:>9.4f} | {test_acc:>8.4f} | {gini:>6.3f} | {w_norm:>8.1f}")
        print("-" * 100)

    print("=" * 100)

    # Key finding: does eff_wd unify grokking time?
    print("\n| KEY TEST: Does eff_wd=1e-3 produce consistent grokking epochs?")
    for wd, lr in [(1.0, 1e-3), (2.0, 5e-4), (0.5, 2e-3)]:
        rid = _run_id_for(wd, lr)
        metrics_path = results_root / rid / "metrics.json"
        if not metrics_path.exists():
            print(f"|   wd={wd}, lr={lr:.0e}: MISSING")
            continue
        with open(metrics_path) as f:
            metrics = json.load(f)
        history = metrics.get("history", {})
        grok_epoch = find_grokking_epoch(history)
        test_acc = metrics.get("final_test_acc", 0)
        if grok_epoch is not None:
            print(f"|   wd={wd}, lr={lr:.0e}: grokked at epoch {grok_epoch} (test_acc={test_acc:.4f})")
        else:
            print(f"|   wd={wd}, lr={lr:.0e}: never grokked (test_acc={test_acc:.4f})")
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Effective WD (wd × lr) unification sweep")
    parser.add_argument("--config", type=str, default="configs/sweep_effective_wd.yaml")
    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--skip-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--summary-only", action="store_true", help="Only print summary of existing runs")
    args = parser.parse_args()

    logger = setup_logging()
    device = get_device()
    output_root = Path(args.output_dir)

    if args.summary_only:
        print_summary(output_root, SWEEP_CONFIGS, logger)
        return

    base_config = load_config(args.config)
    logger.info(f"Base config: p={base_config['p']}, d_model={base_config['d_model']}")
    logger.info(f"Sweep (wd, lr) pairs: {SWEEP_CONFIGS}")
    logger.info(f"Device: {device}")

    total = len(SWEEP_CONFIGS)
    completed = []
    skipped = []
    failed = []

    for run_num, (wd, lr) in enumerate(SWEEP_CONFIGS, 1):
        eff_wd = wd * lr
        config = dict(base_config)
        config["weight_decay"] = wd
        config["lr"] = lr
        rid = run_id(config)
        run_dir = output_root / rid

        print(f"\n{'='*60}")
        logger.info(f"[{run_num}/{total}] wd={wd}, lr={lr:.0e}, eff_wd={eff_wd:.0e} — run_id: {rid}")

        if (run_dir / "metrics.json").exists():
            logger.info(f"SKIP: {run_dir} already exists")
            skipped.append((wd, lr))
            continue

        logger.info(f"Training for {config['max_epochs']} epochs...")
        t0 = time.time()
        try:
            saved_dir = train_single_run(config, output_root, device, logger)
            elapsed = time.time() - t0
            logger.info(f"Training complete in {elapsed:.0f}s")
            completed.append((wd, lr))

            if not args.skip_figures:
                generate_figures_for_run(saved_dir, logger)
        except Exception as e:
            logger.error(f"FAILED wd={wd}, lr={lr:.0e}: {e}")
            failed.append((wd, lr))
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    logger.info(f"Sweep complete: {len(completed)} trained, {len(skipped)} skipped, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed runs: {failed}")

    print_summary(output_root, SWEEP_CONFIGS, logger)


if __name__ == "__main__":
    main()
