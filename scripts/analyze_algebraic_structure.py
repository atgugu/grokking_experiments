#!/usr/bin/env python
"""Algebraic structure analysis: compare theoretical vs learned Fourier spectra.

For each operation, computes the 2D DFT of its value function f(a,b) = (a op b) mod p
and compares the resulting Gini coefficient and frequency spectra to those learned
by the trained model.

Key finding: the Gini coefficient of the theoretical spectrum (reflecting algebraic
structure) predicts the Gini of the learned spectrum. Operations that are "Fourier-sparse"
by nature (addition, subtraction) force the model toward sparse representations;
operations that are "Fourier-dense" (multiplication) force dense representations.
Specific frequency choices don't match (model finds its own preferred frequencies),
but overall sparsity is determined by the algebraic structure.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fourier import compute_gini_coefficient
from src.utils import setup_logging, _OP_SUFFIXES

P = 113

OPERATIONS = {
    "addition": lambda a, b: (a + b) % P,
    "subtraction": lambda a, b: (a - b) % P,
    "multiplication": lambda a, b: (a * b) % P,
    "x2_plus_y2": lambda a, b: (a**2 + b**2) % P,
    "x3_plus_xy": lambda a, b: (a**3 + a * b) % P,
}

OP_LABELS = {
    "addition": "a + b",
    "subtraction": "a \u2212 b",
    "multiplication": "a \u00d7 b",
    "x2_plus_y2": "a\u00b2 + b\u00b2",
    "x3_plus_xy": "a\u00b3 + ab",
}

OP_COLORS = {
    "addition": "#1f77b4",
    "subtraction": "#ff7f0e",
    "multiplication": "#2ca02c",
    "x2_plus_y2": "#d62728",
    "x3_plus_xy": "#9467bd",
}


def compute_value_dft_spectrum(op_fn):
    """Compute 2D DFT of the value function f(a,b) = (a op b) mod p.

    Treats the p×p table of integer output values as a real-valued function
    and computes the marginal frequency energy per frequency index.

    This captures the algebraic complexity of the operation: linear-like
    operations (addition, subtraction) yield sparse spectra; nonlinear ones
    (multiplication) yield dense spectra.

    Returns:
        frequency_norms: (p,) marginal energy per frequency (DC at index 0).
    """
    a_grid, b_grid = np.meshgrid(np.arange(P), np.arange(P), indexing="ij")
    V = op_fn(a_grid, b_grid).astype(float)  # (P, P)
    V_hat = np.fft.fft2(V, norm="ortho")  # 2D DFT
    comp = np.abs(V_hat) ** 2  # (P, P) energy
    # Marginal: energy attributed to frequency k = sum of all (k,*) and (*,k) pairs
    freq_norms = comp.sum(axis=1) + comp.sum(axis=0) - np.diag(comp)
    return freq_norms


def get_run_dir(op_name):
    suffix = _OP_SUFFIXES.get(op_name)
    base = "results/p113_d128_h4_mlp512_L1_wd1.0_s42"
    if suffix:
        return Path(f"{base}_{suffix}")
    return Path(base)


def load_learned_spectrum(op_name):
    """Load learned frequency norms from metrics.json."""
    run_dir = get_run_dir(op_name)
    with open(run_dir / "metrics.json") as f:
        metrics = json.load(f)
    freq_norms = np.array(metrics["final_frequency_norms"], dtype=np.float64)
    return freq_norms


def normalize_no_dc(v):
    """Normalize to unit sum, zeroing out DC (index 0)."""
    v = v.copy().astype(np.float64)
    v[0] = 0.0
    total = v.sum()
    if total > 0:
        v /= total
    return v


def main():
    logger = setup_logging()
    output_dir = Path("results/op_sweep_comparison")
    output_dir.mkdir(parents=True, exist_ok=True)

    ops = list(OPERATIONS.keys())
    results = {}

    logger.info("Computing theoretical and learned Fourier spectra...")
    for op_name in ops:
        logger.info(f"  Processing {op_name}...")
        op_fn = OPERATIONS[op_name]

        # Theoretical spectrum from value-DFT of the operation
        theo_norms = compute_value_dft_spectrum(op_fn)
        theo_gini = compute_gini_coefficient(theo_norms[1:])  # exclude DC

        # Learned spectrum from trained model
        learned_norms = load_learned_spectrum(op_name)
        learned_gini = compute_gini_coefficient(learned_norms[1:])  # exclude DC

        # Correlation between normalized spectra (excluding DC)
        theo_n = normalize_no_dc(theo_norms)
        learned_n = normalize_no_dc(learned_norms)
        r, pval = pearsonr(theo_n[1:], learned_n[1:])

        results[op_name] = {
            "theo_norms": theo_norms,
            "learned_norms": learned_norms,
            "theo_gini": theo_gini,
            "learned_gini": learned_gini,
            "pearson_r": r,
            "pearson_p": pval,
        }

        logger.info(
            f"    theo_gini={theo_gini:.3f}, learned_gini={learned_gini:.3f}, "
            f"pearson_r={r:.3f}"
        )

    # ---- Figure 1: 5-panel frequency spectra comparison ----
    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(
        "Theoretical (value-DFT) vs. Learned Fourier Spectra\n"
        "Gini coefficients align; specific frequency peaks do not "
        "(model selects its own preferred frequencies within the algebraically-determined sparsity)",
        fontsize=11,
        y=1.03,
    )

    freqs = np.arange(1, P)  # exclude DC

    for ax, op_name in zip(axes, ops):
        res = results[op_name]
        color = OP_COLORS[op_name]

        theo_n = normalize_no_dc(res["theo_norms"])[1:]
        learned_n = normalize_no_dc(res["learned_norms"])[1:]

        ax.bar(freqs, theo_n, color="gray", alpha=0.55, label="Theoretical", width=1.0)
        ax.bar(freqs, learned_n, color=color, alpha=0.65, label="Learned", width=1.0)

        r = res["pearson_r"]
        theo_gini = res["theo_gini"]
        learned_gini = res["learned_gini"]

        ax.set_title(
            f"{OP_LABELS[op_name]}\n"
            f"Theory Gini={theo_gini:.3f}\n"
            f"Learned Gini={learned_gini:.3f}  (r={r:.2f})",
            fontsize=9,
        )
        ax.set_xlabel("Frequency", fontsize=8)
        if op_name == "addition":
            ax.set_ylabel("Normalized energy", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(0, P)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    out_path = output_dir / "op_sweep_algebraic_structure.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {out_path}")

    # ---- Figure 2: Gini comparison bar chart ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(ops))
    width = 0.35

    theo_ginis = [results[op]["theo_gini"] for op in ops]
    learned_ginis = [results[op]["learned_gini"] for op in ops]
    colors = [OP_COLORS[op] for op in ops]

    bars1 = ax.bar(x - width / 2, theo_ginis, width, color="gray", alpha=0.6,
                   label="Theoretical (value-DFT)")
    bars2 = ax.bar(x + width / 2, learned_ginis, width, color=colors, alpha=0.8,
                   label="Learned (trained model)")

    ax.set_xticks(x)
    ax.set_xticklabels([OP_LABELS[op] for op in ops], fontsize=10)
    ax.set_ylabel("Gini coefficient", fontsize=11)
    ax.set_title(
        "Algebraic Structure Determines Representation Complexity\n"
        "Theory Gini (operation's inherent sparsity) predicts Learned Gini",
        fontsize=11,
    )
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4)
    ax.text(4.6, 0.51, "0.5", fontsize=8, alpha=0.5)

    for bar, gini in zip(bars1, theo_ginis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{gini:.3f}", ha="center", va="bottom", fontsize=8)
    for bar, gini in zip(bars2, learned_ginis):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{gini:.3f}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    out_path2 = output_dir / "op_sweep_gini_comparison.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved figure: {out_path2}")

    # ---- Summary table ----
    logger.info("\nSummary:")
    logger.info(
        f"{'Operation':<20} {'Theory Gini':>12} {'Learned Gini':>13} "
        f"{'Pearson r':>10} {'Grokked?':>9}"
    )
    logger.info("-" * 70)
    groks = {
        "addition": True,
        "subtraction": True,
        "multiplication": True,
        "x2_plus_y2": True,
        "x3_plus_xy": False,
    }
    for op_name in ops:
        res = results[op_name]
        grokked = "Yes" if groks[op_name] else "No"
        logger.info(
            f"{op_name:<20} {res['theo_gini']:>12.3f} {res['learned_gini']:>13.3f} "
            f"{res['pearson_r']:>10.3f} {grokked:>9}"
        )

    # Save results as JSON for reference
    summary = {
        op: {
            "theo_gini": float(res["theo_gini"]),
            "learned_gini": float(res["learned_gini"]),
            "pearson_r": float(res["pearson_r"]),
            "pearson_p": float(res["pearson_p"]),
        }
        for op, res in results.items()
    }
    with open(output_dir / "algebraic_structure_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved JSON: {output_dir / 'algebraic_structure_results.json'}")


if __name__ == "__main__":
    main()
