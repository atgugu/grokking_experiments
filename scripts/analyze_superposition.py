#!/usr/bin/env python
"""Deep mechanistic analysis of superposition in single-head grokking models.

Analyzes how a single attention head encodes 5+ Fourier frequencies
simultaneously via superposition, comparing h=1, h=4, and h=16 models.

Generates 8 publication-quality figures + JSON metrics to
results/superposition_analysis/.

Usage:
    python scripts/analyze_superposition.py \
        --results-dir results \
        --output-dir results/superposition_analysis
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fourier import dft_matrix, compute_fourier_component_norms, identify_key_frequencies
from src.models.transformer import GrokkingTransformer
from src.models.hooks import ActivationCache
from src.utils import setup_logging

P = 113  # prime modulus


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _run_id_for(n_heads):
    return f"p113_d128_h{n_heads}_mlp512_L1_wd1.0_s42"


def load_model(run_dir, device):
    """Load a trained model from run_dir/model.pt + config.json."""
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)

    model = GrokkingTransformer(
        p=cfg["p"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        d_mlp=cfg["d_mlp"],
        n_layers=cfg["n_layers"],
        activation=cfg.get("activation", "relu"),
        use_layernorm=cfg.get("use_layernorm", False),
        tie_embeddings=cfg.get("tie_embeddings", False),
        mlp_bias=cfg.get("mlp_bias", True),
    )

    ckpt = torch.load(run_dir / "model.pt", map_location=device, weights_only=False)
    state_dict = ckpt if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt else ckpt["model_state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, cfg


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def extract_per_head_circuits(model):
    """Extract OV and QK circuit matrices per head.

    Returns list of dicts with 'W_OV' (d_model, d_model) and 'W_QK' (d_model, d_model).
    """
    block = model.blocks[0]
    W = block.attn.in_proj_weight.detach().cpu()  # (3*d_model, d_model)
    d = model.d_model
    n_h = model.n_heads
    d_h = d // n_h

    W_Q, W_K, W_V = W[:d], W[d:2*d], W[2*d:3*d]
    W_O = block.attn.out_proj.weight.detach().cpu()  # (d_model, d_model)

    heads = []
    for h in range(n_h):
        s = slice(h * d_h, (h + 1) * d_h)
        W_V_h = W_V[s, :]        # (d_head, d_model)
        W_O_h = W_O[:, s]        # (d_model, d_head)
        W_Q_h = W_Q[s, :]        # (d_head, d_model)
        W_K_h = W_K[s, :]        # (d_head, d_model)
        heads.append({
            "W_OV": (W_O_h @ W_V_h).numpy(),   # (d_model, d_model)
            "W_QK": (W_Q_h.T @ W_K_h).numpy(), # (d_model, d_model)
        })
    return heads


# ---------------------------------------------------------------------------
# Fourier subspace analysis
# ---------------------------------------------------------------------------

def compute_fourier_embedding(W_E, p):
    """E_hat = F @ W_E[:p,:] -> (p, d_model) complex Fourier embedding."""
    F = dft_matrix(p)
    return F @ W_E[:p, :]


def get_key_frequencies(model, device, p):
    """Get key frequencies from the model's logit table."""
    logit_table = model.get_logit_table(device).cpu().numpy()
    result = compute_fourier_component_norms(logit_table, p)
    key_freqs = identify_key_frequencies(result["frequency_norms"], n_top=10)
    # De-mirror: keep only f < p/2 representatives
    unique = set()
    for f in key_freqs:
        f = int(f)
        if f != 0:
            unique.add(min(f, p - f))
    return sorted(unique), key_freqs, result["frequency_norms"]


def compute_frequency_subspaces(E_hat, key_freqs, p):
    """For each de-mirrored freq k, extract 2D real subspace from E_hat[k] and E_hat[p-k].

    Returns dict: freq -> (d_model, 2) orthonormal basis via QR.
    """
    subspaces = {}
    for k in key_freqs:
        re = np.real(E_hat[k])
        im = np.imag(E_hat[k])
        basis = np.column_stack([re, im])  # (d_model, 2)
        Q, _ = np.linalg.qr(basis)
        subspaces[k] = Q[:, :2]  # ensure exactly 2 columns
    return subspaces


def compute_subspace_overlaps(subspaces, key_freqs):
    """Compute principal angle cosines between all freq-subspace pairs.

    Returns (n_freq, n_freq) matrix where entry [i,j] = max cosine of
    principal angles between subspace_i and subspace_j.
    """
    n = len(key_freqs)
    overlaps = np.zeros((n, n))
    for i, ki in enumerate(key_freqs):
        for j, kj in enumerate(key_freqs):
            if i == j:
                overlaps[i, j] = 1.0
            else:
                angles = scipy.linalg.subspace_angles(subspaces[ki], subspaces[kj])
                overlaps[i, j] = np.max(np.cos(angles))
    return overlaps


def compute_effective_rank(subspaces, key_freqs, d_model):
    """Stack all subspace bases -> SVD -> effective rank at 99% energy.

    Returns singular values and effective rank.
    """
    bases = np.hstack([subspaces[k] for k in key_freqs])  # (d_model, 2*n_freq)
    _, s, _ = np.linalg.svd(bases, full_matrices=False)
    energy = s ** 2
    cumulative = np.cumsum(energy) / energy.sum()
    eff_rank_99 = int(np.searchsorted(cumulative, 0.99)) + 1
    eff_rank_95 = int(np.searchsorted(cumulative, 0.95)) + 1
    return s, eff_rank_99, eff_rank_95


# ---------------------------------------------------------------------------
# OV frequency coupling
# ---------------------------------------------------------------------------

def compute_ov_frequency_coupling(W_OV, E_hat, key_freqs, p):
    """Compute frequency coupling through OV circuit.

    coupling[i,j] = ||E_hat[ki]^H @ W_OV @ E_hat[kj]||^2_F / (norm_i * norm_j)

    Diagonal = faithful frequency preservation; off-diagonal = interference.
    """
    n = len(key_freqs)
    coupling = np.zeros((n, n))

    for i, ki in enumerate(key_freqs):
        for j, kj in enumerate(key_freqs):
            # E_hat rows are (d_model,) complex vectors
            # For each freq, use both k and p-k directions
            dirs_i = np.vstack([E_hat[ki], E_hat[p - ki]])  # (2, d_model)
            dirs_j = np.vstack([E_hat[kj], E_hat[p - kj]])  # (2, d_model)

            # Coupling: project OV through freq subspaces
            # M = dirs_i @ W_OV @ dirs_j^H -> (2, 2) complex matrix
            M = dirs_i @ W_OV @ dirs_j.conj().T
            coupling[i, j] = np.sum(np.abs(M) ** 2)

    # Normalize by geometric mean of diagonal
    diag = np.diag(coupling).copy()
    diag[diag == 0] = 1e-10
    for i in range(n):
        for j in range(n):
            coupling[i, j] /= np.sqrt(diag[i] * diag[j])

    return coupling


def compute_full_circuit_fourier(model, p, device):
    """Compute full OV circuit coupling in frequency space.

    C = F @ W_U @ W_OV @ W_E[:p,:]^T @ F^H -> (p, p) frequency coupling.
    Summed over all heads.
    """
    heads = extract_per_head_circuits(model)
    W_E = model.W_E.weight.detach().cpu().numpy()  # (p+1, d_model)
    W_U = model.W_U.weight.detach().cpu().numpy()   # (p, d_model)

    # Sum W_OV over all heads
    W_OV_total = sum(h["W_OV"] for h in heads)  # (d_model, d_model)

    F_mat = dft_matrix(p)
    # Full circuit: W_U @ W_OV @ W_E[:p]^T maps (p,) -> (p,)
    # In Fourier basis: F @ W_U @ W_OV @ W_E[:p]^T @ F^H
    circuit = W_U @ W_OV_total @ W_E[:p, :].T  # (p, p)
    C = F_mat @ circuit @ F_mat.conj().T  # (p, p) complex
    return np.abs(C) ** 2


# ---------------------------------------------------------------------------
# Attention pattern analysis
# ---------------------------------------------------------------------------

def build_all_inputs(p, device):
    """Build all p^2 inputs [a, b, =]."""
    a_vals = torch.arange(p, device=device).repeat_interleave(p)
    b_vals = torch.arange(p, device=device).repeat(p)
    eq_val = p  # equals token index = p (vocab_size - 1)
    eq_vals = torch.full((p * p,), eq_val, dtype=torch.long, device=device)
    return torch.stack([a_vals, b_vals, eq_vals], dim=1)


def analyze_attention_patterns(model, p, device):
    """Compute mean attention patterns (per-head if multi-head).

    Returns dict with 'mean_attn' (n_heads, 3, 3) and per-head info.
    """
    inputs = build_all_inputs(p, device)
    n_heads = model.n_heads
    block = model.blocks[0]

    # Temporarily enable per-head attention weights
    old_avg = getattr(block.attn, '_avg_attn_weights', True)
    # PyTorch MHA: average_attn_weights parameter controls averaging
    # We need to set it to False for per-head patterns
    block.attn.average_attn_weights = False

    with torch.no_grad():
        # Forward in chunks
        all_attn = []
        chunk_size = 2048
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            _ = model(chunk)
            # Get per-head attention from cached weights
            attn = block._attn_weights  # (batch, n_heads, 3, 3) or (batch, 3, 3)
            all_attn.append(attn.cpu())

    block.attn.average_attn_weights = old_avg

    attn_all = torch.cat(all_attn, dim=0)  # (p^2, n_heads, 3, 3) or (p^2, 3, 3)
    if attn_all.dim() == 3:
        attn_all = attn_all.unsqueeze(1)  # -> (p^2, 1, 3, 3)

    mean_attn = attn_all.mean(dim=0).numpy()  # (n_heads, 3, 3)

    # Attention variation by (a mod k) for key frequencies
    a_vals = inputs[:, 0].cpu().numpy()

    return {
        "mean_attn": mean_attn,
        "attn_all": attn_all.numpy(),
        "a_vals": a_vals,
    }


# ---------------------------------------------------------------------------
# Neuron polysemanticity
# ---------------------------------------------------------------------------

def get_neuron_activations(model, p, device):
    """Run all p^2 inputs and capture MLP post-ReLU activations at position 2."""
    inputs = build_all_inputs(p, device)

    with ActivationCache(model) as cache:
        with torch.no_grad():
            _ = model(inputs)
        neuron_acts = cache.neuron_activations[0]  # (p^2, 3, d_mlp)

    # Extract position 2 (= token) activations, reshape to (p, p, d_mlp)
    acts_eq = neuron_acts[:, 2, :].cpu().numpy()  # (p^2, d_mlp)
    return acts_eq.reshape(p, p, -1)  # (p, p, d_mlp)


def analyze_neuron_superposition(neuron_acts, key_freqs, p):
    """Analyze neuron polysemanticity: R^2 for top-1 and top-2 de-mirrored frequency groups.

    Mirror frequencies (k and p-k) are grouped together since they encode the
    same cosine component. R^2_top1 = energy in the dominant (k, p-k) pair.
    R^2_top2 = energy in the top two (k, p-k) pairs.

    Returns (d_mlp,) arrays for r2_top1, r2_top2, and dominant frequencies.
    """
    F = dft_matrix(p)
    d_mlp = neuron_acts.shape[2]

    r2_top1 = np.zeros(d_mlp)
    r2_top2 = np.zeros(d_mlp)
    dom_freq = np.zeros(d_mlp, dtype=int)
    second_freq = np.zeros(d_mlp, dtype=int)

    for n in range(d_mlp):
        act_n = neuron_acts[:, :, n]  # (p, p)
        act_hat = F @ act_n @ F.T
        norms = np.abs(act_hat) ** 2

        # Marginal frequency energy
        freq_energy = np.zeros(p)
        for k in range(p):
            freq_energy[k] = norms[k, :].sum() + norms[:, k].sum() - norms[k, k]

        freq_energy[0] = 0  # exclude DC
        total = freq_energy.sum()
        if total == 0:
            continue

        # Group mirror frequencies: combine energy of k and p-k
        grouped_energy = {}  # min(k, p-k) -> total energy
        for k in range(1, p):
            canon = min(k, p - k)
            grouped_energy[canon] = grouped_energy.get(canon, 0) + freq_energy[k]

        # Sort groups by energy
        sorted_groups = sorted(grouped_energy.items(), key=lambda x: x[1], reverse=True)

        dom_freq[n] = sorted_groups[0][0]
        r2_top1[n] = sorted_groups[0][1] / total

        if len(sorted_groups) > 1:
            second_freq[n] = sorted_groups[1][0]
            r2_top2[n] = (sorted_groups[0][1] + sorted_groups[1][1]) / total
        else:
            r2_top2[n] = r2_top1[n]

    return r2_top1, r2_top2, dom_freq, second_freq


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_ov_coupling(coupling, key_freqs, fig_dir):
    """Figure 1: OV frequency coupling heatmap for primary model."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(coupling, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(key_freqs)))
    ax.set_xticklabels([str(k) for k in key_freqs], fontsize=10)
    ax.set_yticks(range(len(key_freqs)))
    ax.set_yticklabels([str(k) for k in key_freqs], fontsize=10)
    ax.set_xlabel("Input frequency (source)", fontsize=11)
    ax.set_ylabel("Output frequency (target)", fontsize=11)
    ax.set_title("OV Circuit Frequency Coupling (h=1)\nDiagonal = preservation, off-diagonal = interference",
                 fontsize=11, fontweight="bold")

    # Annotate values
    for i in range(len(key_freqs)):
        for j in range(len(key_freqs)):
            color = "white" if coupling[i, j] < coupling.max() * 0.7 else "black"
            ax.text(j, i, f"{coupling[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Normalized coupling strength")
    fig.tight_layout()
    fig.savefig(fig_dir / "ov_coupling_h1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: ov_coupling_h1.png")


def fig2_subspace_overlaps(overlaps, key_freqs, fig_dir):
    """Figure 2: Principal angle cosines between frequency subspaces (h=1)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(overlaps, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(len(key_freqs)))
    ax.set_xticklabels([str(k) for k in key_freqs], fontsize=10)
    ax.set_yticks(range(len(key_freqs)))
    ax.set_yticklabels([str(k) for k in key_freqs], fontsize=10)
    ax.set_xlabel("Frequency", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Embedding Subspace Overlaps (h=1)\nMax cos(principal angle) between 2D frequency subspaces",
                 fontsize=11, fontweight="bold")

    for i in range(len(key_freqs)):
        for j in range(len(key_freqs)):
            color = "white" if overlaps[i, j] > 0.6 else "black"
            ax.text(j, i, f"{overlaps[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Max cos(principal angle)")
    fig.tight_layout()
    fig.savefig(fig_dir / "subspace_overlaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: subspace_overlaps.png")


def fig3_effective_rank(all_svd, all_ranks, heads_list, key_freqs_per_model, fig_dir):
    """Figure 3: SVD spectra + effective rank bars for h=1,4,16."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: SVD spectra overlay
    ax = axes[0]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    for i, (nh, s_vals) in enumerate(zip(heads_list, all_svd)):
        n_freq = len(key_freqs_per_model[nh])
        max_rank = 2 * n_freq
        ax.plot(range(1, len(s_vals) + 1), s_vals / s_vals[0], "o-",
                color=colors[i], label=f"h={nh} ({n_freq} freqs, max rank={max_rank})",
                markersize=4, linewidth=1.5)

    ax.axhline(0.01, color="gray", linestyle="--", alpha=0.5, label="1% threshold")
    ax.set_xlabel("Singular value index", fontsize=11)
    ax.set_ylabel("Normalized singular value", fontsize=11)
    ax.set_title("SVD Spectrum of Frequency Subspace Bases", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: effective rank bars
    ax2 = axes[1]
    x = range(len(heads_list))
    eff_99 = [all_ranks[nh]["eff_rank_99"] for nh in heads_list]
    eff_95 = [all_ranks[nh]["eff_rank_95"] for nh in heads_list]
    max_rank = [2 * len(key_freqs_per_model[nh]) for nh in heads_list]

    width = 0.25
    ax2.bar([xi - width for xi in x], max_rank, width, label="Max rank (2 x n_freq)",
            color="#bdc3c7", edgecolor="black", linewidth=0.8)
    ax2.bar([xi for xi in x], eff_99, width, label="Effective rank (99%)",
            color="#3498db", edgecolor="black", linewidth=0.8)
    ax2.bar([xi + width for xi in x], eff_95, width, label="Effective rank (95%)",
            color="#2ecc71", edgecolor="black", linewidth=0.8)

    for xi, r99, r95 in zip(x, eff_99, eff_95):
        ax2.text(xi, r99 + 0.2, str(r99), ha="center", fontsize=10, fontweight="bold")
        ax2.text(xi + width, r95 + 0.2, str(r95), ha="center", fontsize=10)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={nh}" for nh in heads_list], fontsize=10)
    ax2.set_ylabel("Rank", fontsize=11)
    ax2.set_title("Effective Rank of Frequency Subspaces\n(lower = more superposition)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "effective_rank_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: effective_rank_comparison.png")


def fig4_attention_h1(attn_result, key_freqs, p, fig_dir):
    """Figure 4: Mean attention pattern + frequency-conditioned attention for h=1."""
    mean_attn = attn_result["mean_attn"]  # (n_heads, 3, 3)
    n_heads = mean_attn.shape[0]

    n_cols = 1 + min(3, len(key_freqs))
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 3.5))
    if n_cols == 1:
        axes = [axes]

    pos_labels = ["a", "b", "="]

    # Mean attention (sum over heads if multiple, but for h=1 it's just one)
    ax = axes[0]
    attn_display = mean_attn.mean(axis=0) if n_heads > 1 else mean_attn[0]
    im = ax.imshow(attn_display, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_xticklabels(pos_labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(pos_labels)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    ax.set_title("Mean Attention (h=1)", fontweight="bold")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{attn_display[i,j]:.3f}", ha="center", va="center", fontsize=9)

    # Attention conditioned on (a mod k) for top key frequencies
    attn_all = attn_result["attn_all"]  # (p^2, n_heads, 3, 3)
    a_vals = attn_result["a_vals"]

    for col_idx, k in enumerate(key_freqs[:min(3, len(key_freqs))]):
        ax = axes[1 + col_idx]
        # Group by a mod k
        n_groups = min(k, 6)  # show up to 6 residue classes
        attn_eq_row = attn_all[:, 0, 2, :]  # (p^2, 3) - query=2 (=), all keys, head 0

        group_means = []
        for r in range(n_groups):
            mask = (a_vals % k) == r
            if mask.sum() > 0:
                group_means.append(attn_eq_row[mask].mean(axis=0))

        if group_means:
            group_means = np.array(group_means)  # (n_groups, 3)
            im2 = ax.imshow(group_means, cmap="Blues", vmin=0, vmax=1,
                           aspect="auto", interpolation="nearest")
            ax.set_xticks(range(3))
            ax.set_xticklabels(pos_labels)
            ax.set_ylabel(f"a mod {k}")
            ax.set_yticks(range(n_groups))
            ax.set_yticklabels(range(n_groups), fontsize=8)
            ax.set_title(f"Attn at = pos\nby a mod {k}", fontsize=10, fontweight="bold")

    fig.suptitle("Attention Patterns (h=1 model)", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / "attention_patterns_h1.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: attention_patterns_h1.png")


def fig5_full_circuit_fourier(circuit_coupling, p, key_freqs, fig_dir):
    """Figure 5: Full p x p frequency coupling through OV circuit (log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: full p x p
    ax = axes[0]
    log_coupling = np.log10(circuit_coupling + 1e-10)
    im = ax.imshow(log_coupling, cmap="inferno", interpolation="nearest", aspect="auto")
    ax.set_xlabel("Input frequency k2", fontsize=11)
    ax.set_ylabel("Output frequency k1", fontsize=11)
    ax.set_title("Full OV Circuit in Fourier Basis (log10)\nC = F @ W_U @ W_OV @ W_E^T @ F^H",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="log10(|C|^2)")

    # Mark key frequencies
    for k in key_freqs:
        ax.axhline(k, color="cyan", alpha=0.3, linewidth=0.5)
        ax.axvline(k, color="cyan", alpha=0.3, linewidth=0.5)
        ax.axhline(p - k, color="cyan", alpha=0.3, linewidth=0.5)
        ax.axvline(p - k, color="cyan", alpha=0.3, linewidth=0.5)

    # Right: zoomed to key frequencies only
    ax2 = axes[1]
    all_kf = sorted(set(list(key_freqs) + [p - k for k in key_freqs]))
    zoomed = circuit_coupling[np.ix_(all_kf, all_kf)]
    log_zoomed = np.log10(zoomed + 1e-10)
    im2 = ax2.imshow(log_zoomed, cmap="inferno", interpolation="nearest")
    ax2.set_xticks(range(len(all_kf)))
    ax2.set_xticklabels([str(k) for k in all_kf], fontsize=7, rotation=45)
    ax2.set_yticks(range(len(all_kf)))
    ax2.set_yticklabels([str(k) for k in all_kf], fontsize=7)
    ax2.set_xlabel("Input frequency", fontsize=11)
    ax2.set_ylabel("Output frequency", fontsize=11)
    ax2.set_title("Zoomed: Key Frequencies Only (log10)", fontsize=11, fontweight="bold")
    plt.colorbar(im2, ax=ax2, label="log10(|C|^2)")

    fig.tight_layout()
    fig.savefig(fig_dir / "full_circuit_fourier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: full_circuit_fourier.png")


def fig6_neuron_polysemanticity(r2_top1, r2_top2, dom_freq, key_freqs, p, fig_dir):
    """Figure 6: R^2_top1 vs R^2_top2 scatter for monosemantic vs polysemantic."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left: scatter
    ax = axes[0]
    # Color by whether dominant freq is a key freq (dom_freq is already de-mirrored)
    key_set = set(key_freqs)
    is_key = np.array([int(d) in key_set for d in dom_freq])
    active = r2_top1 > 0.05  # only show active neurons

    ax.scatter(r2_top1[active & is_key], r2_top2[active & is_key],
               alpha=0.5, s=20, c="#2ca02c", label=f"Key freq neuron (n={sum(active & is_key)})")
    ax.scatter(r2_top1[active & ~is_key], r2_top2[active & ~is_key],
               alpha=0.5, s=20, c="#7f7f7f", label=f"Other neuron (n={sum(active & ~is_key)})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x (monosemantic)")
    ax.set_xlabel("R^2 (top-1 frequency)", fontsize=11)
    ax.set_ylabel("R^2 (top-2 frequencies)", fontsize=11)
    ax.set_title("Neuron Polysemanticity (h=1)\nAbove diagonal = multi-frequency encoding",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    # Right: histogram of R^2 gap (r2_top2 - r2_top1) for key-freq neurons
    ax2 = axes[1]
    gap = r2_top2[active & is_key] - r2_top1[active & is_key]
    ax2.hist(gap, bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    ax2.axvline(0.1, color="red", linestyle="--", alpha=0.7, label="Polysemantic threshold")
    ax2.set_xlabel("R^2_top2 - R^2_top1 (frequency gap)", fontsize=11)
    ax2.set_ylabel("Count", fontsize=11)
    ax2.set_title("Distribution of Multi-Frequency Gap\n(key-freq neurons only)",
                   fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    n_poly = int((gap > 0.1).sum())
    n_mono = int((gap <= 0.1).sum())
    ax2.text(0.95, 0.95, f"Mono: {n_mono}\nPoly: {n_poly}",
             transform=ax2.transAxes, fontsize=10, va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.tight_layout()
    fig.savefig(fig_dir / "neuron_polysemanticity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: neuron_polysemanticity.png")


def fig7_interference_comparison(interference_data, heads_list, fig_dir):
    """Figure 7: Interference ratio by n_heads."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: interference ratio bar chart
    ax = axes[0]
    ratios = [interference_data[nh]["interference_ratio"] for nh in heads_list]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(range(len(heads_list)), ratios, color=colors,
                  edgecolor="black", linewidth=0.8)
    ax.set_xticks(range(len(heads_list)))
    ax.set_xticklabels([f"h={nh}" for nh in heads_list], fontsize=11)
    ax.set_ylabel("Interference Ratio (off-diag / diag)", fontsize=11)
    ax.set_title("OV Circuit Interference Ratio\n(higher = more cross-frequency coupling)",
                 fontsize=11, fontweight="bold")
    for i, r in enumerate(ratios):
        ax.text(i, r + 0.01, f"{r:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)

    # Right: diagonal dominance
    ax2 = axes[1]
    diag_means = [interference_data[nh]["diag_mean"] for nh in heads_list]
    offdiag_means = [interference_data[nh]["offdiag_mean"] for nh in heads_list]

    x = np.arange(len(heads_list))
    width = 0.35
    ax2.bar(x - width/2, diag_means, width, label="Diagonal (preservation)",
            color="#2ca02c", edgecolor="black", linewidth=0.8)
    ax2.bar(x + width/2, offdiag_means, width, label="Off-diagonal (interference)",
            color="#e74c3c", edgecolor="black", linewidth=0.8)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={nh}" for nh in heads_list], fontsize=11)
    ax2.set_ylabel("Mean Coupling Strength", fontsize=11)
    ax2.set_title("Diagonal vs Off-Diagonal Coupling\n(normalized OV frequency coupling)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(fig_dir / "interference_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: interference_comparison.png")


def fig8_svd_spectra_overlay(all_svd, heads_list, key_freqs_per_model, fig_dir):
    """Figure 8: Overlaid SVD spectra of key-freq subspaces."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    markers = ["o", "s", "D"]

    for i, (nh, s_vals) in enumerate(zip(heads_list, all_svd)):
        n_freq = len(key_freqs_per_model[nh])
        max_dim = 2 * n_freq
        # Cumulative energy
        energy = (s_vals / s_vals[0]) ** 2
        cumulative = np.cumsum(energy) / energy.sum()
        ax.plot(range(1, len(cumulative) + 1), cumulative, f"-{markers[i]}",
                color=colors[i], label=f"h={nh} (max dim={max_dim})",
                markersize=5, linewidth=1.5)

    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="95% energy")
    ax.axhline(0.99, color="gray", linestyle=":", alpha=0.5, label="99% energy")
    ax.set_xlabel("Number of dimensions", fontsize=11)
    ax.set_ylabel("Cumulative energy fraction", fontsize=11)
    ax.set_title("Cumulative SVD Energy of Frequency Subspaces\n(steeper = less superposition)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    fig.tight_layout()
    fig.savefig(fig_dir / "svd_spectra_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: svd_spectra_overlay.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Superposition analysis of grokking models")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/superposition_analysis")
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--primary", type=int, default=1, help="Primary model for detailed analysis")
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    heads_list = args.heads
    primary = args.primary

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    models = {}
    configs = {}
    for nh in heads_list:
        run_dir = results_root / _run_id_for(nh)
        if not (run_dir / "model.pt").exists():
            logger.warning(f"Missing model for h={nh}: {run_dir}")
            continue
        model, cfg = load_model(run_dir, device)
        models[nh] = model
        configs[nh] = cfg
        logger.info(f"Loaded h={nh}: {model.count_parameters()} params")

    if primary not in models:
        logger.error(f"Primary model h={primary} not found!")
        return

    # -----------------------------------------------------------------------
    # Sanity checks
    # -----------------------------------------------------------------------
    logger.info("--- Sanity checks ---")
    F_mat = dft_matrix(P)
    unitarity_err = np.max(np.abs(F_mat @ F_mat.conj().T - np.eye(P)))
    logger.info(f"DFT unitarity error: {unitarity_err:.2e} (should be ~1e-14)")
    assert unitarity_err < 1e-10, "DFT matrix not unitary!"

    # -----------------------------------------------------------------------
    # Per-model analysis
    # -----------------------------------------------------------------------
    all_key_freqs = {}       # de-mirrored key frequencies
    all_raw_key_freqs = {}   # raw top-10
    all_freq_norms = {}
    all_heads_circuits = {}
    all_E_hat = {}
    all_subspaces = {}
    all_svd_vals = []
    all_ranks = {}
    all_coupling = {}
    interference_data = {}

    for nh in heads_list:
        if nh not in models:
            continue
        model = models[nh]
        logger.info(f"\n=== Analyzing h={nh} ===")

        # Key frequencies
        key_freqs, raw_kf, freq_norms = get_key_frequencies(model, device, P)
        all_key_freqs[nh] = key_freqs
        all_raw_key_freqs[nh] = raw_kf
        all_freq_norms[nh] = freq_norms
        logger.info(f"  Key frequencies (de-mirrored): {key_freqs}")

        # Fourier embedding
        W_E = model.W_E.weight.detach().cpu().numpy()
        E_hat = compute_fourier_embedding(W_E, P)
        all_E_hat[nh] = E_hat

        # Per-head circuits
        heads_circuits = extract_per_head_circuits(model)
        all_heads_circuits[nh] = heads_circuits
        logger.info(f"  Extracted {len(heads_circuits)} head circuit(s)")

        # Frequency subspaces
        subspaces = compute_frequency_subspaces(E_hat, key_freqs, P)
        all_subspaces[nh] = subspaces

        # Effective rank
        s_vals, eff_99, eff_95 = compute_effective_rank(subspaces, key_freqs, model.d_model)
        all_svd_vals.append(s_vals)
        all_ranks[nh] = {"eff_rank_99": eff_99, "eff_rank_95": eff_95}
        logger.info(f"  Effective rank: 99%={eff_99}, 95%={eff_95} (max={2*len(key_freqs)})")

        # OV coupling (sum over heads)
        W_OV_total = sum(h["W_OV"] for h in heads_circuits)
        coupling = compute_ov_frequency_coupling(W_OV_total, E_hat, key_freqs, P)
        all_coupling[nh] = coupling

        # Interference ratio
        n_kf = len(key_freqs)
        diag_vals = np.diag(coupling)
        mask = ~np.eye(n_kf, dtype=bool)
        offdiag_vals = coupling[mask]
        interference_ratio = offdiag_vals.mean() / diag_vals.mean() if diag_vals.mean() > 0 else 0.0
        interference_data[nh] = {
            "interference_ratio": float(interference_ratio),
            "diag_mean": float(diag_vals.mean()),
            "offdiag_mean": float(offdiag_vals.mean()),
        }
        logger.info(f"  Interference ratio: {interference_ratio:.4f}")

    # -----------------------------------------------------------------------
    # Primary model: detailed analysis
    # -----------------------------------------------------------------------
    logger.info(f"\n=== Detailed analysis of primary model h={primary} ===")

    # Subspace overlaps
    overlaps = compute_subspace_overlaps(all_subspaces[primary], all_key_freqs[primary])
    logger.info(f"  Max off-diagonal subspace overlap: {overlaps[~np.eye(len(all_key_freqs[primary]), dtype=bool)].max():.4f}")

    # Full circuit Fourier coupling
    circuit_coupling = compute_full_circuit_fourier(models[primary], P, device)

    # Attention patterns
    logger.info("  Computing attention patterns...")
    attn_result = analyze_attention_patterns(models[primary], P, device)
    mean_attn_eq = attn_result["mean_attn"][0, 2, :]  # query=2 (=), head 0
    logger.info(f"  Mean attention at = position: a={mean_attn_eq[0]:.3f}, b={mean_attn_eq[1]:.3f}, ={mean_attn_eq[2]:.3f}")

    # Neuron polysemanticity
    logger.info("  Computing neuron activations...")
    neuron_acts = get_neuron_activations(models[primary], P, device)
    r2_top1, r2_top2, dom_freq, second_freq = analyze_neuron_superposition(
        neuron_acts, all_key_freqs[primary], P
    )
    n_active = int((r2_top1 > 0.05).sum())
    n_polysemantic = int(((r2_top2 - r2_top1) > 0.1).sum())
    logger.info(f"  Active neurons: {n_active}/512, Polysemantic: {n_polysemantic}")

    # Verify Gini matches stored value
    with open(results_root / _run_id_for(primary) / "metrics.json") as f:
        stored_metrics = json.load(f)
    stored_gini = stored_metrics.get("final_gini", 0)
    from src.analysis.fourier import compute_gini_coefficient
    computed_gini = compute_gini_coefficient(all_freq_norms[primary])
    logger.info(f"  Gini check: stored={stored_gini:.4f}, computed={computed_gini:.4f}")

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    logger.info("\n=== Generating figures ===")

    logger.info("1/8: OV coupling heatmap (h=1)")
    fig1_ov_coupling(all_coupling[primary], all_key_freqs[primary], fig_dir)

    logger.info("2/8: Subspace overlaps (h=1)")
    fig2_subspace_overlaps(overlaps, all_key_freqs[primary], fig_dir)

    logger.info("3/8: Effective rank comparison")
    fig3_effective_rank(all_svd_vals, all_ranks, heads_list, all_key_freqs, fig_dir)

    logger.info("4/8: Attention patterns (h=1)")
    fig4_attention_h1(attn_result, all_key_freqs[primary], P, fig_dir)

    logger.info("5/8: Full circuit Fourier coupling")
    fig5_full_circuit_fourier(circuit_coupling, P, all_key_freqs[primary], fig_dir)

    logger.info("6/8: Neuron polysemanticity")
    fig6_neuron_polysemanticity(r2_top1, r2_top2, dom_freq, all_key_freqs[primary], P, fig_dir)

    logger.info("7/8: Interference comparison")
    fig7_interference_comparison(interference_data, heads_list, fig_dir)

    logger.info("8/8: SVD spectra overlay")
    fig8_svd_spectra_overlay(all_svd_vals, heads_list, all_key_freqs, fig_dir)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUPERPOSITION ANALYSIS — SUMMARY")
    print("=" * 90)
    header = (f"{'n_heads':>7} | {'n_key_freq':>10} | {'Eff Rank 99%':>12} | "
              f"{'Eff Rank 95%':>12} | {'Max Rank':>8} | {'Interference':>12} | "
              f"{'Diag Mean':>9}")
    print(header)
    print("-" * 90)

    for nh in heads_list:
        if nh not in models:
            print(f"{nh:>7} | {'MISSING':>10}")
            continue
        n_kf = len(all_key_freqs[nh])
        er99 = all_ranks[nh]["eff_rank_99"]
        er95 = all_ranks[nh]["eff_rank_95"]
        max_r = 2 * n_kf
        ir = interference_data[nh]["interference_ratio"]
        dm = interference_data[nh]["diag_mean"]
        print(f"{nh:>7} | {n_kf:>10} | {er99:>12} | {er95:>12} | {max_r:>8} | {ir:>12.4f} | {dm:>9.4f}")

    print("=" * 90)
    print()
    print(f"Primary model (h={primary}):")
    print(f"  Key frequencies: {all_key_freqs[primary]}")
    print(f"  Max subspace overlap: {overlaps[~np.eye(len(all_key_freqs[primary]), dtype=bool)].max():.4f}")
    print(f"  Active neurons: {n_active}/512")
    print(f"  Polysemantic neurons (gap > 0.1): {n_polysemantic}")
    print(f"  Gini: stored={stored_gini:.4f}, recomputed={computed_gini:.4f}")
    print()

    # -----------------------------------------------------------------------
    # Save JSON metrics
    # -----------------------------------------------------------------------
    metrics_out = {
        "primary_model": primary,
        "heads_analyzed": heads_list,
        "per_model": {},
    }
    for nh in heads_list:
        if nh not in models:
            continue
        metrics_out["per_model"][str(nh)] = {
            "key_frequencies_demirrored": all_key_freqs[nh],
            "n_key_freq": len(all_key_freqs[nh]),
            "effective_rank_99": all_ranks[nh]["eff_rank_99"],
            "effective_rank_95": all_ranks[nh]["eff_rank_95"],
            "max_rank": 2 * len(all_key_freqs[nh]),
            "interference_ratio": interference_data[nh]["interference_ratio"],
            "diag_mean_coupling": interference_data[nh]["diag_mean"],
            "offdiag_mean_coupling": interference_data[nh]["offdiag_mean"],
        }

    metrics_out["primary_details"] = {
        "key_frequencies": all_key_freqs[primary],
        "max_subspace_overlap": float(overlaps[~np.eye(len(all_key_freqs[primary]), dtype=bool)].max()),
        "subspace_overlaps": overlaps.tolist(),
        "active_neurons": n_active,
        "polysemantic_neurons": n_polysemantic,
        "gini_stored": stored_gini,
        "gini_recomputed": computed_gini,
        "mean_attention_at_eq": {
            "to_a": float(mean_attn_eq[0]),
            "to_b": float(mean_attn_eq[1]),
            "to_eq": float(mean_attn_eq[2]),
        },
    }

    out_path = fig_dir / "superposition_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics: {out_path}")
    logger.info(f"\nAll outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
