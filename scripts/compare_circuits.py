#!/usr/bin/env python
"""Comparative circuit analysis across head counts (h=1 vs h=4 vs h=16).

Deep mechanistic comparison using existing trained checkpoints at p=113.
No new training required -- analysis only.

Produces 5 publication-quality figures + JSON metrics:
1. OV coupling matrices side-by-side (h=1 vs h=4 vs h=16)
2. Causal ablation sensitivity heatmaps
3. Frequency subspace principal angles vs n_heads
4. Neuron polysemanticity distribution per head count
5. Head classification table for h=4 and h=16

Usage:
    python scripts/compare_circuits.py \
        --results-dir results \
        --output-dir results/circuit_comparison
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.linalg
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.analysis.fourier import dft_matrix, compute_fourier_component_norms, identify_key_frequencies
from src.models.transformer import GrokkingTransformer
from src.models.hooks import ActivationCache
from src.utils import setup_logging

P = 113


# ---------------------------------------------------------------------------
# Model loading (reused from analyze_superposition.py)
# ---------------------------------------------------------------------------

def _run_id_for(n_heads):
    return f"p113_d128_h{n_heads}_mlp512_L1_wd1.0_s42"


def load_model(run_dir, device):
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    model = GrokkingTransformer(
        p=cfg["p"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        d_mlp=cfg["d_mlp"], n_layers=cfg["n_layers"],
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
    block = model.blocks[0]
    W = block.attn.in_proj_weight.detach().cpu()
    d = model.d_model
    n_h = model.n_heads
    d_h = d // n_h
    W_Q, W_K, W_V = W[:d], W[d:2*d], W[2*d:3*d]
    W_O = block.attn.out_proj.weight.detach().cpu()
    heads = []
    for h in range(n_h):
        s = slice(h * d_h, (h + 1) * d_h)
        W_V_h = W_V[s, :]
        W_O_h = W_O[:, s]
        W_Q_h = W_Q[s, :]
        W_K_h = W_K[s, :]
        heads.append({
            "W_OV": (W_O_h @ W_V_h).numpy(),
            "W_QK": (W_Q_h.T @ W_K_h).numpy(),
        })
    return heads


# ---------------------------------------------------------------------------
# Fourier tools
# ---------------------------------------------------------------------------

def compute_fourier_embedding(W_E, p):
    F = dft_matrix(p)
    return F @ W_E[:p, :]


def get_key_frequencies(model, device, p):
    logit_table = model.get_logit_table(device).cpu().numpy()
    result = compute_fourier_component_norms(logit_table, p)
    key_freqs = identify_key_frequencies(result["frequency_norms"], n_top=10)
    unique = set()
    for f in key_freqs:
        f = int(f)
        if f != 0:
            unique.add(min(f, p - f))
    return sorted(unique), key_freqs, result["frequency_norms"]


def compute_frequency_subspaces(E_hat, key_freqs, p):
    subspaces = {}
    for k in key_freqs:
        re = np.real(E_hat[k])
        im = np.imag(E_hat[k])
        basis = np.column_stack([re, im])
        Q, _ = np.linalg.qr(basis)
        subspaces[k] = Q[:, :2]
    return subspaces


def compute_subspace_overlaps(subspaces, key_freqs):
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


def compute_ov_frequency_coupling(W_OV, E_hat, key_freqs, p):
    n = len(key_freqs)
    coupling = np.zeros((n, n))
    for i, ki in enumerate(key_freqs):
        for j, kj in enumerate(key_freqs):
            dirs_i = np.vstack([E_hat[ki], E_hat[p - ki]])
            dirs_j = np.vstack([E_hat[kj], E_hat[p - kj]])
            M = dirs_i @ W_OV @ dirs_j.conj().T
            coupling[i, j] = np.sum(np.abs(M) ** 2)
    diag = np.diag(coupling).copy()
    diag[diag == 0] = 1e-10
    for i in range(n):
        for j in range(n):
            coupling[i, j] /= np.sqrt(diag[i] * diag[j])
    return coupling


# ---------------------------------------------------------------------------
# Input building
# ---------------------------------------------------------------------------

def build_all_inputs(p, device):
    a_vals = torch.arange(p, device=device).repeat_interleave(p)
    b_vals = torch.arange(p, device=device).repeat(p)
    eq_vals = torch.full((p * p,), p, dtype=torch.long, device=device)
    return torch.stack([a_vals, b_vals, eq_vals], dim=1)


# ---------------------------------------------------------------------------
# Analysis 1: Per-head OV frequency coupling
# ---------------------------------------------------------------------------

def per_head_ov_coupling(heads_circuits, E_hat, key_freqs, p):
    """Compute OV frequency coupling per head. Returns list of (n_freq, n_freq) matrices."""
    per_head = []
    for hc in heads_circuits:
        coupling = compute_ov_frequency_coupling(hc["W_OV"], E_hat, key_freqs, p)
        per_head.append(coupling)
    return per_head


# ---------------------------------------------------------------------------
# Analysis 2: Causal ablation
# ---------------------------------------------------------------------------

def run_ablation_experiment(model, p, device, key_freqs):
    """Zero out each head's OV contribution and measure per-frequency logit norm change.

    Returns:
        head_ablation: (n_heads, n_freq) fractional change in frequency norm
        neuron_ablation: (n_freq,) fractional change when ablating non-key neurons
    """
    F_mat = dft_matrix(p)
    inputs = build_all_inputs(p, device)

    # Baseline logit table
    with torch.no_grad():
        baseline_logits = model(inputs)  # (p^2, p)
    baseline_table = baseline_logits.cpu().numpy().reshape(p, p, p)

    # Compute baseline frequency norms per key freq
    baseline_fnorms = _logit_freq_norms(baseline_table, F_mat, key_freqs, p)

    n_heads = model.n_heads
    block = model.blocks[0]
    W = block.attn.in_proj_weight.detach().clone()
    W_O = block.attn.out_proj.weight.detach().clone()
    W_O_bias = block.attn.out_proj.bias.detach().clone() if block.attn.out_proj.bias is not None else None
    d = model.d_model
    d_h = d // n_heads

    head_ablation = np.zeros((n_heads, len(key_freqs)))

    for h_idx in range(n_heads):
        # Zero out head h_idx's output projection columns
        s = slice(h_idx * d_h, (h_idx + 1) * d_h)
        saved_cols = block.attn.out_proj.weight.data[:, s].clone()
        block.attn.out_proj.weight.data[:, s] = 0

        with torch.no_grad():
            ablated_logits = model(inputs)
        ablated_table = ablated_logits.cpu().numpy().reshape(p, p, p)
        ablated_fnorms = _logit_freq_norms(ablated_table, F_mat, key_freqs, p)

        for fi, k in enumerate(key_freqs):
            if baseline_fnorms[fi] > 1e-10:
                head_ablation[h_idx, fi] = (baseline_fnorms[fi] - ablated_fnorms[fi]) / baseline_fnorms[fi]

        # Restore
        block.attn.out_proj.weight.data[:, s] = saved_cols

    # Neuron ablation: zero out MLP neurons NOT tuned to key frequencies
    # First, identify which neurons are key-freq tuned
    neuron_acts = _get_neuron_activations(model, p, device)
    dom_freqs = _neuron_dominant_freq(neuron_acts, key_freqs, p)
    key_set = set(key_freqs)
    non_key_mask = np.array([d not in key_set for d in dom_freqs])

    d_mlp = model.blocks[0].mlp[0].out_features
    mlp_w1 = model.blocks[0].mlp[0].weight.data.clone()
    mlp_b1 = model.blocks[0].mlp[0].bias.data.clone() if model.blocks[0].mlp[0].bias is not None else None

    # Zero out non-key neurons' weights
    saved_w1 = model.blocks[0].mlp[0].weight.data[non_key_mask].clone()
    model.blocks[0].mlp[0].weight.data[non_key_mask] = 0
    if mlp_b1 is not None:
        saved_b1 = model.blocks[0].mlp[0].bias.data[non_key_mask].clone()
        model.blocks[0].mlp[0].bias.data[non_key_mask] = 0

    with torch.no_grad():
        ablated_logits = model(inputs)
    ablated_table = ablated_logits.cpu().numpy().reshape(p, p, p)
    neuron_ablation = _logit_freq_norms(ablated_table, F_mat, key_freqs, p)

    # Restore
    model.blocks[0].mlp[0].weight.data[non_key_mask] = saved_w1
    if mlp_b1 is not None:
        model.blocks[0].mlp[0].bias.data[non_key_mask] = saved_b1

    neuron_frac_change = np.zeros(len(key_freqs))
    for fi in range(len(key_freqs)):
        if baseline_fnorms[fi] > 1e-10:
            neuron_frac_change[fi] = (baseline_fnorms[fi] - neuron_ablation[fi]) / baseline_fnorms[fi]

    return head_ablation, neuron_frac_change, baseline_fnorms


def _logit_freq_norms(logit_table, F_mat, key_freqs, p):
    """Compute Frobenius norm of frequency components across output classes."""
    norms = np.zeros(len(key_freqs))
    for c in range(p):
        L = logit_table[:, :, c]
        L_hat = F_mat @ L @ F_mat.T
        for fi, k in enumerate(key_freqs):
            norms[fi] += (np.abs(L_hat[k, :])**2).sum() + (np.abs(L_hat[p-k, :])**2).sum()
            norms[fi] += (np.abs(L_hat[:, k])**2).sum() + (np.abs(L_hat[:, p-k])**2).sum()
    return np.sqrt(norms)


def _get_neuron_activations(model, p, device):
    inputs = build_all_inputs(p, device)
    with ActivationCache(model) as cache:
        with torch.no_grad():
            _ = model(inputs)
        neuron_acts = cache.neuron_activations[0]
    return neuron_acts[:, 2, :].cpu().numpy().reshape(p, p, -1)


def _neuron_dominant_freq(neuron_acts, key_freqs, p):
    """Return dominant de-mirrored frequency for each neuron."""
    F = dft_matrix(p)
    d_mlp = neuron_acts.shape[2]
    dom_freq = np.zeros(d_mlp, dtype=int)

    for n in range(d_mlp):
        act_n = neuron_acts[:, :, n]
        act_hat = F @ act_n @ F.T
        norms = np.abs(act_hat) ** 2
        freq_energy = np.zeros(p)
        for k in range(p):
            freq_energy[k] = norms[k, :].sum() + norms[:, k].sum() - norms[k, k]
        freq_energy[0] = 0
        grouped = {}
        for k in range(1, p):
            canon = min(k, p - k)
            grouped[canon] = grouped.get(canon, 0) + freq_energy[k]
        if grouped:
            dom_freq[n] = max(grouped, key=grouped.get)
    return dom_freq


# ---------------------------------------------------------------------------
# Analysis 3: Attention pattern analysis & head classification
# ---------------------------------------------------------------------------

def analyze_attention_patterns(model, p, device):
    """Compute per-head attention patterns for all p^2 inputs.

    Uses a temporary monkey-patch on TransformerBlock.forward to pass
    average_attn_weights=False, so we get (batch, n_heads, 3, 3).
    """
    inputs = build_all_inputs(p, device)
    block = model.blocks[0]
    n_heads = model.n_heads

    # Monkey-patch to get per-head attention
    _orig_forward = block.forward

    def _patched_forward(x):
        attn_out, attn_weights = block.attn(
            x, x, x, need_weights=True, average_attn_weights=False
        )
        block._attn_weights = attn_weights.detach()
        x = x + attn_out
        if block.use_layernorm:
            x = block.norm1(x)
        x = x + block.mlp(x)
        if block.use_layernorm:
            x = block.norm2(x)
        return x

    block.forward = _patched_forward

    with torch.no_grad():
        all_attn = []
        chunk_size = 2048
        for i in range(0, len(inputs), chunk_size):
            chunk = inputs[i:i + chunk_size]
            _ = model(chunk)
            attn = block._attn_weights  # (batch, n_heads, 3, 3)
            all_attn.append(attn.cpu())

    block.forward = _orig_forward

    attn_all = torch.cat(all_attn, dim=0)  # (p^2, n_heads, 3, 3)
    if attn_all.dim() == 3:
        attn_all = attn_all.unsqueeze(1)
    return attn_all.numpy()


def classify_heads(model, heads_circuits, E_hat, key_freqs, attn_patterns, p, device):
    """Classify each head as 'trig', 'amplification', or 'inactive'.

    Trig head: high frequency-preserving OV coupling (diag >> offdiag)
    Amplification head: high overall OV norm but low frequency selectivity
    Inactive: negligible OV norm

    Also computes per-head statistics.
    """
    n_heads = model.n_heads
    W_U = model.W_U.weight.detach().cpu().numpy()  # (p, d_model)
    F_mat = dft_matrix(p)

    classifications = []

    for h_idx in range(n_heads):
        W_OV = heads_circuits[h_idx]["W_OV"]

        # OV frequency coupling for this head
        coupling = compute_ov_frequency_coupling(W_OV, E_hat, key_freqs, p)
        n_kf = len(key_freqs)
        diag_mean = np.diag(coupling).mean()
        mask = ~np.eye(n_kf, dtype=bool)
        offdiag_mean = coupling[mask].mean()
        interference = offdiag_mean / diag_mean if diag_mean > 0 else float("inf")

        # OV Frobenius norm
        ov_norm = np.linalg.norm(W_OV, 'fro')

        # Attention pattern at = position
        mean_attn_eq = attn_patterns[:, h_idx, 2, :].mean(axis=0)  # (3,)
        attn_to_ab = mean_attn_eq[0] + mean_attn_eq[1]

        # Full circuit contribution: W_U @ W_OV @ W_E[:p]^T in Fourier basis
        circuit = W_U @ W_OV @ model.W_E.weight.detach().cpu().numpy()[:p, :].T
        C_fourier = F_mat @ circuit @ F_mat.conj().T
        key_energy = sum((np.abs(C_fourier[k, :])**2).sum() + (np.abs(C_fourier[p-k, :])**2).sum()
                         for k in key_freqs)
        total_energy = (np.abs(C_fourier)**2).sum()
        key_frac = key_energy / total_energy if total_energy > 0 else 0

        # Classification
        if ov_norm < 0.1:
            role = "inactive"
        elif interference < 0.5 and key_frac > 0.3:
            role = "trig"
        elif key_frac > 0.2:
            role = "amplification"
        else:
            role = "other"

        classifications.append({
            "head_idx": h_idx,
            "role": role,
            "ov_norm": float(ov_norm),
            "diag_coupling": float(diag_mean),
            "offdiag_coupling": float(offdiag_mean),
            "interference_ratio": float(interference),
            "key_freq_fraction": float(key_frac),
            "attn_to_a": float(mean_attn_eq[0]),
            "attn_to_b": float(mean_attn_eq[1]),
            "attn_to_eq": float(mean_attn_eq[2]),
            "attn_to_ab": float(attn_to_ab),
        })

    return classifications


# ---------------------------------------------------------------------------
# Analysis 4: Neuron polysemanticity comparison
# ---------------------------------------------------------------------------

def neuron_polysemanticity(neuron_acts, key_freqs, p):
    """Returns r2_top1, r2_top2, dom_freq arrays."""
    F = dft_matrix(p)
    d_mlp = neuron_acts.shape[2]
    r2_top1 = np.zeros(d_mlp)
    r2_top2 = np.zeros(d_mlp)
    dom_freq = np.zeros(d_mlp, dtype=int)

    for n in range(d_mlp):
        act_n = neuron_acts[:, :, n]
        act_hat = F @ act_n @ F.T
        norms = np.abs(act_hat) ** 2
        freq_energy = np.zeros(p)
        for k in range(p):
            freq_energy[k] = norms[k, :].sum() + norms[:, k].sum() - norms[k, k]
        freq_energy[0] = 0
        total = freq_energy.sum()
        if total == 0:
            continue
        grouped = {}
        for k in range(1, p):
            canon = min(k, p - k)
            grouped[canon] = grouped.get(canon, 0) + freq_energy[k]
        sorted_g = sorted(grouped.items(), key=lambda x: x[1], reverse=True)
        dom_freq[n] = sorted_g[0][0]
        r2_top1[n] = sorted_g[0][1] / total
        if len(sorted_g) > 1:
            r2_top2[n] = (sorted_g[0][1] + sorted_g[1][1]) / total
        else:
            r2_top2[n] = r2_top1[n]

    return r2_top1, r2_top2, dom_freq


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def fig1_ov_coupling_comparison(all_coupling, all_per_head_coupling, heads_list, all_key_freqs, fig_dir):
    """Side-by-side OV coupling matrices (total and per-head) for each head count."""
    n_models = len(heads_list)

    # Top row: total OV coupling per model
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]

    for ax, nh in zip(axes, heads_list):
        coupling = all_coupling[nh]
        kf = all_key_freqs[nh]
        im = ax.imshow(coupling, cmap="magma", interpolation="nearest")
        ax.set_xticks(range(len(kf)))
        ax.set_xticklabels([str(k) for k in kf], fontsize=8)
        ax.set_yticks(range(len(kf)))
        ax.set_yticklabels([str(k) for k in kf], fontsize=8)

        diag_mean = np.diag(coupling).mean()
        mask = ~np.eye(len(kf), dtype=bool)
        offdiag_mean = coupling[mask].mean()
        ir = offdiag_mean / diag_mean if diag_mean > 0 else 0

        ax.set_title(f"h={nh} (total OV)\nIR={ir:.3f}", fontsize=10, fontweight="bold")
        for i in range(len(kf)):
            for j in range(len(kf)):
                color = "white" if coupling[i, j] < coupling.max() * 0.7 else "black"
                ax.text(j, i, f"{coupling[i,j]:.2f}", ha="center", va="center",
                        fontsize=7, color=color)
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("OV Circuit Frequency Coupling (Total)\nDiag = preservation, off-diag = interference",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "circuit_ov_coupling_total.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: circuit_ov_coupling_total.png")

    # Per-head coupling for multi-head models
    for nh in heads_list:
        if nh <= 1:
            continue
        per_head = all_per_head_coupling[nh]
        n_h = len(per_head)
        kf = all_key_freqs[nh]
        cols = min(n_h, 4)
        rows = (n_h + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        if rows == 1 and cols == 1:
            axes = np.array([[axes]])
        elif rows == 1:
            axes = axes.reshape(1, -1)
        elif cols == 1:
            axes = axes.reshape(-1, 1)

        for h_idx in range(n_h):
            r, c = h_idx // cols, h_idx % cols
            ax = axes[r, c]
            coup = per_head[h_idx]
            im = ax.imshow(coup, cmap="magma", interpolation="nearest")
            ax.set_xticks(range(len(kf)))
            ax.set_xticklabels([str(k) for k in kf], fontsize=7)
            ax.set_yticks(range(len(kf)))
            ax.set_yticklabels([str(k) for k in kf], fontsize=7)

            diag_m = np.diag(coup).mean()
            mask_h = ~np.eye(len(kf), dtype=bool)
            offdiag_m = coup[mask_h].mean()
            ir_h = offdiag_m / diag_m if diag_m > 0 else 0
            ax.set_title(f"Head {h_idx} (IR={ir_h:.2f})", fontsize=9, fontweight="bold")

            for i in range(len(kf)):
                for j in range(len(kf)):
                    color = "white" if coup[i, j] < coup.max() * 0.7 else "black"
                    ax.text(j, i, f"{coup[i,j]:.1f}", ha="center", va="center",
                            fontsize=6, color=color)

        # Hide unused axes
        for h_idx in range(n_h, rows * cols):
            r, c = h_idx // cols, h_idx % cols
            axes[r, c].axis("off")

        fig.suptitle(f"Per-Head OV Coupling (h={nh})", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_dir / f"circuit_ov_coupling_h{nh}_perhead.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: circuit_ov_coupling_h{nh}_perhead.png")


def fig2_ablation_heatmaps(all_head_ablation, all_neuron_ablation, heads_list, all_key_freqs, fig_dir):
    """Ablation sensitivity heatmaps: (component zeroed) x (frequency affected)."""
    n_models = len(heads_list)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))
    if n_models == 1:
        axes = [axes]

    for ax, nh in zip(axes, heads_list):
        ha = all_head_ablation[nh]  # (n_heads, n_freq)
        na = all_neuron_ablation[nh]  # (n_freq,)
        kf = all_key_freqs[nh]

        # Stack: rows = heads + "non-key neurons"
        data = np.vstack([ha, na.reshape(1, -1)])
        row_labels = [f"Head {i}" for i in range(ha.shape[0])] + ["Non-key\nneurons"]

        im = ax.imshow(data, cmap="RdYlBu_r", aspect="auto", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(len(kf)))
        ax.set_xticklabels([str(k) for k in kf], fontsize=9)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("Frequency", fontsize=10)
        ax.set_title(f"h={nh}: Ablation Impact\n(frac. norm change)", fontsize=10, fontweight="bold")

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                color = "white" if data[i, j] > 0.5 else "black"
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label="Frac. norm change")

    fig.suptitle("Causal Ablation: Component Impact on Frequency Norms\n"
                 "(positive = component contributes to that frequency)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "circuit_ablation_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: circuit_ablation_heatmaps.png")


def fig3_subspace_angles(all_overlaps, heads_list, all_key_freqs, fig_dir):
    """Frequency subspace principal angles as function of n_heads."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: max off-diagonal overlap per model
    ax = axes[0]
    colors = plt.get_cmap("tab10")
    for i, nh in enumerate(heads_list):
        ov = all_overlaps[nh]
        kf = all_key_freqs[nh]
        n_kf = len(kf)
        mask = ~np.eye(n_kf, dtype=bool)
        offdiag = ov[mask]
        ax.bar(i, np.max(offdiag), 0.6, color=colors(i), edgecolor="black",
               linewidth=0.8, label=f"h={nh}", alpha=0.85)
        ax.text(i, np.max(offdiag) + 0.01, f"{np.max(offdiag):.3f}",
                ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(range(len(heads_list)))
    ax.set_xticklabels([f"h={nh}" for nh in heads_list], fontsize=11)
    ax.set_ylabel("Max cos(principal angle)", fontsize=11)
    ax.set_title("Max Off-Diagonal Subspace Overlap\n(lower = better separated frequencies)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    # Right: full overlap matrices side-by-side
    ax2 = axes[1]
    # Show as grouped bar chart: mean and max off-diagonal overlap
    means = []
    maxes = []
    for nh in heads_list:
        ov = all_overlaps[nh]
        mask = ~np.eye(len(all_key_freqs[nh]), dtype=bool)
        means.append(ov[mask].mean())
        maxes.append(ov[mask].max())

    x = np.arange(len(heads_list))
    width = 0.35
    ax2.bar(x - width/2, means, width, label="Mean overlap", color="#3498db",
            edgecolor="black", linewidth=0.8)
    ax2.bar(x + width/2, maxes, width, label="Max overlap", color="#e74c3c",
            edgecolor="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"h={nh}" for nh in heads_list], fontsize=11)
    ax2.set_ylabel("cos(principal angle)", fontsize=11)
    ax2.set_title("Frequency Subspace Overlap Statistics",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(0, 1.0)

    fig.tight_layout()
    fig.savefig(fig_dir / "circuit_subspace_angles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: circuit_subspace_angles.png")


def fig4_neuron_polysemanticity(all_poly_data, heads_list, all_key_freqs, fig_dir):
    """Neuron polysemanticity distribution per head count."""
    n_models = len(heads_list)
    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 9))
    if n_models == 1:
        axes = axes.reshape(-1, 1)

    for col, nh in enumerate(heads_list):
        r2_top1, r2_top2, dom_freq = all_poly_data[nh]
        kf = all_key_freqs[nh]
        key_set = set(kf)
        is_key = np.array([int(d) in key_set for d in dom_freq])
        active = r2_top1 > 0.05

        # Top row: scatter R2_top1 vs R2_top2
        ax = axes[0, col]
        ax.scatter(r2_top1[active & is_key], r2_top2[active & is_key],
                   alpha=0.5, s=15, c="#2ca02c", label=f"Key freq ({sum(active & is_key)})")
        ax.scatter(r2_top1[active & ~is_key], r2_top2[active & ~is_key],
                   alpha=0.3, s=15, c="#7f7f7f", label=f"Other ({sum(active & ~is_key)})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax.set_xlabel("R^2 top-1", fontsize=9)
        ax.set_ylabel("R^2 top-2", fontsize=9)
        ax.set_title(f"h={nh}: Neuron Polysemanticity", fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        # Bottom row: histogram of gap
        ax2 = axes[1, col]
        gap = r2_top2[active & is_key] - r2_top1[active & is_key]
        ax2.hist(gap, bins=25, color="#3498db", edgecolor="black", alpha=0.8)
        ax2.axvline(0.1, color="red", linestyle="--", alpha=0.7, label="Poly threshold")
        n_poly = int((gap > 0.1).sum())
        n_mono = int((gap <= 0.1).sum())
        ax2.text(0.95, 0.95, f"Mono: {n_mono}\nPoly: {n_poly}",
                 transform=ax2.transAxes, fontsize=9, va="top", ha="right",
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))
        ax2.set_xlabel("R^2_top2 - R^2_top1", fontsize=9)
        ax2.set_ylabel("Count", fontsize=9)
        ax2.set_title(f"h={nh}: Multi-Freq Gap", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Neuron Frequency Selectivity Comparison",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_dir / "circuit_neuron_polysemanticity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: circuit_neuron_polysemanticity.png")


def fig5_head_classification(all_classifications, heads_list, fig_dir):
    """Head classification table for multi-head models."""
    multi_head_models = [nh for nh in heads_list if nh > 1]
    if not multi_head_models:
        print("  Skipping head classification (no multi-head models)")
        return

    for nh in multi_head_models:
        classifications = all_classifications[nh]
        n_h = len(classifications)

        fig, ax = plt.subplots(figsize=(12, max(3, n_h * 0.5 + 2)))
        ax.axis("off")

        col_labels = ["Head", "Role", "OV norm", "Diag coupling", "Off-diag",
                      "IR", "Key-freq %", "Attn a", "Attn b", "Attn ="]

        table_data = []
        cell_colors = []
        role_colors = {"trig": "#d4edda", "amplification": "#fff3cd",
                       "inactive": "#f8d7da", "other": "#e2e3e5"}

        for c in classifications:
            row = [
                str(c["head_idx"]),
                c["role"],
                f"{c['ov_norm']:.3f}",
                f"{c['diag_coupling']:.3f}",
                f"{c['offdiag_coupling']:.3f}",
                f"{c['interference_ratio']:.3f}",
                f"{c['key_freq_fraction']:.1%}",
                f"{c['attn_to_a']:.3f}",
                f"{c['attn_to_b']:.3f}",
                f"{c['attn_to_eq']:.3f}",
            ]
            table_data.append(row)
            color = role_colors.get(c["role"], "#ffffff")
            cell_colors.append([color] * len(col_labels))

        table = ax.table(cellText=table_data, colLabels=col_labels,
                         cellColours=cell_colors, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.5)
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#343a40")
            table[0, j].set_text_props(color="white", fontweight="bold")

        role_counts = {}
        for c in classifications:
            role_counts[c["role"]] = role_counts.get(c["role"], 0) + 1
        role_str = ", ".join(f"{r}: {n}" for r, n in sorted(role_counts.items()))

        ax.set_title(f"Head Classification (h={nh}): {role_str}\n"
                     f"(green=trig, yellow=amplification, red=inactive, gray=other)",
                     fontsize=11, fontweight="bold", pad=20)

        fig.tight_layout()
        fig.savefig(fig_dir / f"circuit_head_classification_h{nh}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: circuit_head_classification_h{nh}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Comparative circuit analysis across head counts")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--output-dir", type=str, default="results/circuit_comparison")
    parser.add_argument("--heads", type=int, nargs="+", default=[1, 4, 16])
    args = parser.parse_args()

    logger = setup_logging()
    results_root = Path(args.results_dir)
    fig_dir = Path(args.output_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    heads_list = args.heads

    # -----------------------------------------------------------------------
    # Load models
    # -----------------------------------------------------------------------
    models = {}
    for nh in heads_list:
        run_dir = results_root / _run_id_for(nh)
        if not (run_dir / "model.pt").exists():
            logger.warning(f"Missing model for h={nh}: {run_dir}")
            continue
        model, cfg = load_model(run_dir, device)
        models[nh] = model
        logger.info(f"Loaded h={nh}: {model.count_parameters():,} params")

    if not models:
        logger.error("No models loaded!")
        return

    heads_list = [nh for nh in heads_list if nh in models]

    # -----------------------------------------------------------------------
    # Per-model analysis
    # -----------------------------------------------------------------------
    all_key_freqs = {}
    all_E_hat = {}
    all_heads_circuits = {}
    all_coupling = {}
    all_per_head_coupling = {}
    all_subspaces = {}
    all_overlaps = {}
    all_head_ablation = {}
    all_neuron_ablation = {}
    all_baseline_fnorms = {}
    all_attn_patterns = {}
    all_classifications = {}
    all_poly_data = {}

    for nh in heads_list:
        model = models[nh]
        logger.info(f"\n=== Analyzing h={nh} ===")

        # Key frequencies
        key_freqs, _, freq_norms = get_key_frequencies(model, device, P)
        all_key_freqs[nh] = key_freqs
        logger.info(f"  Key frequencies: {key_freqs}")

        # Fourier embedding
        W_E = model.W_E.weight.detach().cpu().numpy()
        E_hat = compute_fourier_embedding(W_E, P)
        all_E_hat[nh] = E_hat

        # Per-head circuits
        heads_circuits = extract_per_head_circuits(model)
        all_heads_circuits[nh] = heads_circuits

        # Total OV coupling
        W_OV_total = sum(h["W_OV"] for h in heads_circuits)
        coupling = compute_ov_frequency_coupling(W_OV_total, E_hat, key_freqs, P)
        all_coupling[nh] = coupling

        # Per-head OV coupling
        ph_coupling = per_head_ov_coupling(heads_circuits, E_hat, key_freqs, P)
        all_per_head_coupling[nh] = ph_coupling

        n_kf = len(key_freqs)
        diag_mean = np.diag(coupling).mean()
        mask = ~np.eye(n_kf, dtype=bool)
        offdiag_mean = coupling[mask].mean()
        ir = offdiag_mean / diag_mean if diag_mean > 0 else 0
        logger.info(f"  Total interference ratio: {ir:.4f}")

        # Frequency subspaces & overlaps
        subspaces = compute_frequency_subspaces(E_hat, key_freqs, P)
        all_subspaces[nh] = subspaces
        overlaps = compute_subspace_overlaps(subspaces, key_freqs)
        all_overlaps[nh] = overlaps
        max_offdiag = overlaps[~np.eye(n_kf, dtype=bool)].max()
        logger.info(f"  Max subspace overlap: {max_offdiag:.4f}")

        # Causal ablation
        logger.info("  Running causal ablation...")
        ha, na, bf = run_ablation_experiment(model, P, device, key_freqs)
        all_head_ablation[nh] = ha
        all_neuron_ablation[nh] = na
        all_baseline_fnorms[nh] = bf
        logger.info(f"  Head ablation shape: {ha.shape}")

        # Attention patterns
        logger.info("  Analyzing attention patterns...")
        attn_patterns = analyze_attention_patterns(model, P, device)
        all_attn_patterns[nh] = attn_patterns

        # Head classification
        classifications = classify_heads(model, heads_circuits, E_hat, key_freqs,
                                         attn_patterns, P, device)
        all_classifications[nh] = classifications
        role_counts = {}
        for c in classifications:
            role_counts[c["role"]] = role_counts.get(c["role"], 0) + 1
        logger.info(f"  Head roles: {role_counts}")

        # Neuron polysemanticity
        logger.info("  Computing neuron polysemanticity...")
        neuron_acts = _get_neuron_activations(model, P, device)
        r2_1, r2_2, dom = neuron_polysemanticity(neuron_acts, key_freqs, P)
        all_poly_data[nh] = (r2_1, r2_2, dom)
        key_set = set(key_freqs)
        active = r2_1 > 0.05
        is_key = np.array([int(d) in key_set for d in dom])
        gap = r2_2[active & is_key] - r2_1[active & is_key]
        n_poly = int((gap > 0.1).sum())
        n_mono = int((gap <= 0.1).sum())
        logger.info(f"  Active neurons: {sum(active)}, Key-freq: {sum(active & is_key)}, "
                     f"Mono: {n_mono}, Poly: {n_poly}")

    # -----------------------------------------------------------------------
    # Generate figures
    # -----------------------------------------------------------------------
    logger.info("\n=== Generating figures ===")

    logger.info("1/5: OV coupling comparison")
    fig1_ov_coupling_comparison(all_coupling, all_per_head_coupling, heads_list, all_key_freqs, fig_dir)

    logger.info("2/5: Ablation heatmaps")
    fig2_ablation_heatmaps(all_head_ablation, all_neuron_ablation, heads_list, all_key_freqs, fig_dir)

    logger.info("3/5: Subspace angles")
    fig3_subspace_angles(all_overlaps, heads_list, all_key_freqs, fig_dir)

    logger.info("4/5: Neuron polysemanticity")
    fig4_neuron_polysemanticity(all_poly_data, heads_list, all_key_freqs, fig_dir)

    logger.info("5/5: Head classification")
    fig5_head_classification(all_classifications, heads_list, fig_dir)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 100)
    print("COMPARATIVE CIRCUIT ANALYSIS -- SUMMARY")
    print("=" * 100)

    print(f"\n{'n_heads':>7} | {'Key Freqs':>12} | {'IR (total)':>10} | "
          f"{'Max Overlap':>11} | {'Active':>6} | {'Mono':>5} | {'Poly':>5} | "
          f"{'Head Roles':>30}")
    print("-" * 100)

    for nh in heads_list:
        kf = all_key_freqs[nh]
        n_kf = len(kf)
        coup = all_coupling[nh]
        diag_m = np.diag(coup).mean()
        mask = ~np.eye(n_kf, dtype=bool)
        offdiag_m = coup[mask].mean()
        ir = offdiag_m / diag_m if diag_m > 0 else 0

        ov = all_overlaps[nh]
        max_ov = ov[~np.eye(n_kf, dtype=bool)].max()

        r2_1, r2_2, dom = all_poly_data[nh]
        key_set = set(kf)
        active = r2_1 > 0.05
        is_key = np.array([int(d) in key_set for d in dom])
        gap = r2_2[active & is_key] - r2_1[active & is_key]
        n_poly = int((gap > 0.1).sum())
        n_mono = int((gap <= 0.1).sum())

        roles = {}
        for c in all_classifications[nh]:
            roles[c["role"]] = roles.get(c["role"], 0) + 1
        roles_str = ", ".join(f"{r}:{n}" for r, n in sorted(roles.items()))

        print(f"{nh:>7} | {str(kf):>12} | {ir:>10.4f} | "
              f"{max_ov:>11.4f} | {sum(active):>6} | {n_mono:>5} | {n_poly:>5} | "
              f"{roles_str:>30}")

    print("=" * 100)

    # Mechanistic narrative
    print("\n--- MECHANISTIC NARRATIVE ---")
    if len(heads_list) >= 2 and 1 in heads_list:
        ir_h1 = all_coupling[1]
        n_kf = len(all_key_freqs[1])
        diag_h1 = np.diag(ir_h1).mean()
        mask_h1 = ~np.eye(n_kf, dtype=bool)
        offdiag_h1 = ir_h1[mask_h1].mean()
        ir1 = offdiag_h1 / diag_h1 if diag_h1 > 0 else 0

        print(f"\nh=1 has interference ratio {ir1:.3f} (HIGH cross-frequency coupling)")
        print("Yet h=1 groks 2-3x FASTER than multi-head models.")
        print("\nPossible explanations from the circuit analysis:")
        print("1. High interference acts as IMPLICIT REGULARIZATION")
        print("   - Forces compact representations, preventing overfitting to spurious patterns")
        print("2. Single-head bottleneck forces COORDINATED learning")
        print("   - All frequencies must pass through one OV circuit -> less redundancy")
        print("3. Superposition is efficient: 5 freq pairs in d_head=128 >> required dim")
        print("   - The interference is tolerable because d_head is large relative to need")

    # -----------------------------------------------------------------------
    # Save JSON metrics
    # -----------------------------------------------------------------------
    metrics_out = {
        "experiment": "circuit_comparison",
        "heads_analyzed": heads_list,
        "p": P,
        "per_model": {},
    }

    for nh in heads_list:
        kf = all_key_freqs[nh]
        coup = all_coupling[nh]
        n_kf = len(kf)
        diag_m = np.diag(coup).mean()
        mask = ~np.eye(n_kf, dtype=bool)
        offdiag_m = coup[mask].mean()

        r2_1, r2_2, dom = all_poly_data[nh]
        key_set = set(kf)
        active = r2_1 > 0.05
        is_key = np.array([int(d) in key_set for d in dom])
        gap = r2_2[active & is_key] - r2_1[active & is_key]

        metrics_out["per_model"][str(nh)] = {
            "key_frequencies": kf,
            "interference_ratio": float(offdiag_m / diag_m if diag_m > 0 else 0),
            "diag_coupling_mean": float(diag_m),
            "offdiag_coupling_mean": float(offdiag_m),
            "max_subspace_overlap": float(all_overlaps[nh][~np.eye(n_kf, dtype=bool)].max()),
            "mean_subspace_overlap": float(all_overlaps[nh][~np.eye(n_kf, dtype=bool)].mean()),
            "active_neurons": int(sum(active)),
            "key_freq_neurons": int(sum(active & is_key)),
            "monosemantic_neurons": int((gap <= 0.1).sum()),
            "polysemantic_neurons": int((gap > 0.1).sum()),
            "head_classifications": all_classifications[nh],
            "head_ablation": all_head_ablation[nh].tolist(),
            "neuron_ablation_frac_change": all_neuron_ablation[nh].tolist(),
            "per_head_interference_ratios": [],
        }

        for h_idx, ph_coup in enumerate(all_per_head_coupling[nh]):
            d_m = np.diag(ph_coup).mean()
            m = ~np.eye(n_kf, dtype=bool)
            od_m = ph_coup[m].mean()
            metrics_out["per_model"][str(nh)]["per_head_interference_ratios"].append(
                float(od_m / d_m if d_m > 0 else 0)
            )

    out_path = fig_dir / "circuit_comparison_metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics_out, f, indent=2, default=str)
    print(f"\nSaved metrics: {out_path}")
    logger.info(f"All outputs saved to {fig_dir}")


if __name__ == "__main__":
    main()
