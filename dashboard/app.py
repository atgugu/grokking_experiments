#!/usr/bin/env python
"""Streamlit interactive dashboard for exploring grokking experiment results."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="Grokking Dashboard", layout="wide")


@st.cache_data
def load_run_result(run_dir: str) -> dict | None:
    """Load a single run's metrics and config."""
    run_path = Path(run_dir)
    config_path = run_path / "config.json"
    metrics_path = run_path / "metrics.json"
    if not config_path.exists() or not metrics_path.exists():
        return None
    with open(config_path) as f:
        config = json.load(f)
    with open(metrics_path) as f:
        metrics = json.load(f)
    return {"config": config, "metrics": metrics}


@st.cache_data
def load_fourier_snapshots(run_dir: str) -> dict | None:
    """Load Fourier snapshots."""
    snap_path = Path(run_dir) / "fourier_snapshots.npz"
    if not snap_path.exists():
        return None
    data = dict(np.load(snap_path))
    return data


@st.cache_data(ttl=30)
def find_runs(results_dir: str) -> list[str]:
    """Find all completed runs in results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        return []
    return sorted([
        d.name for d in results_path.iterdir()
        if d.is_dir() and (d / "metrics.json").exists()
    ])


def main():
    st.title("Grokking: Mechanistic Interpretability")
    st.markdown("*Replicating Nanda et al. (ICLR 2023) — Progress measures for grokking*")

    # Sidebar
    st.sidebar.header("Controls")
    results_dir = st.sidebar.text_input("Results Directory", "results")
    runs = find_runs(results_dir)

    if not runs:
        st.error(f"No runs found in `{results_dir}/`. Run training first.")
        return

    selected_run = st.sidebar.selectbox("Select Run", runs)
    run_dir = str(Path(results_dir) / selected_run)

    run_data = load_run_result(run_dir)
    if run_data is None:
        st.error(f"Could not load run: {selected_run}")
        return

    config = run_data["config"]
    metrics = run_data["metrics"]
    history = metrics.get("history", {})

    # Display config
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Config**")
    st.sidebar.markdown(f"- p = {config.get('p', '?')}")
    st.sidebar.markdown(f"- d_model = {config.get('d_model', '?')}")
    st.sidebar.markdown(f"- n_heads = {config.get('n_heads', '?')}")
    st.sidebar.markdown(f"- d_mlp = {config.get('d_mlp', '?')}")
    st.sidebar.markdown(f"- weight_decay = {config.get('weight_decay', '?')}")
    st.sidebar.markdown(f"- Params = {metrics.get('n_params', '?'):,}")

    # Epoch slider for fourier snapshots
    fourier_snaps = load_fourier_snapshots(run_dir)

    # --- Tabs ---
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Training Curves", "Fourier Analysis", "Progress Measures",
        "Mechanistic Interp", "Weight Matrices",
        "Neuron Analysis", "Logit Tables", "Trajectories",
    ])

    # Tab 1: Training Curves
    with tab1:
        eval_epochs = history.get("eval_epochs", [])

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Loss (log scale)", "Accuracy"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        if history.get("train_loss"):
            fig.add_trace(go.Scatter(
                x=eval_epochs[:len(history["train_loss"])],
                y=history["train_loss"],
                name="Train Loss", line=dict(color="#636EFA"),
            ), row=1, col=1)
        if history.get("test_loss"):
            fig.add_trace(go.Scatter(
                x=eval_epochs[:len(history["test_loss"])],
                y=history["test_loss"],
                name="Test Loss", line=dict(color="#EF553B"),
            ), row=1, col=1)

        if history.get("train_acc"):
            fig.add_trace(go.Scatter(
                x=eval_epochs[:len(history["train_acc"])],
                y=history["train_acc"],
                name="Train Acc", line=dict(color="#636EFA"),
            ), row=1, col=2)
        if history.get("test_acc"):
            fig.add_trace(go.Scatter(
                x=eval_epochs[:len(history["test_acc"])],
                y=history["test_acc"],
                name="Test Acc", line=dict(color="#EF553B"),
            ), row=1, col=2)

        fig.update_yaxes(type="log", title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", range=[0, 1.05], row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True, key=f"training_curves_{selected_run}")

        # Weight norm
        if history.get("weight_norm"):
            fig_wn = go.Figure()
            fig_wn.add_trace(go.Scatter(
                x=eval_epochs[:len(history["weight_norm"])],
                y=history["weight_norm"],
                name="Weight Norm", line=dict(color="#00CC96"),
            ))
            fig_wn.update_layout(
                title="Weight Norm",
                xaxis_title="Epoch", yaxis_title="L2 Norm",
                height=300,
            )
            st.plotly_chart(fig_wn, use_container_width=True, key=f"weight_norm_{selected_run}")

    # Tab 2: Fourier Analysis
    with tab2:
        if fourier_snaps is not None:
            freq_norms_all = fourier_snaps["frequency_norms"]
            f_epochs = fourier_snaps.get("fourier_epochs", np.arange(len(freq_norms_all)))

            epoch_idx = st.slider(
                "Snapshot Index", 0, len(f_epochs) - 1, len(f_epochs) - 1,
                key=f"fourier_slider_{selected_run}",
            )
            st.markdown(f"**Epoch: {int(f_epochs[epoch_idx])}**")

            freq_norms = freq_norms_all[epoch_idx]
            p = config.get("p", 113)

            # Frequency spectrum bar chart
            fig_spec = go.Figure()
            fig_spec.add_trace(go.Bar(
                x=list(range(p)), y=freq_norms,
                marker_color="#636EFA",
            ))
            fig_spec.update_layout(
                title="Frequency Spectrum",
                xaxis_title="Frequency k",
                yaxis_title="Energy",
                height=350,
            )
            st.plotly_chart(fig_spec, use_container_width=True, key=f"freq_spectrum_{selected_run}")

            # Key frequency evolution
            key_freqs = fourier_snaps.get("key_frequencies", np.array([]))
            if len(key_freqs) > 0:
                final_keys = key_freqs[-1]
                fig_evo = go.Figure()
                for k in final_keys:
                    fig_evo.add_trace(go.Scatter(
                        x=f_epochs.tolist(),
                        y=freq_norms_all[:, int(k)].tolist(),
                        name=f"k={k}",
                        mode="lines",
                    ))
                fig_evo.update_layout(
                    title="Key Frequency Evolution",
                    xaxis_title="Epoch",
                    yaxis_title="Frequency Energy",
                    height=350,
                )
                st.plotly_chart(fig_evo, use_container_width=True, key=f"freq_evolution_{selected_run}")
        else:
            st.info("No Fourier snapshots found for this run.")

    # Tab 3: Progress Measures
    with tab3:
        fourier_epochs = history.get("fourier_epochs", [])

        col1, col2 = st.columns(2)
        with col1:
            if history.get("gini"):
                fig_gini = go.Figure()
                fig_gini.add_trace(go.Scatter(
                    x=fourier_epochs[:len(history["gini"])],
                    y=history["gini"],
                    name="Gini", line=dict(color="#AB63FA", width=2),
                ))
                fig_gini.update_layout(
                    title="Fourier Gini Coefficient",
                    xaxis_title="Epoch", yaxis_title="Gini",
                    yaxis=dict(range=[0, 1.05]),
                    height=350,
                )
                st.plotly_chart(fig_gini, use_container_width=True, key=f"gini_{selected_run}")

        with col2:
            st.markdown("**Final Metrics**")
            st.metric("Final Train Acc", f"{metrics.get('final_train_acc', 0):.4f}")
            st.metric("Final Test Acc", f"{metrics.get('final_test_acc', 0):.4f}")
            st.metric("Final Gini", f"{metrics.get('final_gini', 0):.4f}")
            st.metric("Final Weight Norm", f"{metrics.get('final_weight_norm', 0):.1f}")
            key_freqs_final = metrics.get("final_key_frequencies", [])
            st.metric("Key Frequencies", str(key_freqs_final))

    # Tab 4: Mechanistic Interp
    with tab4:
        st.markdown("### Mechanistic Interpretability")
        st.markdown("""
        Load analysis results from `scripts/analyze_run.py` to view:
        - Neuron frequency clustering
        - Embedding Fourier spectra
        - Restricted vs excluded loss
        """)

        analysis_path = Path(run_dir) / "analysis.json"
        if analysis_path.exists():
            with open(analysis_path) as f:
                analysis = json.load(f)

            epochs_analyzed = sorted(analysis.keys(), key=int)
            sel_epoch = st.selectbox("Analyzed Epoch", epochs_analyzed, key=f"analysis_epoch_{selected_run}")

            if sel_epoch in analysis:
                a = analysis[sel_epoch]
                col1, col2, col3 = st.columns(3)
                col1.metric("Restricted Loss", f"{a.get('restricted_loss', 0):.4f}")
                col2.metric("Excluded Loss", f"{a.get('excluded_loss', 0):.4f}")
                col3.metric("Gini", f"{a.get('gini', 0):.4f}")

                st.markdown(f"**Key Frequencies:** {a.get('key_frequencies', [])}")

                clusters = a.get("neuron_clusters", {})
                if clusters:
                    st.markdown("**Neuron Clusters:**")
                    for freq, neurons in sorted(clusters.items(), key=lambda x: int(x[0])):
                        st.markdown(f"- k={freq}: {len(neurons)} neurons")
        else:
            st.info("Run `python scripts/analyze_run.py --run-dir <path>` to generate analysis.")

    fig_dir = Path(run_dir) / "figures"

    # Tab 5: Weight Matrices
    with tab5:
        st.markdown("### Weight Matrix Visualizations")
        st.markdown("Generate weight heatmaps using `scripts/generate_figures.py`.")

        if fig_dir.exists():
            for img_name in ["weight_heatmaps.png", "weight_evolution.png",
                             "embedding_circles.png", "attention_patterns.png"]:
                img_path = fig_dir / img_name
                if img_path.exists():
                    st.image(img_path.read_bytes(), caption=img_name.replace(".png", "").replace("_", " ").title())
        else:
            st.info("Run `python scripts/generate_figures.py --run-dir <path>` to generate figures.")

    # Tab 6: Neuron Analysis
    with tab6:
        st.markdown("### Neuron Analysis")
        st.markdown("Per-neuron activation patterns, logit map, and frequency spectrum.")

        if fig_dir.exists():
            for img_name, caption in [
                ("neuron_activation_grids.png", "Neuron Activation Grids"),
                ("neuron_logit_map.png", "Neuron-Logit Map W_L"),
                ("neuron_freq_spectrum_heatmap.png", "Neuron Frequency Spectrum"),
            ]:
                img_path = fig_dir / img_name
                if img_path.exists():
                    st.image(img_path.read_bytes(), caption=caption)
        else:
            st.info("Run `python scripts/generate_figures.py --run-dir <path>` to generate figures.")

    # Tab 7: Logit Tables
    with tab7:
        st.markdown("### Logit Table Visualizations")
        st.markdown("Full vs restricted logit comparison, 3D surface, and per-sample loss.")

        if fig_dir.exists():
            for img_name, caption in [
                ("logit_heatmap_comparison.png", "Logit Heatmap: Full vs Restricted"),
                ("correct_logit_surface.png", "Correct-Class Logit 3D Surface"),
                ("per_sample_loss_heatmap.png", "Per-Sample Loss Heatmap"),
            ]:
                img_path = fig_dir / img_name
                if img_path.exists():
                    st.image(img_path.read_bytes(), caption=caption)
        else:
            st.info("Run `python scripts/generate_figures.py --run-dir <path>` to generate figures.")

    # Tab 8: Trajectories
    with tab8:
        st.markdown("### Training Trajectories")
        st.markdown("Embedding PCA evolution, weight trajectory, and Fourier spectrum strip.")

        if fig_dir.exists():
            for img_name, caption in [
                ("embedding_pca_evolution.png", "Embedding PCA Evolution"),
                ("weight_trajectory_pca.png", "Weight Trajectory PCA"),
                ("fourier_spectrum_strip.png", "Fourier Spectrum Strip"),
            ]:
                img_path = fig_dir / img_name
                if img_path.exists():
                    st.image(img_path.read_bytes(), caption=caption)

            # Animation (gif)
            gif_path = fig_dir / "grokking_animation.gif"
            if gif_path.exists():
                st.image(gif_path.read_bytes(), caption="Grokking Animation")
        else:
            st.info("Run `python scripts/generate_figures.py --run-dir <path>` to generate figures.")


if __name__ == "__main__":
    main()
