"""Training loop for grokking experiments.

Full-batch gradient descent on modular arithmetic, following Nanda et al. (2023).
Each epoch = one gradient step on the entire training set.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..analysis.fourier import (
    compute_fourier_component_norms,
    compute_gini_coefficient,
    identify_key_frequencies,
)


class Trainer:
    """Trains a GrokkingTransformer and records metrics + Fourier snapshots."""

    def __init__(
        self,
        model: nn.Module,
        env,
        config: dict,
        device: torch.device,
        output_dir: Path | None = None,
    ):
        self.model = model.to(device)
        self.env = env
        self.config = config
        self.device = device
        self.output_dir = Path(output_dir) if output_dir is not None else None

        # Setup optimizer
        lr = config.get("lr", 1e-3)
        wd = config.get("weight_decay", 1.0)
        betas = (config.get("beta1", 0.9), config.get("beta2", 0.98))
        optimizer_type = config.get("optimizer", "adamw")
        if optimizer_type == "adamw":
            self.optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=wd, betas=betas,
            )
        else:
            self.optimizer = torch.optim.Adam(
                model.parameters(), lr=lr, weight_decay=wd, betas=betas,
            )

        self.criterion = nn.CrossEntropyLoss()

        # Pre-load train/test data to GPU
        train_data = env.get_train_dataset()
        test_data = env.get_test_dataset()

        self.train_inputs = train_data.inputs.to(device)
        self.train_targets = train_data.target_tensor.to(device)
        self.test_inputs = test_data.inputs.to(device)
        self.test_targets = test_data.target_tensor.to(device)

    def train(self, quiet: bool = False) -> dict:
        """Run full training loop.

        Returns:
            Dictionary with training history, Fourier snapshots, and final metrics.
        """
        max_epochs = self.config.get("max_epochs", 40000)
        eval_interval = self.config.get("eval_interval", 100)
        fourier_interval = self.config.get("fourier_interval", 500)
        checkpoint_interval = self.config.get("checkpoint_interval", 5000)
        p = self.config.get("p", 113)
        n_top = self.config.get("n_top_frequencies", 5)

        history = {
            "train_loss": [],
            "test_loss": [],
            "train_acc": [],
            "test_acc": [],
            "weight_norm": [],
            "eval_epochs": [],
            "gini": [],
            "frequency_norms_snapshots": [],  # list of (p,) arrays at fourier_interval
            "key_frequencies_snapshots": [],  # list of top-k freq arrays
            "fourier_epochs": [],
        }
        checkpoints = {}  # epoch -> model state dict path

        self.model.train()
        pbar = tqdm(range(max_epochs), disable=quiet, desc="Training")

        for epoch in pbar:
            # Full-batch forward + backward
            self.optimizer.zero_grad()
            logits = self.model(self.train_inputs)
            loss = self.criterion(logits, self.train_targets)
            loss.backward()
            self.optimizer.step()

            loss_val = loss.item()

            # Periodic evaluation
            if epoch % eval_interval == 0 or epoch == max_epochs - 1:
                eval_result = self._evaluate()
                history["train_loss"].append(loss_val)
                history["test_loss"].append(eval_result["test_loss"])
                history["train_acc"].append(eval_result["train_acc"])
                history["test_acc"].append(eval_result["test_acc"])
                history["weight_norm"].append(eval_result["weight_norm"])
                history["eval_epochs"].append(epoch)

                self._save_live_metrics(history, epoch, max_epochs)

                pbar.set_postfix(
                    loss=f"{loss_val:.4f}",
                    t_acc=f"{eval_result['train_acc']:.3f}",
                    v_acc=f"{eval_result['test_acc']:.3f}",
                    wnorm=f"{eval_result['weight_norm']:.1f}",
                )

            # Periodic Fourier analysis
            if epoch % fourier_interval == 0 or epoch == max_epochs - 1:
                fourier_result = self._fourier_snapshot(p, n_top)
                history["gini"].append(fourier_result["gini"])
                history["frequency_norms_snapshots"].append(
                    fourier_result["frequency_norms"].tolist()
                )
                history["key_frequencies_snapshots"].append(
                    fourier_result["key_frequencies"].tolist()
                )
                history["fourier_epochs"].append(epoch)

            # Periodic checkpointing
            if epoch % checkpoint_interval == 0 or epoch == max_epochs - 1:
                if self.output_dir is not None:
                    ckpt_dir = self.output_dir / "checkpoints"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    ckpt_path = ckpt_dir / f"epoch_{epoch}.pt"
                    torch.save(self.model.state_dict(), ckpt_path)
                    checkpoints[epoch] = str(ckpt_path)

        pbar.close()
        self._cleanup_live_metrics()

        # Final evaluation
        final_eval = self._evaluate()
        final_fourier = self._fourier_snapshot(p, n_top)

        return {
            "history": history,
            "checkpoints": checkpoints,
            "final_train_loss": final_eval["train_loss"],
            "final_test_loss": final_eval["test_loss"],
            "final_train_acc": final_eval["train_acc"],
            "final_test_acc": final_eval["test_acc"],
            "final_weight_norm": final_eval["weight_norm"],
            "final_gini": final_fourier["gini"],
            "final_key_frequencies": final_fourier["key_frequencies"].tolist(),
            "final_frequency_norms": final_fourier["frequency_norms"].tolist(),
            "total_epochs": max_epochs,
            "n_params": self.model.count_parameters(),
        }

    @torch.no_grad()
    def _evaluate(self) -> dict:
        """Compute train/test loss, accuracy, and weight norm."""
        self.model.eval()

        # Train metrics
        train_logits = self.model(self.train_inputs)
        train_loss = self.criterion(train_logits, self.train_targets).item()
        train_preds = train_logits.argmax(dim=-1)
        train_acc = (train_preds == self.train_targets).float().mean().item()

        # Test metrics
        test_logits = self.model(self.test_inputs)
        test_loss = self.criterion(test_logits, self.test_targets).item()
        test_preds = test_logits.argmax(dim=-1)
        test_acc = (test_preds == self.test_targets).float().mean().item()

        # Weight norm
        w_norm = sum(p.data.norm() ** 2 for p in self.model.parameters()).sqrt().item()

        self.model.train()
        return {
            "train_loss": train_loss,
            "test_loss": test_loss,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "weight_norm": w_norm,
        }

    @torch.no_grad()
    def _fourier_snapshot(self, p: int, n_top: int) -> dict:
        """Compute Fourier analysis on current model's logit table."""
        logit_table = self.model.get_logit_table(self.device).cpu().numpy()
        result = compute_fourier_component_norms(logit_table, p)
        freq_norms = result["frequency_norms"]
        key_freqs = identify_key_frequencies(freq_norms, n_top=n_top)
        gini = compute_gini_coefficient(freq_norms)
        return {
            "frequency_norms": freq_norms,
            "key_frequencies": key_freqs,
            "gini": gini,
            "component_norms": result["component_norms"],
        }

    def _save_live_metrics(self, history: dict, epoch: int, max_epochs: int):
        """Write current training history for real-time monitoring."""
        if self.output_dir is None:
            return
        self.output_dir.mkdir(parents=True, exist_ok=True)
        live_path = self.output_dir / "live_metrics.json"

        # Serialize
        ser_history = {}
        for k, v in history.items():
            if isinstance(v, list) and v:
                if isinstance(v[0], (list, np.ndarray)):
                    # Skip array-valued entries for live metrics (too large)
                    continue
                elif isinstance(v[0], (int, float, np.floating)):
                    ser_history[k] = [float(x) for x in v]
                else:
                    ser_history[k] = v
            else:
                ser_history[k] = v

        data = {
            "history": ser_history,
            "current_epoch": epoch,
            "max_epochs": max_epochs,
            "timestamp": time.time(),
            "config": {
                "p": self.config.get("p"),
                "d_model": self.config.get("d_model"),
                "n_heads": self.config.get("n_heads"),
                "d_mlp": self.config.get("d_mlp"),
                "n_layers": self.config.get("n_layers"),
                "weight_decay": self.config.get("weight_decay"),
                "seed": self.config.get("seed"),
            },
        }
        tmp_path = live_path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        tmp_path.rename(live_path)

    def _cleanup_live_metrics(self):
        """Remove live_metrics.json after training completes."""
        if self.output_dir is None:
            return
        live_path = self.output_dir / "live_metrics.json"
        if live_path.exists():
            live_path.unlink()
