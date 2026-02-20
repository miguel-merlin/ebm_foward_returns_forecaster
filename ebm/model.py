import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, List, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm.auto import tqdm


class TrialPrunedSignal(Exception):
    """Internal signal used to propagate pruning from the training loop."""


class FinancialEnergyModel(nn.Module):
    """
    Energy-Based Model for financial forecasting.
    Lower energy = more plausible predictions
    """

    def __init__(
        self,
        price_lookback: int = 60,
        n_macro_features: int = 10,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        activation: str = "relu",
        use_layernorm: bool = True,
    ):
        super().__init__()

        self.price_lookback = price_lookback
        self.n_macro_features = n_macro_features

        # Input size: price_lookback + 1 (current) + macro + 2 (prediction + confidence)
        input_dim = price_lookback + 1 + n_macro_features + 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(_build_activation(activation))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.energy_net = nn.Sequential(*layers)

    def forward(
        self,
        past_prices: torch.Tensor,
        current_price: torch.Tensor,
        macro_factors: torch.Tensor,
        forward_return_pred: torch.Tensor,
        confidence_interval: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy for given inputs.

        Args:
            past_prices: (batch, lookback) - historical price sequence
            current_price: (batch, 1) - current price
            macro_factors: (batch, n_macro) - macro indicators
            forward_return_pred: (batch, 1) - predicted forward return
            confidence_interval: (batch, 1) - prediction confidence interval

        Returns:
            energy: (batch, 1) - energy score (lower is better)
        """
        x = torch.cat(
            [
                past_prices,
                current_price,
                macro_factors,
                forward_return_pred,
                confidence_interval,
            ],
            dim=1,
        )

        energy = self.energy_net(x)
        return energy


def _build_activation(name: str) -> nn.Module:
    activation = name.lower()
    if activation == "relu":
        return nn.ReLU()
    if activation == "gelu":
        return nn.GELU()
    if activation == "silu":
        return nn.SiLU()
    if activation == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported activation '{name}'")


class EnergyOptimizer:
    """Optimize predictions by minimizing energy"""

    def __init__(self, model: FinancialEnergyModel, device: str = "cuda"):
        self.model = model.to(device)
        self.device = device

    def sample_predictions(
        self,
        batch_size: int,
        return_mean: float = 0.0,
        return_std: float = 0.05,
        ci_mean: float = 0.02,
        ci_std: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample random predictions and confidence intervals

        Returns:
            forward_return_pred, confidence_interval
        """
        forward_returns = torch.randn(batch_size, 1) * return_std + return_mean
        confidence_intervals = torch.abs(torch.randn(batch_size, 1) * ci_std + ci_mean)

        return forward_returns.to(self.device), confidence_intervals.to(self.device)

    def optimize_predictions(
        self,
        past_prices: torch.Tensor,
        current_price: torch.Tensor,
        macro_factors: torch.Tensor,
        n_samples: int = 100,
        n_iterations: int = 50,
        lr: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """
        Find predictions that minimize energy via gradient descent

        Returns:
            Best forward_return_pred, confidence_interval, energy
        """
        batch_size = past_prices.shape[0]

        forward_pred = torch.randn(
            batch_size, 1, requires_grad=True, device=self.device
        )
        confidence_raw = torch.randn(
            batch_size, 1, requires_grad=True, device=self.device
        )

        optimizer = optim.Adam([forward_pred, confidence_raw], lr=lr)

        self.model.eval()
        with torch.enable_grad():
            for _ in range(n_iterations):
                optimizer.zero_grad()

                energy = self.model(
                    past_prices,
                    current_price,
                    macro_factors,
                    forward_pred,
                    torch.abs(confidence_raw),
                )

                loss = energy.mean()
                loss.backward()
                optimizer.step()

        final_energy = energy.detach()

        return (
            forward_pred.detach(),
            torch.abs(confidence_raw.detach()),
            final_energy.mean().item(),
        )


def contrastive_divergence_loss(
    model: FinancialEnergyModel,
    data: Dict[str, torch.Tensor],
    n_negative_samples: int = 10,
    noise_std: float = 0.05,
    reg_weight: float = 0.01,
) -> torch.Tensor:
    """
    Contrastive Divergence loss for training the energy model

    Positive samples: actual data with real forward returns
    Negative samples: data with corrupted/random forward returns
    """
    # Positive samples (real data with actual returns)
    positive_energy = model(
        data["past_prices"],
        data["current_prices"],
        data["macro_factors"],
        data["actual_returns"],
        torch.abs(
            torch.randn_like(data["actual_returns"]) * 0.01 + 0.02
        ),  # reasonable CI
    )

    # Negative samples (corrupted predictions)
    negative_energies = []
    for _ in range(n_negative_samples):
        # Add noise to actual returns to create implausible predictions
        corrupted_returns = (
            data["actual_returns"]
            + torch.randn_like(data["actual_returns"]) * noise_std
        )
        corrupted_ci = torch.abs(torch.randn_like(data["actual_returns"]) * 0.05)

        neg_energy = model(
            data["past_prices"],
            data["current_prices"],
            data["macro_factors"],
            corrupted_returns,
            corrupted_ci,
        )
        negative_energies.append(neg_energy)

    negative_energy = torch.stack(negative_energies).mean(dim=0)

    # Loss: minimize energy of real data, maximize energy of fake data
    loss = positive_energy.mean() - negative_energy.mean()

    # Add regularization to prevent energy from becoming arbitrarily negative
    loss += reg_weight * (positive_energy**2).mean()

    return loss


def optimize_predictions_in_batches(
    model: FinancialEnergyModel,
    data: Dict[str, torch.Tensor],
    device: str,
    optimization_iterations: int = 50,
    optimization_lr: float = 0.01,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run energy minimization over a dataset and return predictions, CIs, and energies."""
    model.eval()
    optimizer_helper = EnergyOptimizer(model, device=device)

    y_pred_batches = []
    ci_batches = []
    pred_energy_batches = []
    true_energy_batches = []

    n_samples = data["past_prices"].shape[0]
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch_slice = slice(i, i + batch_size)
            past = data["past_prices"][batch_slice].to(device)
            current = data["current_prices"][batch_slice].to(device)
            macro = data["macro_factors"][batch_slice].to(device)
            actual = data["actual_returns"][batch_slice].to(device)

            pred_returns, pred_ci, _ = optimizer_helper.optimize_predictions(
                past_prices=past,
                current_price=current,
                macro_factors=macro,
                n_iterations=optimization_iterations,
                lr=optimization_lr,
            )

            pred_energy = model(past, current, macro, pred_returns, pred_ci)
            true_ci = torch.full_like(actual, 0.02)
            true_energy = model(past, current, macro, actual, true_ci)

            y_pred_batches.append(pred_returns.detach().cpu().numpy())
            ci_batches.append(pred_ci.detach().cpu().numpy())
            pred_energy_batches.append(pred_energy.detach().cpu().numpy())
            true_energy_batches.append(true_energy.detach().cpu().numpy())

    return (
        np.vstack(y_pred_batches).reshape(-1),
        np.vstack(ci_batches).reshape(-1),
        np.vstack(pred_energy_batches).reshape(-1),
        np.vstack(true_energy_batches).reshape(-1),
    )


def evaluate_model(
    model: FinancialEnergyModel,
    data: Dict[str, torch.Tensor],
    device: str,
    optimization_iterations: int = 50,
    optimization_lr: float = 0.01,
    batch_size: int = 256,
) -> Dict[str, Any]:
    """Compute prediction/energy metrics for a dataset."""
    y_true = data["actual_returns"].detach().cpu().numpy().reshape(-1)
    y_pred, y_ci, pred_energy, true_energy = optimize_predictions_in_batches(
        model=model,
        data=data,
        device=device,
        optimization_iterations=optimization_iterations,
        optimization_lr=optimization_lr,
        batch_size=batch_size,
    )

    residuals = y_true - y_pred
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), 1e-6)
    mape = float(np.mean(np.abs(residuals) / denom))
    direction_acc = float((np.sign(y_true) == np.sign(y_pred)).mean())

    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        corr = 0.0
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])

    metrics = {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": mape,
        "directional_accuracy": direction_acc,
        "correlation": corr,
        "pred_energy_mean": float(np.mean(pred_energy)),
        "true_energy_mean": float(np.mean(true_energy)),
        "energy_gap": float(np.mean(pred_energy - true_energy)),
        "pred_ci_mean": float(np.mean(y_ci)),
        "pred_ci_std": float(np.std(y_ci)),
    }

    return {
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "residuals": residuals,
        "y_ci": y_ci,
        "pred_energy": pred_energy,
        "true_energy": true_energy,
    }


def train_energy_model(
    model: FinancialEnergyModel,
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor] = {},
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    n_negative_samples: int = 10,
    noise_std: float = 0.05,
    reg_weight: float = 0.01,
    eval_optimization_iterations: int = 50,
    eval_optimization_lr: float = 0.01,
    device: str = "cuda",
    trial: Any = None,
) -> List[Dict[str, float]]:
    """Train the energy-based model and return per-epoch metrics."""

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    train_data_device = {key: val.to(device) for key, val in train_data.items()}
    val_data_device = (
        {key: val.to(device) for key, val in val_data.items()} if val_data else None
    )

    n_samples = train_data_device["past_prices"].shape[0]
    history: List[Dict[str, float]] = []

    epoch_desc = (
        f"Trial {trial.number:04d} epochs"
        if trial is not None and hasattr(trial, "number")
        else "Training epochs"
    )
    epoch_pbar = tqdm(range(n_epochs), desc=epoch_desc, unit="epoch", leave=False)
    for epoch in epoch_pbar:
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_samples)  # Mini-batch training

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]

            batch_data = {
                key: val[batch_indices] for key, val in train_data_device.items()
            }

            optimizer.zero_grad()
            loss = contrastive_divergence_loss(
                model,
                batch_data,
                n_negative_samples=n_negative_samples,
                noise_std=noise_std,
                reg_weight=reg_weight,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches

        epoch_result = {
            "epoch": float(epoch + 1),
            "train_loss": float(avg_loss),
            "lr": float(scheduler.get_last_lr()[0]),
        }

        if val_data_device:
            eval_out = evaluate_model(
                model=model,
                data=val_data_device,
                device=device,
                optimization_iterations=eval_optimization_iterations,
                optimization_lr=eval_optimization_lr,
                batch_size=batch_size,
            )
            for metric_name, metric_val in eval_out["metrics"].items():
                epoch_result[f"val_{metric_name}"] = float(metric_val)

            if trial is not None:
                trial.report(eval_out["metrics"]["rmse"], step=epoch + 1)
                if trial.should_prune():
                    raise TrialPrunedSignal()

        history.append(epoch_result)

        tqdm_postfix = {
            "loss": f"{avg_loss:.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.6f}",
        }
        val_rmse = epoch_result.get("val_rmse")
        if val_rmse is not None:
            tqdm_postfix["val_rmse"] = f"{val_rmse:.6f}"
        epoch_pbar.set_postfix(tqdm_postfix)

    return history
