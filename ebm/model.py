import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Tuple, List


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
    ):
        super().__init__()

        self.price_lookback = price_lookback
        self.n_macro_features = n_macro_features

        # Input size: price_lookback + 1 (current) + macro + 2 (prediction + confidence)
        input_dim = price_lookback + 1 + n_macro_features + 2

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
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
        confidence = torch.abs(
            torch.randn(batch_size, 1, requires_grad=True, device=self.device)
        )

        optimizer = optim.Adam([forward_pred, confidence], lr=lr)

        self.model.eval()
        with torch.enable_grad():
            for _ in range(n_iterations):
                optimizer.zero_grad()

                energy = self.model(
                    past_prices,
                    current_price,
                    macro_factors,
                    forward_pred,
                    torch.abs(confidence),
                )

                loss = energy.mean()
                loss.backward()
                optimizer.step()

        final_energy = energy.detach()

        return (
            forward_pred.detach(),
            torch.abs(confidence.detach()),
            final_energy.mean().item(),
        )


def contrastive_divergence_loss(
    model: FinancialEnergyModel,
    data: Dict[str, torch.Tensor],
    n_negative_samples: int = 10,
    noise_std: float = 0.05,
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
    loss += 0.01 * (positive_energy**2).mean()

    return loss


def train_energy_model(
    model: FinancialEnergyModel,
    train_data: Dict[str, torch.Tensor],
    n_epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-4,
    device: str = "cuda",
):
    """Train the energy-based model"""

    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for key in train_data:
        train_data[key] = train_data[key].to(device)

    n_samples = train_data["past_prices"].shape[0]

    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        indices = torch.randperm(n_samples)  # Mini-batch training

        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i : i + batch_size]

            batch_data = {key: val[batch_indices] for key, val in train_data.items()}

            optimizer.zero_grad()
            loss = contrastive_divergence_loss(model, batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / n_batches

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
            )
