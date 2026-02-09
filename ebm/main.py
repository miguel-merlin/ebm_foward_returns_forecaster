import torch
from loader import DataLoader
from model import FinancialEnergyModel, train_energy_model, EnergyOptimizer


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("Loading data...")
    data_loader = DataLoader()
    etf_data = data_loader.load_etf_data()
    macro_data = data_loader.load_macro_data()

    print(f"ETF data shape: {etf_data.shape}")
    print(f"Macro data shape: {macro_data.shape}")
    print("Preparing training data...")
    train_data = data_loader.prepare_training_data(
        etf_data, macro_data, lookback=60, forward_horizon=5
    )
    print(f"Training samples: {train_data['past_prices'].shape[0]}")
    print("Initializing model...")
    model = FinancialEnergyModel(
        price_lookback=60,
        n_macro_features=macro_data.shape[1],
        hidden_dims=[256, 128, 64],
        dropout=0.1,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("\nTraining energy model...")
    train_energy_model(
        model=model,
        train_data=train_data,
        n_epochs=100,
        batch_size=256,
        lr=1e-4,
        device=device,
    )

    torch.save(model.state_dict(), "energy_model.pt")
    print("\nModel saved to energy_model.pt")

    print("\nExample prediction optimization...")
    optimizer = EnergyOptimizer(model, device=device)

    sample_idx = torch.randint(0, train_data["past_prices"].shape[0], (10,))

    best_pred, best_ci, best_energy = optimizer.optimize_predictions(
        past_prices=train_data["past_prices"][sample_idx].to(device),
        current_price=train_data["current_prices"][sample_idx].to(device),
        macro_factors=train_data["macro_factors"][sample_idx].to(device),
        n_iterations=100,
        lr=0.01,
    )

    print(f"\nOptimized predictions:")
    print(f"Predicted returns: {best_pred.cpu().numpy().flatten()}")
    print(f"Confidence intervals: {best_ci.cpu().numpy().flatten()}")
    print(f"Final energy: {best_energy:.4f}")
    print(
        f"Actual returns: {train_data['actual_returns'][sample_idx].cpu().numpy().flatten()}"
    )


if __name__ == "__main__":
    main()
