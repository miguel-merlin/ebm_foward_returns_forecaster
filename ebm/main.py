import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch

from loader import DataLoader
from model import (
    FinancialEnergyModel,
    train_energy_model,
    evaluate_model,
    TrialPrunedSignal,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_data_by_time(
    data: Dict[str, torch.Tensor],
    n_assets: int,
    train_ratio: float,
    val_ratio: float,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    n_samples = data["past_prices"].shape[0]
    n_timesteps = n_samples // n_assets

    train_steps = int(n_timesteps * train_ratio)
    val_steps = int(n_timesteps * val_ratio)
    test_steps = n_timesteps - train_steps - val_steps
    if test_steps <= 0:
        raise ValueError(
            "Invalid split configuration. Ensure train_ratio + val_ratio < 1.0"
        )

    def _slice(start_step: int, end_step: int) -> Dict[str, torch.Tensor]:
        start = start_step * n_assets
        end = end_step * n_assets
        return {key: value[start:end] for key, value in data.items()}

    train_data = _slice(0, train_steps)
    val_data = _slice(train_steps, train_steps + val_steps)
    test_data = _slice(train_steps + val_steps, n_timesteps)
    return train_data, val_data, test_data


def build_hidden_dims(base_width: int, n_layers: int, decay: float) -> List[int]:
    dims = []
    for i in range(n_layers):
        dims.append(max(16, int(base_width * (decay**i))))
    return dims


def save_training_plots(history_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    axes[0].plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(history_df["epoch"], history_df["lr"], label="lr", color="tab:green")
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].grid(True, alpha=0.2)

    if "val_rmse" in history_df:
        axes[2].plot(history_df["epoch"], history_df["val_rmse"], label="val_rmse")
        axes[2].plot(history_df["epoch"], history_df["val_mae"], label="val_mae")
        axes[2].set_title("Validation Errors")
        axes[2].set_xlabel("Epoch")
        axes[2].legend()
        axes[2].grid(True, alpha=0.2)

    if "val_r2" in history_df:
        axes[3].plot(history_df["epoch"], history_df["val_r2"], label="val_r2")
        axes[3].plot(
            history_df["epoch"],
            history_df["val_directional_accuracy"],
            label="val_direction_acc",
        )
        axes[3].set_title("Validation Signal Quality")
        axes[3].set_xlabel("Epoch")
        axes[3].legend()
        axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def save_prediction_plots(eval_out: Dict[str, Any], split_name: str, out_dir: Path) -> None:
    y_true = eval_out["y_true"]
    y_pred = eval_out["y_pred"]
    residuals = eval_out["residuals"]
    y_ci = eval_out["y_ci"]
    pred_energy = eval_out["pred_energy"]
    true_energy = eval_out["true_energy"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    min_v = min(float(np.min(y_true)), float(np.min(y_pred)))
    max_v = max(float(np.max(y_true)), float(np.max(y_pred)))

    axes[0].scatter(y_true, y_pred, alpha=0.25, s=8)
    axes[0].plot([min_v, max_v], [min_v, max_v], color="tab:red", linestyle="--")
    axes[0].set_title(f"{split_name}: Predicted vs Actual")
    axes[0].set_xlabel("Actual Return")
    axes[0].set_ylabel("Predicted Return")
    axes[0].grid(True, alpha=0.2)

    axes[1].hist(residuals, bins=50, alpha=0.85)
    axes[1].set_title(f"{split_name}: Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.2)

    axes[2].hist(y_ci, bins=50, alpha=0.85, color="tab:green")
    axes[2].set_title(f"{split_name}: Predicted Confidence Interval Distribution")
    axes[2].set_xlabel("Predicted CI")
    axes[2].set_ylabel("Count")
    axes[2].grid(True, alpha=0.2)

    axes[3].hist(pred_energy, bins=50, alpha=0.65, label="pred_energy")
    axes[3].hist(true_energy, bins=50, alpha=0.65, label="true_energy")
    axes[3].set_title(f"{split_name}: Energy Distribution")
    axes[3].set_xlabel("Energy")
    axes[3].set_ylabel("Count")
    axes[3].legend()
    axes[3].grid(True, alpha=0.2)

    plt.tight_layout()
    fig.savefig(out_dir / f"{split_name.lower()}_prediction_diagnostics.png", dpi=200)
    plt.close(fig)


def save_trial_summary(
    trial_dir: Path,
    trial_number: int,
    params: Dict[str, Any],
    best_epoch: int,
    best_val_rmse: float,
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    lines = [
        f"Trial: {trial_number}",
        f"Timestamp (UTC): {datetime.utcnow().isoformat()}",
        "",
        "Best Validation RMSE",
        f"{best_val_rmse:.6f}",
        f"Best Epoch: {best_epoch}",
        "",
        "Parameters",
    ]
    for key in sorted(params.keys()):
        lines.append(f"- {key}: {params[key]}")

    lines.extend(["", "Validation Metrics"])
    for key in sorted(val_metrics.keys()):
        lines.append(f"- {key}: {val_metrics[key]:.6f}")

    lines.extend(["", "Test Metrics"])
    for key in sorted(test_metrics.keys()):
        lines.append(f"- {key}: {test_metrics[key]:.6f}")

    (trial_dir / "summary.txt").write_text("\n".join(lines))


def save_study_summary(study: optuna.Study, output_dir: Path) -> None:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    trials_df = study.trials_dataframe(
        attrs=("number", "value", "state", "params", "user_attrs")
    )
    trials_df.to_csv(output_dir / "study_trials.csv", index=False)

    if not completed:
        return

    ordered = sorted(completed, key=lambda t: t.number)
    trial_ids = [t.number for t in ordered]
    values = [t.value for t in ordered]
    best_so_far = np.minimum.accumulate(values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_ids, values, marker="o", linestyle="-", alpha=0.7, label="val_rmse")
    ax.plot(
        trial_ids,
        best_so_far,
        linestyle="--",
        color="tab:red",
        label="best_so_far",
    )
    ax.set_title("Optuna Optimization History")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation RMSE")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "optuna_optimization_history.png", dpi=200)
    plt.close(fig)

    importances = optuna.importance.get_param_importances(study)
    imp_df = pd.DataFrame(
        {
            "param": list(importances.keys()),
            "importance": list(importances.values()),
        }
    )
    imp_df.to_csv(output_dir / "optuna_param_importance.csv", index=False)

    if not imp_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(imp_df["param"], imp_df["importance"])
        ax.set_title("Optuna Parameter Importance")
        ax.set_xlabel("Importance")
        ax.grid(True, axis="x", alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / "optuna_param_importance.png", dpi=200)
        plt.close(fig)

    best = study.best_trial
    best_payload = {
        "trial_number": best.number,
        "best_value": best.value,
        "params": best.params,
        "timestamp_utc": datetime.utcnow().isoformat(),
    }
    (output_dir / "best_trial.json").write_text(json.dumps(best_payload, indent=2))


def build_objective(
    train_data: Dict[str, torch.Tensor],
    val_data: Dict[str, torch.Tensor],
    test_data: Dict[str, torch.Tensor],
    n_macro_features: int,
    price_lookback: int,
    n_epochs: int,
    device: str,
    output_dir: Path,
):
    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        n_layers = trial.suggest_int("n_layers", 2, 5)
        base_width = trial.suggest_categorical("base_width", [64, 128, 256, 384, 512])
        decay = trial.suggest_float("width_decay", 0.55, 1.0)
        hidden_dims = build_hidden_dims(base_width, n_layers, decay)

        params = {
            "hidden_dims": hidden_dims,
            "dropout": trial.suggest_float("dropout", 0.0, 0.45),
            "activation": trial.suggest_categorical(
                "activation", ["relu", "gelu", "silu", "elu"]
            ),
            "use_layernorm": trial.suggest_categorical("use_layernorm", [True, False]),
            "lr": trial.suggest_float("lr", 1e-5, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-7, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512, 1024]),
            "n_negative_samples": trial.suggest_int("n_negative_samples", 5, 24),
            "noise_std": trial.suggest_float("noise_std", 0.01, 0.2),
            "reg_weight": trial.suggest_float("reg_weight", 1e-4, 1e-1, log=True),
            "eval_optimization_iterations": trial.suggest_int(
                "eval_optimization_iterations", 20, 120
            ),
            "eval_optimization_lr": trial.suggest_float(
                "eval_optimization_lr", 1e-3, 5e-2, log=True
            ),
        }

        model = FinancialEnergyModel(
            price_lookback=price_lookback,
            n_macro_features=n_macro_features,
            hidden_dims=hidden_dims,
            dropout=params["dropout"],
            activation=params["activation"],
            use_layernorm=params["use_layernorm"],
        )

        try:
            history = train_energy_model(
                model=model,
                train_data=train_data,
                val_data=val_data,
                n_epochs=n_epochs,
                batch_size=params["batch_size"],
                lr=params["lr"],
                weight_decay=params["weight_decay"],
                n_negative_samples=params["n_negative_samples"],
                noise_std=params["noise_std"],
                reg_weight=params["reg_weight"],
                eval_optimization_iterations=params["eval_optimization_iterations"],
                eval_optimization_lr=params["eval_optimization_lr"],
                device=device,
                trial=trial,
            )
        except TrialPrunedSignal:
            raise optuna.exceptions.TrialPruned()

        history_df = pd.DataFrame(history)
        history_df.to_csv(trial_dir / "epoch_metrics.csv", index=False)
        save_training_plots(history_df, trial_dir / "training_curves.png")

        if "val_rmse" not in history_df.columns:
            raise RuntimeError("Validation metrics were not generated during training.")

        best_idx = int(history_df["val_rmse"].idxmin())
        best_epoch = int(history_df.loc[best_idx, "epoch"])
        best_val_rmse = float(history_df.loc[best_idx, "val_rmse"])

        val_eval = evaluate_model(
            model=model,
            data=val_data,
            device=device,
            optimization_iterations=params["eval_optimization_iterations"],
            optimization_lr=params["eval_optimization_lr"],
            batch_size=params["batch_size"],
        )
        test_eval = evaluate_model(
            model=model,
            data=test_data,
            device=device,
            optimization_iterations=params["eval_optimization_iterations"],
            optimization_lr=params["eval_optimization_lr"],
            batch_size=params["batch_size"],
        )

        save_prediction_plots(val_eval, "Validation", trial_dir)
        save_prediction_plots(test_eval, "Test", trial_dir)

        model_path = trial_dir / "model_state.pt"
        torch.save(model.state_dict(), model_path)

        payload = {
            "trial_number": trial.number,
            "best_epoch": best_epoch,
            "best_val_rmse": best_val_rmse,
            "params": params,
            "val_metrics": val_eval["metrics"],
            "test_metrics": test_eval["metrics"],
        }
        (trial_dir / "results.json").write_text(json.dumps(payload, indent=2))

        save_trial_summary(
            trial_dir=trial_dir,
            trial_number=trial.number,
            params=params,
            best_epoch=best_epoch,
            best_val_rmse=best_val_rmse,
            val_metrics=val_eval["metrics"],
            test_metrics=test_eval["metrics"],
        )

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("best_epoch", best_epoch)
        trial.set_user_attr("val_rmse", best_val_rmse)
        for metric_name, metric_value in test_eval["metrics"].items():
            trial.set_user_attr(f"test_{metric_name}", float(metric_value))

        return best_val_rmse

    return objective


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Long-running EBM training with Optuna HPO and per-trial artifacts."
    )
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lookback", type=int, default=60)
    parser.add_argument("--forward-horizon", type=int, default=5)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", type=str, default="ebm_hpo")
    parser.add_argument("--storage", type=str, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/optuna")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print(f"Using device: {device}")
    print("Loading data...")
    loader = DataLoader()
    etf_data = loader.load_etf_data()
    macro_data = loader.load_macro_data()
    dataset = loader.prepare_training_data(
        etf_data=etf_data,
        macro_data=macro_data,
        lookback=args.lookback,
        forward_horizon=args.forward_horizon,
    )

    n_assets = etf_data.shape[1]
    train_data, val_data, test_data = split_data_by_time(
        data=dataset,
        n_assets=n_assets,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    print(
        f"Samples -> train: {train_data['past_prices'].shape[0]}, "
        f"val: {val_data['past_prices'].shape[0]}, "
        f"test: {test_data['past_prices'].shape[0]}"
    )

    output_dir = Path(args.output_dir) / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    run_config["timestamp_utc"] = datetime.utcnow().isoformat()
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=args.storage,
        load_if_exists=True,
    )

    objective = build_objective(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        n_macro_features=macro_data.shape[1],
        price_lookback=args.lookback,
        n_epochs=args.epochs,
        device=device,
        output_dir=output_dir,
    )

    print(
        f"Starting Optuna optimization | trials={args.n_trials}, "
        f"epochs_per_trial={args.epochs}"
    )
    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    save_study_summary(study, output_dir)
    print("Optimization complete.")
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best validation RMSE: {study.best_trial.value:.6f}")
    print(f"Best params: {study.best_trial.params}")
    print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
