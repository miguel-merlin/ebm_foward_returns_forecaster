import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd


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


def save_prediction_plots(
    eval_out: Dict[str, Any], split_name: str, out_dir: Path
) -> None:
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
    best_so_far = np.minimum.accumulate(values)  # type: ignore

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(trial_ids, values, marker="o", linestyle="-", alpha=0.7, label="val_rmse")  # type: ignore
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
