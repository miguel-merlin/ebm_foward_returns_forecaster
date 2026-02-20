import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import mlflow
import optuna
import pandas as pd
import torch
from tqdm.auto import tqdm

from loader import DataLoader
from model import (
    FinancialEnergyModel,
    TrialPrunedSignal,
    evaluate_model,
    train_energy_model,
)
from reporting import save_trial_summary
from tracking import to_mlflow_params
from utils import build_hidden_dims, set_seed, split_data_by_time
from visualization import (
    save_prediction_plots,
    save_study_summary,
    save_training_plots,
)


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
        print(f"[trial {trial.number:04d}] Starting hyperparameter sampling")
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
            "batch_size": trial.suggest_categorical(
                "batch_size", [128, 256, 512, 1024]
            ),
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
        with mlflow.start_run(run_name=f"trial_{trial.number:04d}", nested=True):
            print(
                f"[trial {trial.number:04d}] Run started | artifact_dir={trial_dir} "
                f"| params={params}"
            )
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_param("optuna_study", trial.study.study_name)
            mlflow.log_params(to_mlflow_params(trial.params, prefix="optuna."))
            mlflow.log_params(to_mlflow_params(params, prefix="trial."))

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
                print(f"[trial {trial.number:04d}] Pruned by Optuna")
                mlflow.set_tag("trial_state", "PRUNED")
                raise optuna.exceptions.TrialPruned()

            history_df = pd.DataFrame(history)
            history_df.to_csv(trial_dir / "epoch_metrics.csv", index=False)
            save_training_plots(history_df, trial_dir / "training_curves.png")

            if "val_rmse" not in history_df.columns:
                raise RuntimeError(
                    "Validation metrics were not generated during training."
                )

            best_idx = int(history_df["val_rmse"].idxmin())
            best_epoch = int(history_df.loc[best_idx, "epoch"])  # type: ignore
            best_val_rmse = float(history_df.loc[best_idx, "val_rmse"])  # type: ignore

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

            mlflow.log_metric("best_val_rmse", best_val_rmse)
            mlflow.log_metric("best_epoch", float(best_epoch))
            for metric_name, metric_value in val_eval["metrics"].items():
                mlflow.log_metric(f"val_{metric_name}", float(metric_value))
            for metric_name, metric_value in test_eval["metrics"].items():
                mlflow.log_metric(f"test_{metric_name}", float(metric_value))
            mlflow.set_tag("trial_state", "COMPLETE")
            mlflow.log_artifacts(str(trial_dir), artifact_path="trial_artifacts")
            print(
                f"[trial {trial.number:04d}] Complete | best_val_rmse={best_val_rmse:.6f} "
                f"| best_epoch={best_epoch} | val_rmse={val_eval['metrics']['rmse']:.6f} "
                f"| test_rmse={test_eval['metrics']['rmse']:.6f}"
            )

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
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="runs/optuna")
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="sqlite:///runs/mlflow/mlflow.db",
    )
    parser.add_argument("--mlflow-experiment", type=str, default="returns_ebm")
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
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
    with tqdm(total=5, desc="Data pipeline", unit="step") as data_pbar:
        loader = DataLoader()
        data_pbar.set_postfix({"stage": "init_loader"})
        data_pbar.update(1)

        etf_data = loader.load_etf_data()
        data_pbar.set_postfix({"stage": "load_etf"})
        data_pbar.update(1)

        macro_data = loader.load_macro_data()
        data_pbar.set_postfix({"stage": "load_macro"})
        data_pbar.update(1)

        dataset = loader.prepare_training_data(
            etf_data=etf_data,
            macro_data=macro_data,
            lookback=args.lookback,
            forward_horizon=args.forward_horizon,
        )
        data_pbar.set_postfix({"stage": "prepare_dataset"})
        data_pbar.update(1)

        n_assets = etf_data.shape[1]
        train_data, val_data, test_data = split_data_by_time(
            data=dataset,
            n_assets=n_assets,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        data_pbar.set_postfix({"stage": "split_data"})
        data_pbar.update(1)

    print(
        f"Samples -> train: {train_data['past_prices'].shape[0]}, "
        f"val: {val_data['past_prices'].shape[0]}, "
        f"test: {test_data['past_prices'].shape[0]}"
    )
    print(
        "Feature shapes -> "
        f"past_prices: {tuple(train_data['past_prices'].shape)}, "
        f"current_prices: {tuple(train_data['current_prices'].shape)}, "
        f"macro_factors: {tuple(train_data['macro_factors'].shape)}"
    )

    output_dir = Path(args.output_dir) / args.study_name
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = vars(args).copy()
    run_config["resolved_device"] = device
    run_config["timestamp_utc"] = datetime.utcnow().isoformat()
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))

    if args.mlflow_tracking_uri.startswith("sqlite:///"):
        raw_path = args.mlflow_tracking_uri[len("sqlite:///") :]
        if raw_path:
            db_path = (
                Path(f"/{raw_path}")
                if args.mlflow_tracking_uri.startswith("sqlite:////")
                else Path(raw_path)
            )
            db_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    print(
        "MLflow configured -> "
        f"tracking_uri: {args.mlflow_tracking_uri}, "
        f"experiment: {args.mlflow_experiment}"
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    print(
        "Optuna configured -> "
        f"study={args.study_name}, sampler={sampler.__class__.__name__}, "
        f"pruner={pruner.__class__.__name__}, output_dir={output_dir}"
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

    with mlflow.start_run(run_name=args.study_name):
        mlflow.log_params(to_mlflow_params(run_config, prefix="run."))

        print(
            f"Starting Optuna optimization | trials={args.n_trials}, "
            f"epochs_per_trial={args.epochs}"
        )
        with tqdm(
            total=args.n_trials, desc="Optuna trials", unit="trial"
        ) as trial_pbar:

            def _optuna_tqdm_callback(
                callback_study: optuna.Study, callback_trial: optuna.Trial
            ) -> None:
                trial_pbar.update(1)
                postfix = {"last_state": callback_trial.state.name}  # type: ignore
                try:
                    best_trial = callback_study.best_trial
                except ValueError:
                    best_trial = None
                if best_trial is not None:
                    postfix["best_rmse"] = f"{float(best_trial.value):.6f}"  # type: ignore
                    postfix["best_trial"] = best_trial.number
                trial_pbar.set_postfix(postfix)

            study.optimize(
                objective,
                n_trials=args.n_trials,
                timeout=args.timeout,
                gc_after_trial=True,
                show_progress_bar=False,
                callbacks=[_optuna_tqdm_callback],  # type: ignore
            )

        save_study_summary(study, output_dir)
        mlflow.log_metric("best_trial_number", float(study.best_trial.number))
        mlflow.log_metric("best_validation_rmse", float(study.best_trial.value))  # type: ignore
        mlflow.log_params(to_mlflow_params(study.best_trial.params, prefix="best."))
        mlflow.log_artifacts(str(output_dir), artifact_path="study_artifacts")

        print("Optimization complete.")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation RMSE: {study.best_trial.value:.6f}")
        print(f"Best params: {study.best_trial.params}")
        print(f"Artifacts saved under: {output_dir}")


if __name__ == "__main__":
    main()
