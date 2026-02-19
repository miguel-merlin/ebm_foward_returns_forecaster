# Training Pipeline, Model, and Results Interpretation

This document explains how the training pipeline works, what the Energy-Based Model (EBM) is learning, and how to interpret outputs from each trial.

## 1. End-to-end pipeline

The CLI entrypoint is `ebm/main.py`.

1. Parse run config
- Reads CLI flags like `--n-trials`, `--epochs`, `--lookback`, `--forward-horizon`, and MLflow settings.

2. Load and prepare data
- `DataLoader.load_etf_data()` loads `data/etfs/*.csv` close prices and forward-fills missing values.
- `DataLoader.load_macro_data()` loads `data/macro/*.csv` and forward-fills missing values.
- `DataLoader.prepare_training_data()`:
  - aligns ETF and macro dates
  - converts ETF prices to returns
  - z-normalizes macro features
  - builds supervised samples with:
    - `past_prices` (lookback window of returns)
    - `current_prices` (current return)
    - `macro_factors` (normalized macro row)
    - `actual_returns` (sum of returns over the forward horizon)

3. Time-based split
- `split_data_by_time(...)` splits by timestep (not random shuffle):
  - train: earliest segment
  - validation: middle segment
  - test: most recent segment

4. Hyperparameter optimization (Optuna)
- Each trial samples architecture and training/eval hyperparameters.
- Objective minimizes validation RMSE.
- Pruning uses `MedianPruner` based on intermediate validation RMSE.

5. Per-trial training
- `train_energy_model(...)` trains with contrastive divergence:
  - lower energy for real samples
  - higher energy for corrupted samples
- Validation evaluation runs each epoch and is logged for pruning and diagnostics.

6. Per-trial evaluation
- `evaluate_model(...)` runs energy minimization to infer prediction + confidence interval for each sample.
- Computes predictive metrics and energy diagnostics for validation/test.

7. Tracking and artifacts
- MLflow logs params, metrics, and trial/study artifacts.
- Files are written to `runs/optuna/<study-name>/trial_XXXX/` and study-level outputs under the study root.

## 2. Model description

The core model is `FinancialEnergyModel` in `ebm/model.py`.

### Inputs

For each sample, the model receives:
- lookback returns vector (`past_prices`, length = `lookback`)
- current return (`current_prices`, scalar)
- macro feature vector (`macro_factors`)
- candidate forward return prediction (`forward_return_pred`, scalar)
- candidate confidence interval (`confidence_interval`, scalar)

These are concatenated and passed to an MLP.

### Output

- A scalar energy value.
- Interpretation: lower energy means the candidate `(forward_return_pred, confidence_interval)` is more plausible given recent price behavior and macro state.

### Why this is different from direct regression

Instead of directly outputting one return value, the EBM learns an energy landscape over candidate predictions.  
At inference/evaluation time, candidate prediction variables are optimized to minimize energy.

### Training objective intuition

`contrastive_divergence_loss(...)` uses:
- positive samples: real observed forward returns
- negative samples: corrupted returns and CIs

Loss pushes:
- energy(real) down
- energy(corrupted) up

with a regularization term to prevent unbounded energy drift.

## 3. How to interpret trial outputs

Each trial writes:
- `epoch_metrics.csv`
- `training_curves.png`
- `validation_prediction_diagnostics.png`
- `test_prediction_diagnostics.png`
- `results.json`
- `summary.txt`
- `model_state.pt`

### Primary model-quality metrics

Focus first on validation/test:
- `rmse` and `mae`: lower is better, unit is return space.
- `r2`: closer to 1 is better; negative means worse than predicting mean.
- `directional_accuracy`: fraction of correct sign predictions.
- `correlation`: linear alignment of predicted vs true returns.

For selection, prioritize:
1. low validation RMSE
2. stable validation-to-test gap
3. acceptable directional accuracy/correlation for your use case

### Energy-specific diagnostics

From `evaluate_model(...)`:
- `pred_energy_mean`: mean energy at inferred predictions.
- `true_energy_mean`: mean energy at realized returns.
- `energy_gap = mean(pred_energy - true_energy)`.

Interpretation guideline:
- `energy_gap` near 0: inferred predictions and realized outcomes are similarly plausible under model energy.
- large positive `energy_gap`: model assigns higher energy to inferred predictions than to realized outcomes; inference settings or learned landscape may be misaligned.
- large negative `energy_gap`: inferred predictions are unrealistically favored; can indicate overconfident optimization dynamics.

Energy metrics are diagnostic, not standalone business KPIs. Use them with RMSE/MAE/directional metrics.

### Plot interpretation

`training_curves.png`
- Training loss should generally trend down.
- Validation errors (`val_rmse`, `val_mae`) should improve early and then flatten.
- Divergence (train improves, validation worsens) suggests overfitting.

`*_prediction_diagnostics.png`
- Predicted vs Actual scatter:
  - tighter around diagonal is better calibration.
- Residual histogram:
  - centered near 0 is desirable.
  - fat tails imply occasional large misses.
- Predicted CI histogram:
  - very narrow CIs can indicate overconfidence.
  - very wide CIs may indicate weak signal.
- Energy histogram:
  - compare predicted-energy vs true-energy distributions for consistency.

## 4. Common failure modes to watch

- Data leakage: ensure only time-forward splits are used.
- Degenerate variance: if predictions are near constant, correlation and R2 can collapse.
- Over-optimization in inference:
  - too many optimization iterations or high inference LR can produce unstable inferred predictions.
- Horizon mismatch:
  - ensure `forward_horizon` aligns with the return horizon your downstream process expects.

## 5. Practical model-selection checklist

1. Start from top trials by validation RMSE.
2. Remove trials with large validation/test degradation.
3. Check directional accuracy and correlation.
4. Review residual and scatter plots for calibration issues.
5. Use energy diagnostics to detect inference/landscape mismatches.
6. Confirm stability across reruns with different seeds before promoting a model.
