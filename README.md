# returns_ebm

Energy-based model (EBM) prototype for ETF return forecasting with macro features.

## Sources

The code in `ebm/main.py` loads data exclusively from local CSVs via `DataLoader`:

- `data/etfs/*.csv`: ETF price history. Each file should include a `Close` column; filenames become ETF tickers in the combined table.
- `data/macro/*.csv`: macro indicator time series. All columns are concatenated into a single macro feature table.

## Optuna Training

Run hyperparameter optimization with per-trial directories containing metrics, plots, summaries, and saved model weights:

```bash
python3 ebm/main.py \
  --device cuda \
  --n-trials 100 \
  --epochs 200 \
  --study-name ebm_long_gpu_run \
  --output-dir runs/optuna \
  --mlflow-tracking-uri file:./runs/mlflow \
  --mlflow-experiment returns_ebm
```

Useful flags:

- `--mlflow-tracking-uri file:./runs/mlflow`: set the MLflow backend store.
- `--mlflow-experiment returns_ebm`: group runs under an MLflow experiment.
- `--timeout 86400`: stop optimization after a fixed number of seconds.
- `--train-ratio 0.70 --val-ratio 0.15`: adjust time-based data splits.

Artifacts are written under `runs/optuna/<study-name>/`, including:

- `trial_XXXX/summary.txt`
- `trial_XXXX/results.json`
- `trial_XXXX/epoch_metrics.csv`
- `trial_XXXX/training_curves.png`
- `trial_XXXX/validation_prediction_diagnostics.png`
- `trial_XXXX/test_prediction_diagnostics.png`
- `trial_XXXX/model_state.pt`

All trial metrics, params, and artifacts are also logged to MLflow.
