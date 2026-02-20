# returns_ebm

Energy-based model (EBM) prototype for ETF return forecasting with macro features.

## Documentation

- `docs/training_pipeline.md`: training pipeline walkthrough, model description, and guidance for interpreting metrics/plots/artifacts.

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
  --mlflow-tracking-uri sqlite:///runs/mlflow/mlflow.db \
  --mlflow-experiment returns_ebm
```

Useful flags:

- `--mlflow-tracking-uri sqlite:///runs/mlflow/mlflow.db`: use a SQLite DB backend for MLflow tracking.
- `--mlflow-experiment returns_ebm`: group runs under an MLflow experiment.
- `--timeout 86400`: stop optimization after a fixed number of seconds.
- `--train-ratio 0.70 --val-ratio 0.15`: adjust time-based data splits.

## Start MLflow DB with Docker

Start a PostgreSQL container for MLflow backend storage:

```bash
./scripts/start_mlflow_db.sh
```

Then point training to that DB:

```bash
python3 ebm/main.py \
  --mlflow-tracking-uri postgresql+psycopg2://mlflow:mlflow@localhost:5432/mlflow \
  --mlflow-experiment returns_ebm
```

Optional overrides (defaults shown):

- `MLFLOW_DB_CONTAINER=returns-ebm-mlflow-db`
- `MLFLOW_DB_IMAGE=postgres:16`
- `MLFLOW_DB_PORT=5432`
- `MLFLOW_DB_NAME=mlflow`
- `MLFLOW_DB_USER=mlflow`
- `MLFLOW_DB_PASSWORD=mlflow`

Artifacts are written under `runs/optuna/<study-name>/`, including:

- `trial_XXXX/summary.txt`
- `trial_XXXX/results.json`
- `trial_XXXX/epoch_metrics.csv`
- `trial_XXXX/training_curves.png`
- `trial_XXXX/validation_prediction_diagnostics.png`
- `trial_XXXX/test_prediction_diagnostics.png`
- `trial_XXXX/model_state.pt`

All trial metrics, params, and artifacts are also logged to MLflow.
