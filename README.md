# returns_ebm

Energy-based model (EBM) prototype for ETF return forecasting with macro features.

## Sources

The code in `ebm/main.py` loads data exclusively from local CSVs via `DataLoader`:

- `data/etfs/*.csv`: ETF price history. Each file should include a `Close` column; filenames become ETF tickers in the combined table.
- `data/macro/*.csv`: macro indicator time series. All columns are concatenated into a single macro feature table.

