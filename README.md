# ARIMAX — Grunfeld Investment Forecast (Streamlit)

Interactive Streamlit version of `ARIMAX_NonSeasonal_Grunfeld_Friend.ipynb`.
Non-seasonal ARIMAX on annual Grunfeld investment data: one firm at a time,
target `Investment`, explanatory variables `Firm_Value` and `Capital_Stock`,
train on all but the last N years, forecast and score the held-out years.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## Defaults (same as the notebook)

| Setting | Default |
|---|---|
| Firm | General Motors |
| Target | `Investment` |
| Exog | `Firm_Value`, `Capital_Stock` |
| Order | ARIMA(1, 1, 0) |
| Test size | Last 4 years |
| Metrics | MAE, RMSE |

Change the firm, exog variables, `(p, d, q)` order, and test size in the
sidebar — the charts, forecast table, metrics, and diagnostics re-fit live.
With no exog selected the app fits a plain ARIMA instead of ARIMAX.

## Files

- `app.py` — Streamlit app
- `grunfeld_investment_nonseasonal.csv` — bundled dataset (loaded by default)
- `ARIMAX_NonSeasonal_Grunfeld_Friend.ipynb` — original notebook
- `requirements.txt` — Python dependencies

## Note on forecasting

The test-period exog values are supplied to score the forecast against
actuals, exactly as in the notebook. For a genuine future forecast, future
`Firm_Value` / `Capital_Stock` would need to be known or forecast separately.
