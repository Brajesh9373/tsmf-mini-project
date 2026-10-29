"""ARIMAX (non-seasonal) — Grunfeld Investment forecasting app.

Interactive Streamlit version of ``ARIMAX_NonSeasonal_Grunfeld_Friend.ipynb``.

Notebook mapping
----------------
Step 2  load ``grunfeld_investment_nonseasonal.csv``
Step 3  pick one firm (default: General Motors), annual ``Date`` index (``YS``)
Step 4  plot annual ``Investment``
Step 5  target ``Investment``; exog ``Firm_Value`` + ``Capital_Stock``;
        first ``n - TEST_SIZE`` rows train, last ``TEST_SIZE`` rows test
Step 6  ``ARIMA(endog, exog=..., order=(1, 1, 0))`` — ARIMAX when exog is used
Step 7  forecast the test years, supplying the test-period exog values
Step 8  MAE / RMSE (``sklearn.metrics``)
Step 9  actual vs predicted plot

Run:  streamlit run app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

# --------------------------------------------------------------------------
# Constants — defaults mirror the notebook exactly
# --------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "grunfeld_investment_nonseasonal.csv"
REQUIRED_COLUMNS = ["Year", "Firm", "Investment", "Firm_Value", "Capital_Stock"]
EXOG_CANDIDATES = ["Firm_Value", "Capital_Stock"]
DEFAULT_FIRM = "General Motors"
DEFAULT_ORDER = (1, 1, 0)  # (p, d, q)
DEFAULT_TEST_SIZE = 4

ACCENT = "#0f766e"  # teal-700
ACCENT_LIGHT = "rgba(15, 118, 110, 0.12)"
GREY = "#64748b"

st.set_page_config(
    page_title="ARIMAX — Grunfeld Investment Forecast",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
      h1 {{ font-size: 1.9rem; letter-spacing: -0.02em; }}
      h2 {{ font-size: 1.25rem; margin-top: 1.6rem; }}
      div[data-testid="stMetric"] {{
        border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.6rem 0.8rem;
      }}
      div[data-testid="stMetric"] label {{ color: {GREY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Data loading (Step 2 + Step 3 preparation)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path_str: str) -> pd.DataFrame:
    """Load and validate the bundled CSV. Raises on missing/bad input."""
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path.name} — place it next to app.py."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")
    df = df.copy()
    for col in ["Year", "Investment", "Firm_Value", "Capital_Stock"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Firm"] = df["Firm"].astype(str).str.strip()
    df = df.dropna(subset=["Year", "Investment"])
    df["Year"] = df["Year"].astype(int)
    return df.sort_values(["Firm", "Year"]).reset_index(drop=True)


def prepare_firm_series(df: pd.DataFrame, firm: str) -> pd.DataFrame:
    """Step 3: one firm, annual DatetimeIndex with yearly-start frequency."""
    company = df[df["Firm"] == firm].copy()
    company["Date"] = pd.to_datetime(company["Year"].astype(str) + "-01-01")
    company = company.sort_values("Date").set_index("Date")
    return company.asfreq("YS")


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def investment_chart(company: pd.DataFrame, firm: str) -> go.Figure:
    """Step 4: annual Investment for the selected firm."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=company.index,
            y=company["Investment"],
            mode="lines+markers",
            name="Investment",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=7),
            hovertemplate="%{x|%Y}<br>Investment: %{y:,.1f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{firm} — Annual Investment",
        xaxis_title="Year",
        yaxis_title="Investment (1947 dollars)",
        hovermode="x unified",
        margin=dict(l=40, r=20, t=50, b=40),
        height=380,
    )
    return fig


def actual_vs_predicted_chart(
    train_target: pd.Series,
    test_target: pd.Series,
    forecast: pd.Series,
    conf_int: pd.DataFrame | None,
    firm: str,
) -> go.Figure:
    """Step 9: training line, actual test points, forecast + optional 95% CI."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_target.index, y=train_target, mode="lines",
            name="Training Investment", line=dict(color=GREY, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_target.index, y=test_target, mode="lines+markers",
            name="Actual Test Investment",
            line=dict(color="#1e293b", width=2.5), marker=dict(size=8),
        )
    )
    if conf_int is not None and len(conf_int) == len(forecast):
        lower = conf_int.iloc[:, 0]
        upper = conf_int.iloc[:, 1]
        fig.add_trace(
            go.Scatter(
                x=list(forecast.index) + list(forecast.index[::-1]),
                y=list(upper) + list(lower[::-1]),
                fill="toself", fillcolor=ACCENT_LIGHT,
                line=dict(color="rgba(0,0,0,0)"),
                name="95% confidence interval", hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=forecast.index, y=forecast, mode="lines+markers",
            name="ARIMAX Forecast",
            line=dict(color=ACCENT, width=2.5, dash="dash"),
            marker=dict(symbol="x", size=9),
        )
    )
    fig.update_layout(
        title=f"{firm} — Actual vs Predicted Annual Investment",
        xaxis_title="Year",
        yaxis_title="Investment (1947 dollars)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=70, b=40),
        height=420,
    )
    return fig


def residuals_chart(residuals: pd.Series) -> go.Figure:
    """In-sample residuals over time (diagnostics)."""
    clean = residuals.dropna()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=clean.index, y=clean, mode="lines+markers",
            name="Residual", line=dict(color=ACCENT, width=2), marker=dict(size=6),
        )
    )
    fig.add_hline(y=0, line=dict(color=GREY, dash="dot"))
    fig.update_layout(
        title="In-sample residuals",
        xaxis_title="Year", yaxis_title="Residual",
        margin=dict(l=40, r=20, t=50, b=40), height=300,
    )
    return fig


# --------------------------------------------------------------------------
# Load data — fail loudly, never show a half-built page
# --------------------------------------------------------------------------
try:
    df = load_data(str(DATA_FILE))
except (FileNotFoundError, ValueError) as exc:
    st.title("ARIMAX — Grunfeld Investment Forecast")
    st.error(str(exc))
    st.stop()

firms = sorted(df["Firm"].unique().tolist())
if DEFAULT_FIRM not in firms:
    DEFAULT_FIRM_FALLBACK = firms[0]
else:
    DEFAULT_FIRM_FALLBACK = DEFAULT_FIRM

# --------------------------------------------------------------------------
# Sidebar — every notebook choice, exposed
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("Model settings")
    firm = st.selectbox(
        "Firm (Step 3)",
        options=firms,
        index=firms.index(DEFAULT_FIRM_FALLBACK),
        help="Notebook default: General Motors. One firm = one time series.",
        key="firm",
    )
    st.caption("Annual data — no monthly/weekly seasonality, hence non-seasonal ARIMAX.")

    st.subheader("Explanatory variables (Step 5)")
    use_firm_value = st.checkbox("Firm_Value", value=True, key="use_firm_value")
    use_capital_stock = st.checkbox("Capital_Stock", value=True, key="use_capital_stock")
    exog_cols = [
        c for c, on in zip(EXOG_CANDIDATES, [use_firm_value, use_capital_stock]) if on
    ]
    if not exog_cols:
        st.info("No exog selected — this fits a plain ARIMA, not ARIMAX.")

    st.subheader("ARIMA order (Step 6)")
    p = st.number_input("p — autoregressive lags", 0, 3, DEFAULT_ORDER[0], step=1, key="p")
    d = st.number_input("d — differencing", 0, 2, DEFAULT_ORDER[1], step=1, key="d")
    q = st.number_input("q — moving-average lags", 0, 3, DEFAULT_ORDER[2], step=1, key="q")
    order = (int(p), int(d), int(q))
    st.caption(f"Order = {order} · notebook default (1, 1, 0)")

    st.subheader("Train / test split (Step 5)")
    test_size = st.slider(
        "Test size (last N years)", min_value=2, max_value=8,
        value=DEFAULT_TEST_SIZE, step=1,
        help="Notebook default: 4. Training uses all earlier years.",
        key="test_size",
    )
    show_ci = st.checkbox("Show 95% confidence interval", value=True, key="show_ci")

    st.divider()
    st.caption(f"Data: `{DATA_FILE.name}` · {len(df)} rows · {len(firms)} firms")
    if st.button("Reset to notebook defaults"):
        for _k in ["firm", "use_firm_value", "use_capital_stock", "p", "d", "q", "test_size", "show_ci"]:
            st.session_state.pop(_k, None)
        st.rerun()

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("ARIMAX — Grunfeld Investment Forecast")
st.write(
    "Non-seasonal ARIMAX on annual Grunfeld investment data. "
    "Pick a firm and model settings in the sidebar — everything below "
    "re-fits and re-forecasts, mirroring Steps 2–9 of the notebook."
)
model_label = "ARIMAX" if exog_cols else "ARIMA (no exog)"
st.caption(
    f"Model: {model_label} {order} · Target: Investment · "
    f"Exog: {', '.join(exog_cols) if exog_cols else '—'} · Test: last {test_size} years"
)

# --------------------------------------------------------------------------
# Step 3 + 4: firm series
# --------------------------------------------------------------------------
company_df = prepare_firm_series(df, firm)
if len(company_df) < 10:
    st.error(f"Not enough annual observations for {firm} ({len(company_df)}). Pick another firm.")
    st.stop()
if test_size >= len(company_df) - 4:
    st.error(
        f"Test size {test_size} leaves too little training data for {firm} "
        f"({len(company_df)} rows). Lower the test size."
    )
    st.stop()

missing_exog = company_df[exog_cols].isnull().sum() if exog_cols else pd.Series(dtype=int)
if exog_cols and missing_exog.any():
    st.warning(
        "Missing exog values after yearly reindexing: "
        + ", ".join(f"{c}={int(v)}" for c, v in missing_exog.items() if v)
        + ". Those rows may drop out of the fit."
    )

c1, c2, c3, c4 = st.columns(4)
c1.metric("Firm", firm)
c2.metric("Observations", len(company_df))
c3.metric("Years", f"{int(company_df['Year'].min())}–{int(company_df['Year'].max())}")
c4.metric("Missing (Investment)", int(company_df["Investment"].isnull().sum()))

st.plotly_chart(investment_chart(company_df, firm), use_container_width=True)

with st.expander("Explanatory variables over time"):
    for col in EXOG_CANDIDATES:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=company_df.index, y=company_df[col], mode="lines+markers",
            name=col, line=dict(color=ACCENT, width=2),
        ))
        fig.update_layout(
            title=col, xaxis_title="Year", yaxis_title=col,
            margin=dict(l=40, r=20, t=50, b=40), height=260,
        )
        st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Step 5: split
# --------------------------------------------------------------------------
target = company_df["Investment"]
exog = company_df[exog_cols] if exog_cols else None
train_target, test_target = target.iloc[:-test_size], target.iloc[-test_size:]
train_exog = exog.iloc[:-test_size] if exog is not None else None
test_exog = exog.iloc[-test_size:] if exog is not None else None

s1, s2, s3 = st.columns(3)
s1.metric("Training observations", len(train_target))
s2.metric("Testing observations", len(test_target))
s3.metric("Test years", ", ".join(str(y) for y in test_target.index.year.tolist()))

# --------------------------------------------------------------------------
# Steps 6–8: fit, forecast, evaluate
# --------------------------------------------------------------------------
st.header("Forecast & evaluation")
st.caption(
    "Future exog values are not known in real forecasting — here the test-period "
    "exog is supplied (as in the notebook) so the forecast can be scored."
)
try:
    with st.spinner(f"Fitting {model_label}{order} on {len(train_target)} training years…"):
        result = ARIMA(train_target, exog=train_exog, order=order).fit()
        forecast_obj = result.get_forecast(steps=len(test_target), exog=test_exog)
        forecast = forecast_obj.predicted_mean
        forecast.index = test_target.index
        conf_int = forecast_obj.conf_int() if show_ci else None
        if conf_int is not None:
            conf_int.index = test_target.index
except Exception as exc:  # statsmodels raises several types on bad orders
    st.error(f"Model failed to fit with order {order} for {firm}: {exc}")
    st.stop()

comparison = pd.DataFrame({
    "Year": test_target.index.year,
    "Actual_Investment": test_target.to_numpy(),
    "Predicted_Investment": np.asarray(forecast, dtype=float),
})
comparison["Error"] = comparison["Actual_Investment"] - comparison["Predicted_Investment"]
comparison["Abs_Error"] = comparison["Error"].abs()
if conf_int is not None:
    comparison["CI_Lower"] = conf_int.iloc[:, 0].to_numpy()
    comparison["CI_Upper"] = conf_int.iloc[:, 1].to_numpy()

mae = mean_absolute_error(test_target, forecast)
rmse = float(np.sqrt(mean_squared_error(test_target, forecast)))

m1, m2, m3 = st.columns(3)
m1.metric("MAE", f"{mae:.3f}", help="Mean Absolute Error on the test years (1947 dollars).")
m2.metric("RMSE", f"{rmse:.3f}", help="Root Mean Squared Error on the test years.")
m3.metric("AIC / BIC", f"{result.aic:.1f} / {result.bic:.1f}")

st.dataframe(
    comparison.style.format({
        "Actual_Investment": "{:,.2f}", "Predicted_Investment": "{:,.2f}",
        "Error": "{:,.2f}", "Abs_Error": "{:,.2f}",
        "CI_Lower": "{:,.2f}", "CI_Upper": "{:,.2f}",
    }),
    use_container_width=True, hide_index=True,
)
st.download_button(
    "Download forecast (CSV)",
    comparison.to_csv(index=False).encode(),
    file_name=f"arimax_forecast_{firm.replace(' ', '_')}.csv",
    mime="text/csv",
)

# Step 9
st.plotly_chart(
    actual_vs_predicted_chart(train_target, test_target, forecast, conf_int, firm),
    use_container_width=True,
)

# --------------------------------------------------------------------------
# Diagnostics + data
# --------------------------------------------------------------------------
tab_diag, tab_data = st.tabs(["Diagnostics", "Data"])

with tab_diag:
    st.plotly_chart(residuals_chart(result.resid), use_container_width=True)
    resid = pd.Series(result.resid).dropna()
    d1, d2, d3 = st.columns(3)
    d1.metric("Residual mean", f"{resid.mean():.3f}")
    d2.metric("Residual std", f"{resid.std():.3f}")
    d3.metric("Log-likelihood", f"{result.llf:.2f}")
    with st.expander("Full model summary (statsmodels)"):
        st.text(result.summary().as_text())

with tab_data:
    st.subheader(f"{firm} — prepared annual series")
    display = company_df.reset_index().rename(columns={"Date": "Date"})
    display["Date"] = pd.to_datetime(display["Date"]).dt.date
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.subheader("Raw dataset (all firms)")
    st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns · "
               f"Missing: {int(df.isnull().sum().sum())} cells")
    st.dataframe(df, use_container_width=True, hide_index=True)
    with st.expander("Dataset summary"):
        st.write(df.describe(include="all").T)
        st.write("Missing values by column:")
        st.write(df.isnull().sum())

st.divider()
st.caption(
    "Notebook defaults: General Motors · exog Firm_Value + Capital_Stock · "
    "ARIMA(1, 1, 0) · test = last 4 years. Change them in the sidebar to explore."
)
