"""
Purpose:
Interactive web dashboard for exploring historical data
and generating predictions using the trained model.

Built with Streamlit.

Responsibilities:
- Load processed data and trained model
- Visualize rate decisions and stock performance
- Allow users to test hypothetical scenarios
- Display model predictions and insights
"""


"""
app.py
Interactive Streamlit dashboard for exploring Fed rate changes,
stock performance, and ML model predictions.
"""

import os
import pickle
import psycopg2
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fed Rates & Financial Stocks",
    page_icon="📈",
    layout="wide",
)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "fed_rates_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "best_model.pkl")

TICKERS = ["JPM", "GS", "BAC", "WFC", "C", "MS"]

CAT_MAPPINGS = {
    "direction":         {"hike": 1, "cut": -1, "hold": 0},
    "rate_level_regime": {"low": 0, "mid": 1, "high": 2},
}

# ── Data loading (cached) ─────────────────────────────────────────────────────

@st.cache_data
def load_fed_rates():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT date, rate, change_bp FROM fed_rates ORDER BY date",
            conn, parse_dates=["date"]
        )

@st.cache_data
def load_fomc_meetings():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT date, rate, change_bp FROM fomc_meetings ORDER BY date",
            conn, parse_dates=["date"]
        )

@st.cache_data
def load_stock_prices():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT date, ticker, close FROM stock_prices ORDER BY date",
            conn, parse_dates=["date"]
        )

@st.cache_data
def load_features():
    with psycopg2.connect(**DB_CONFIG) as conn:
        return pd.read_sql(
            "SELECT * FROM features ORDER BY fomc_date, ticker",
            conn, parse_dates=["fomc_date"]
        )

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


# ── Helpers ───────────────────────────────────────────────────────────────────

def direction_label(bp):
    if bp > 0:   return "🔺 Hike"
    if bp < 0:   return "🔻 Cut"
    return "⏸ Hold"

def direction_color(bp):
    if bp > 0:   return "#EF4444"
    if bp < 0:   return "#22C55E"
    return "#94A3B8"


# ── Sidebar ───────────────────────────────────────────────────────────────────

st.sidebar.title("📈 Fed Rates Dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["🏠 Overview", "📊 Rate History", "📉 Stock Performance", "🤖 Model Predictions", "🔮 Scenario Testing"]
)
st.sidebar.markdown("---")
st.sidebar.caption("Data: FRED API + Yahoo Finance\nModel: Gradient Boosting Classifier")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════

if page == "🏠 Overview":
    st.title("Fed Interest Rate Changes & Financial Sector Stock Performance")
    st.markdown("*Can Federal Reserve interest rate decisions predict short-term stock outperformance?*")
    st.markdown("---")

    fed     = load_fed_rates()
    fomc    = load_fomc_meetings()
    stocks  = load_stock_prices()
    features = load_features()

    # KPI row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Daily Rate Observations", f"{len(fed):,}")
    col2.metric("FOMC Meetings", f"{len(fomc):,}")
    col3.metric("Stock Price Rows", f"{len(stocks):,}")
    col4.metric("Feature Rows", f"{len(features):,}")

    st.markdown("---")

    # Mini rate chart
    st.subheader("Federal Funds Rate — 2000 to Present")
    fig = px.area(
        fed, x="date", y="rate",
        labels={"rate": "Rate (%)", "date": ""},
        color_discrete_sequence=["#1E2761"],
    )
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Recent FOMC meetings table
    st.subheader("Recent FOMC Meetings")
    recent = fomc.tail(8).sort_values("date", ascending=False).copy()
    recent["Direction"] = recent["change_bp"].apply(direction_label)
    recent["change_bp"] = recent["change_bp"].apply(lambda x: f"{x:+.0f} bps")
    recent["rate"]      = recent["rate"].apply(lambda x: f"{x:.2f}%")
    recent.columns      = ["Date", "Rate", "Change", "Direction"]
    st.dataframe(recent.set_index("Date"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Rate History
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Rate History":
    st.title("Federal Reserve Rate History")

    fed  = load_fed_rates()
    fomc = load_fomc_meetings()

    # Date range filter
    min_date = fed["date"].min().date()
    max_date = fed["date"].max().date()
    col1, col2 = st.columns(2)
    start = col1.date_input("Start date", value=pd.to_datetime("2008-01-01").date(), min_value=min_date, max_value=max_date)
    end   = col2.date_input("End date",   value=max_date, min_value=min_date, max_value=max_date)

    mask = (fed["date"].dt.date >= start) & (fed["date"].dt.date <= end)
    fed_filtered  = fed[mask]
    fomc_filtered = fomc[(fomc["date"].dt.date >= start) & (fomc["date"].dt.date <= end)]

    # Rate line chart with FOMC markers
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fed_filtered["date"], y=fed_filtered["rate"],
        mode="lines", name="Fed Funds Rate",
        line=dict(color="#1E2761", width=2)
    ))
    # FOMC hike markers
    hikes = fomc_filtered[fomc_filtered["change_bp"] > 0]
    cuts  = fomc_filtered[fomc_filtered["change_bp"] < 0]
    fig.add_trace(go.Scatter(
        x=hikes["date"], y=hikes["rate"],
        mode="markers", name="Rate Hike",
        marker=dict(color="#EF4444", size=10, symbol="triangle-up")
    ))
    fig.add_trace(go.Scatter(
        x=cuts["date"], y=cuts["rate"],
        mode="markers", name="Rate Cut",
        marker=dict(color="#22C55E", size=10, symbol="triangle-down")
    ))
    fig.update_layout(
        height=420,
        xaxis_title="", yaxis_title="Rate (%)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig, use_container_width=True)

    # Change distribution
    st.subheader("Distribution of Rate Changes at FOMC Meetings")
    fig2 = px.histogram(
        fomc_filtered, x="change_bp", nbins=20,
        labels={"change_bp": "Change (basis points)"},
        color_discrete_sequence=["#1E2761"]
    )
    fig2.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Raw FOMC table
    with st.expander("View all FOMC meetings in range"):
        show = fomc_filtered.copy().sort_values("date", ascending=False)
        show["Direction"] = show["change_bp"].apply(direction_label)
        st.dataframe(show, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Stock Performance
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📉 Stock Performance":
    st.title("Stock Performance Around FOMC Meetings")

    features = load_features()
    stocks   = load_stock_prices()

    col1, col2 = st.columns(2)
    ticker  = col1.selectbox("Ticker", TICKERS)
    direction_filter = col2.selectbox("Rate direction", ["All", "Hike", "Cut", "Hold"])

    df = features[features["ticker"] == ticker].copy()
    if direction_filter != "All":
        df = df[df["direction"] == direction_filter.lower()]

    if df.empty:
        st.warning("No data for that combination.")
    else:
        # Win rate
        win_rate = df["outperformed"].mean() * 100
        col1, col2, col3 = st.columns(3)
        col1.metric("Meetings analysed", len(df))
        col2.metric("Outperformed SPY", f"{df['outperformed'].sum()} times")
        col3.metric("Win rate", f"{win_rate:.1f}%")

        # 30-day forward returns scatter
        st.subheader(f"{ticker} vs SPY — 30-Day Returns After Each FOMC Meeting")
        fig = go.Figure()
        colors = ["#22C55E" if o else "#EF4444" for o in df["outperformed"]]
        fig.add_trace(go.Scatter(
            x=df["fomc_date"], y=df["stock_return_30d"] * 100,
            mode="markers+lines", name=ticker,
            marker=dict(color=colors, size=10),
            line=dict(color="#94A3B8", width=1, dash="dot")
        ))
        fig.add_trace(go.Scatter(
            x=df["fomc_date"], y=df["spy_return_30d"] * 100,
            mode="lines", name="SPY",
            line=dict(color="#1E2761", width=2)
        ))
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.update_layout(
            height=400, xaxis_title="FOMC Date",
            yaxis_title="30-Day Return (%)",
            legend=dict(orientation="h", y=1.1),
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Stock price chart
        st.subheader(f"{ticker} Price History")
        ticker_prices = stocks[stocks["ticker"] == ticker]
        fig2 = px.line(ticker_prices, x="date", y="close",
                       labels={"close": "Price ($)", "date": ""},
                       color_discrete_sequence=["#1E2761"])
        # Mark FOMC dates
        fomc_prices = ticker_prices[ticker_prices["date"].isin(df["fomc_date"])]
        fig2.add_trace(go.Scatter(
            x=fomc_prices["date"], y=fomc_prices["close"],
            mode="markers", name="FOMC Date",
            marker=dict(color="#F5C842", size=8)
        ))
        fig2.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Predictions
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🤖 Model Predictions":
    st.title("ML Model Predictions")

    payload = load_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` first.")
        st.stop()

    model        = payload["model"]
    model_name   = payload["model_name"]
    feature_cols = payload["feature_cols"]
    cat_mappings = payload["cat_mappings"]

    st.success(f"Loaded model: **{model_name}**")

    features = load_features()
    df = features.copy()

    # Encode categoricals
    for col, mapping in cat_mappings.items():
        df[col] = df[col].map(mapping)
    df = df.dropna(subset=feature_cols)

    X = df[feature_cols].astype(float)
    df["predicted"] = model.predict(X)
    df["confidence"] = model.predict_proba(X)[:, 1]

    # Re-attach string labels
    features_display = features.copy()
    features_display["predicted"]  = df["predicted"].values
    features_display["confidence"] = df["confidence"].values
    features_display["correct"]    = (df["predicted"].values == features_display["outperformed"].values)

    # Overall accuracy
    acc = features_display["correct"].mean() * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Accuracy", f"{acc:.1f}%")
    col2.metric("Samples", len(features_display))
    col3.metric("Model", model_name)

    # Feature importance bar chart
    st.subheader("Feature Importance")
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    else:
        importances = np.abs(clf.coef_[0])

    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": importances
    }).sort_values("Importance", ascending=True)

    fig = px.bar(
        imp_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale=["#CADCFC", "#1E2761"],
        labels={"Importance": "Importance Score"}
    )
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # Predictions table
    st.subheader("Predictions by Meeting & Ticker")
    ticker_filter = st.selectbox("Filter by ticker", ["All"] + TICKERS)
    show = features_display.copy()
    if ticker_filter != "All":
        show = show[show["ticker"] == ticker_filter]

    show["Predicted"] = show["predicted"].apply(lambda x: "✅ Outperform" if x == 1 else "❌ Underperform")
    show["Actual"]    = show["outperformed"].apply(lambda x: "✅ Outperform" if x == 1 else "❌ Underperform")
    show["Correct"]   = show["correct"].apply(lambda x: "✓" if x else "✗")
    show["Confidence"] = show["confidence"].apply(lambda x: f"{x:.1%}")

    cols_to_show = ["fomc_date", "ticker", "direction", "change_bp", "Predicted", "Actual", "Correct", "Confidence"]
    st.dataframe(show[cols_to_show].rename(columns={
        "fomc_date": "FOMC Date", "ticker": "Ticker",
        "direction": "Direction", "change_bp": "Change (bps)"
    }).set_index("FOMC Date"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Scenario Testing
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🔮 Scenario Testing":
    st.title("Scenario Testing")
    st.markdown("Adjust the inputs below to simulate a hypothetical FOMC meeting and see what the model predicts.")

    payload = load_model()
    if payload is None:
        st.error("No trained model found. Run `python src/model.py` first.")
        st.stop()

    model        = payload["model"]
    feature_cols = payload["feature_cols"]
    cat_mappings = payload["cat_mappings"]

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 FOMC Meeting Inputs")
        rate_before    = st.slider("Current Fed funds rate (%)", 0.0, 10.0, 5.25, step=0.25)
        direction      = st.selectbox("Rate decision", ["hike", "cut", "hold"])
        change_bp_abs  = st.slider("Size of change (basis points)", 0, 100, 25, step=25)
        change_bp      = change_bp_abs if direction == "hike" else (-change_bp_abs if direction == "cut" else 0)

        if rate_before < 2:
            regime = "low"
        elif rate_before <= 4:
            regime = "mid"
        else:
            regime = "high"
        st.caption(f"Rate regime: **{regime}**")

    with col2:
        st.subheader("📉 Stock Inputs (Pre-Meeting)")
        ticker         = st.selectbox("Ticker", TICKERS)
        pre_return_10d = st.slider("10-day return before meeting (%)", -20.0, 20.0, 0.0, step=0.5) / 100
        pre_return_30d = st.slider("30-day return before meeting (%)", -30.0, 30.0, 0.0, step=0.5) / 100
        pre_vol_30d    = st.slider("30-day annualised volatility (%)", 5.0, 80.0, 25.0, step=1.0) / 100
        pre_rel_10d    = st.slider("Relative return vs SPY — 10d (%)", -15.0, 15.0, 0.0, step=0.5) / 100
        pre_rel_30d    = st.slider("Relative return vs SPY — 30d (%)", -20.0, 20.0, 0.0, step=0.5) / 100

    st.markdown("---")

    # Build input vector
    input_data = {
        "rate_before":        rate_before,
        "change_bp":          change_bp,
        "abs_change_bp":      abs(change_bp),
        "pre_return_10d":     pre_return_10d,
        "pre_return_30d":     pre_return_30d,
        "pre_volatility_30d": pre_vol_30d,
        "pre_rel_return_10d": pre_rel_10d,
        "pre_rel_return_30d": pre_rel_30d,
        "direction":          cat_mappings["direction"][direction],
        "rate_level_regime":  cat_mappings["rate_level_regime"][regime],
    }
    X_input = pd.DataFrame([input_data])[feature_cols].astype(float)

    prob        = model.predict_proba(X_input)[0][1]
    prediction  = model.predict(X_input)[0]

    # Result display
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if prediction == 1:
            st.success(f"### ✅ {ticker} will OUTPERFORM SPY")
        else:
            st.error(f"### ❌ {ticker} will UNDERPERFORM SPY")
        st.markdown(f"**Confidence: {prob:.1%}**")
        st.progress(prob)
        st.caption("Probability that the stock outperforms SPY in the 30 days following the FOMC meeting.")

    # Probability gauge
    st.markdown("---")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={"text": "Outperformance Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#1E2761"},
            "steps": [
                {"range": [0, 40],  "color": "#FEE2E2"},
                {"range": [40, 60], "color": "#FEF9C3"},
                {"range": [60, 100],"color": "#DCFCE7"},
            ],
            "threshold": {
                "line": {"color": "#1E2761", "width": 4},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=350, margin=dict(l=40, r=40, t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)