"""
Purpose:
Create model-ready features from raw economic and stock data.

This script transforms raw price and rate data into signals that
machine learning models can use.

Responsibilities:
- Calculate rate change magnitude and direction
- Compute rolling returns and volatility windows
- Generate technical indicators
- Build target labels (outperform vs underperform)
- Output a final training dataset
"""

"""
features.py
Engineers predictive features and target labels from the raw database tables,
then stores them in a features table ready for model training.
"""

import os
import psycopg2
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "fed_rates_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

TICKERS = ["JPM", "GS", "BAC", "WFC", "C", "MS"]  # excludes SPY (used as benchmark)
FORWARD_WINDOW = 30   # days after FOMC meeting to measure performance

# ── Database helpers ──────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_features_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS features (
        fomc_date           DATE,
        ticker              VARCHAR(10),

        -- Macro / rate features
        rate_before         NUMERIC(6, 4),   -- fed funds rate on meeting day
        change_bp           NUMERIC(8, 4),   -- size of rate change in basis points
        direction           VARCHAR(10),     -- 'hike', 'cut', or 'hold'
        abs_change_bp       NUMERIC(8, 4),   -- absolute value of change
        rate_level_regime   VARCHAR(10),     -- 'low' (<2%), 'mid' (2–4%), 'high' (>4%)

        -- Pre-meeting stock features
        pre_return_10d      NUMERIC(10, 6),  -- stock return 10 days before meeting
        pre_return_30d      NUMERIC(10, 6),  -- stock return 30 days before meeting
        pre_volatility_30d  NUMERIC(10, 6),  -- rolling std of daily returns (30d)
        pre_rel_return_10d  NUMERIC(10, 6),  -- stock return vs SPY, 10d before
        pre_rel_return_30d  NUMERIC(10, 6),  -- stock return vs SPY, 30d before

        -- Target variable
        stock_return_30d    NUMERIC(10, 6),  -- stock return in 30d after meeting
        spy_return_30d      NUMERIC(10, 6),  -- SPY return in 30d after meeting
        outperformed        SMALLINT,        -- 1 if stock > SPY, 0 otherwise

        PRIMARY KEY (fomc_date, ticker)
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    print("✅ Features table ready.")


# ── Load raw data ─────────────────────────────────────────────────────────────

def load_data():
    with get_connection() as conn:
        fomc = pd.read_sql(
            "SELECT date, rate, change_bp FROM fomc_meetings ORDER BY date",
            conn, parse_dates=["date"]
        )
        prices = pd.read_sql(
            "SELECT date, ticker, close FROM stock_prices ORDER BY date",
            conn, parse_dates=["date"]
        )
        fed = pd.read_sql(
            "SELECT date, rate FROM fed_rates ORDER BY date",
            conn, parse_dates=["date"]
        )
    return fomc, prices, fed


# ── Feature engineering ───────────────────────────────────────────────────────

def compute_returns(close_series: pd.Series) -> pd.Series:
    """Daily log returns."""
    return np.log(close_series / close_series.shift(1))


def get_forward_return(prices_series: pd.Series, meeting_date, window: int):
    """Return over the next `window` trading days after meeting_date."""
    future = prices_series[prices_series.index > meeting_date].head(window)
    if len(future) < window * 0.8:   # need at least 80% of days
        return np.nan
    return (future.iloc[-1] / prices_series.get(meeting_date, np.nan)) - 1


def get_pre_return(prices_series: pd.Series, meeting_date, window: int):
    """Return over the `window` trading days before meeting_date."""
    past = prices_series[prices_series.index <= meeting_date].tail(window + 1)
    if len(past) < 2:
        return np.nan
    return (past.iloc[-1] / past.iloc[0]) - 1


def get_pre_volatility(prices_series: pd.Series, meeting_date, window: int):
    """Annualised volatility of daily returns over the `window` days before meeting."""
    past = prices_series[prices_series.index <= meeting_date].tail(window + 1)
    returns = compute_returns(past).dropna()
    if len(returns) < window * 0.5:
        return np.nan
    return returns.std() * np.sqrt(252)


def rate_regime(rate):
    if rate < 2:
        return "low"
    elif rate <= 4:
        return "mid"
    else:
        return "high"


def engineer_features(fomc, prices, fed):
    # Pivot prices to wide format: index=date, columns=ticker
    wide = prices.pivot(index="date", columns="ticker", values="close")
    fed_indexed = fed.set_index("date")["rate"]

    records = []

    for _, meeting in fomc.iterrows():
        meeting_date = meeting["date"]
        change_bp    = float(meeting["change_bp"])

        # Rate on/before meeting day
        past_rates = fed_indexed[fed_indexed.index <= meeting_date]
        if past_rates.empty:
            continue
        rate_before = float(past_rates.iloc[-1])

        # SPY features (benchmark)
        if "SPY" not in wide.columns:
            continue
        spy_fwd = get_forward_return(wide["SPY"], meeting_date, FORWARD_WINDOW)
        if np.isnan(spy_fwd):
            continue

        spy_pre10 = get_pre_return(wide["SPY"], meeting_date, 10)
        spy_pre30 = get_pre_return(wide["SPY"], meeting_date, 30)

        for ticker in TICKERS:
            if ticker not in wide.columns:
                continue

            stock_fwd  = get_forward_return(wide[ticker], meeting_date, FORWARD_WINDOW)
            pre_ret10  = get_pre_return(wide[ticker], meeting_date, 10)
            pre_ret30  = get_pre_return(wide[ticker], meeting_date, 30)
            pre_vol30  = get_pre_volatility(wide[ticker], meeting_date, 30)

            if np.isnan(stock_fwd) or np.isnan(pre_ret10) or np.isnan(pre_ret30):
                continue

            records.append({
                "fomc_date":          meeting_date.date(),
                "ticker":             ticker,
                "rate_before":        rate_before,
                "change_bp":          change_bp,
                "direction":          "hike" if change_bp > 0 else ("cut" if change_bp < 0 else "hold"),
                "abs_change_bp":      abs(change_bp),
                "rate_level_regime":  rate_regime(rate_before),
                "pre_return_10d":     pre_ret10,
                "pre_return_30d":     pre_ret30,
                "pre_volatility_30d": pre_vol30,
                "pre_rel_return_10d": (pre_ret10 or 0) - (spy_pre10 or 0),
                "pre_rel_return_30d": (pre_ret30 or 0) - (spy_pre30 or 0),
                "stock_return_30d":   stock_fwd,
                "spy_return_30d":     spy_fwd,
                "outperformed":       1 if stock_fwd > spy_fwd else 0,
            })

    return pd.DataFrame(records)


# ── Save to database ──────────────────────────────────────────────────────────

def save_features(df: pd.DataFrame):
    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO features (
                        fomc_date, ticker,
                        rate_before, change_bp, direction, abs_change_bp, rate_level_regime,
                        pre_return_10d, pre_return_30d, pre_volatility_30d,
                        pre_rel_return_10d, pre_rel_return_30d,
                        stock_return_30d, spy_return_30d, outperformed
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (fomc_date, ticker) DO UPDATE SET
                        rate_before        = EXCLUDED.rate_before,
                        change_bp          = EXCLUDED.change_bp,
                        direction          = EXCLUDED.direction,
                        abs_change_bp      = EXCLUDED.abs_change_bp,
                        rate_level_regime  = EXCLUDED.rate_level_regime,
                        pre_return_10d     = EXCLUDED.pre_return_10d,
                        pre_return_30d     = EXCLUDED.pre_return_30d,
                        pre_volatility_30d = EXCLUDED.pre_volatility_30d,
                        pre_rel_return_10d = EXCLUDED.pre_rel_return_10d,
                        pre_rel_return_30d = EXCLUDED.pre_rel_return_30d,
                        stock_return_30d   = EXCLUDED.stock_return_30d,
                        spy_return_30d     = EXCLUDED.spy_return_30d,
                        outperformed       = EXCLUDED.outperformed
                    """,
                    (
                        row["fomc_date"], row["ticker"],
                        row["rate_before"], row["change_bp"], row["direction"],
                        row["abs_change_bp"], row["rate_level_regime"],
                        row["pre_return_10d"], row["pre_return_30d"], row["pre_volatility_30d"],
                        row["pre_rel_return_10d"], row["pre_rel_return_30d"],
                        row["stock_return_30d"], row["spy_return_30d"], row["outperformed"],
                    ),
                )
        conn.commit()
    print(f"  → {len(df)} feature rows saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Feature Engineering ===\n")
    create_features_table()

    print("Loading raw data from database …")
    fomc, prices, fed = load_data()
    print(f"  → {len(fomc)} FOMC meetings, {len(prices)} price rows loaded.")

    print("Engineering features …")
    features_df = engineer_features(fomc, prices, fed)
    print(f"  → {len(features_df)} feature rows created.")
    print(f"\nClass balance (outperformed):\n{features_df['outperformed'].value_counts().to_string()}")
    print(f"\nSample:\n{features_df.head(3).to_string()}")

    print("\nSaving to database …")
    save_features(features_df)

    print("\n✅ Feature engineering complete.")