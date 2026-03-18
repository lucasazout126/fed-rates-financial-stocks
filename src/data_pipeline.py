"""
Purpose:
Collect and store all raw data used in the project.

This script connects to:
- FRED API for Federal Funds Rate + FOMC meeting data
- yfinance for daily stock prices

Responsibilities:
- Fetch macroeconomic and stock data
- Handle API rate limits and errors
- Clean and standardize formats
- Save results to PostgreSQL
"""

import os
import requests
import psycopg2
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────

FRED_API_KEY = os.getenv("FRED_API_KEY")

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     os.getenv("DB_PORT", "5432"),
    "dbname":   os.getenv("DB_NAME", "fed_rates_db"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD"),
}

TICKERS = ["JPM", "GS", "BAC", "WFC", "C", "MS", "SPY"]   # SPY = S&P 500 proxy

# ── Database helpers ──────────────────────────────────────────────────────────

def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def create_tables():
    """Create all tables if they don't already exist."""
    ddl = """
    CREATE TABLE IF NOT EXISTS fed_rates (
        date        DATE PRIMARY KEY,
        rate        NUMERIC(6, 4),
        change_bp   NUMERIC(8, 4)   -- change in basis points vs previous row
    );

    CREATE TABLE IF NOT EXISTS fomc_meetings (
        date        DATE PRIMARY KEY,
        rate        NUMERIC(6, 4),
        change_bp   NUMERIC(8, 4)
    );

    CREATE TABLE IF NOT EXISTS stock_prices (
        date        DATE,
        ticker      VARCHAR(10),
        open        NUMERIC(12, 4),
        high        NUMERIC(12, 4),
        low         NUMERIC(12, 4),
        close       NUMERIC(12, 4),
        volume      BIGINT,
        PRIMARY KEY (date, ticker)
    );
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    print("✅ Tables ready.")


# ── FRED data ─────────────────────────────────────────────────────────────────

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """Download a FRED time series and return a dated DataFrame."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id":      series_id,
        "api_key":        FRED_API_KEY,
        "file_type":      "json",
        "observation_start": "2000-01-01",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()["observations"]
    df = pd.DataFrame(data)[["date", "value"]].copy()
    df["date"]  = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df.dropna(inplace=True)
    return df


def load_fed_rates():
    """Fetch the effective federal funds rate and store in fed_rates."""
    print("Fetching FRED: federal funds rate …")
    df = fetch_fred_series("DFF")               # Daily effective fed funds rate
    df.rename(columns={"value": "rate"}, inplace=True)
    df["change_bp"] = (df["rate"] - df["rate"].shift(1)) * 100  # % → basis points
    df.dropna(inplace=True)

    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                cur.execute(
                    """
                    INSERT INTO fed_rates (date, rate, change_bp)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (date) DO UPDATE
                        SET rate = EXCLUDED.rate,
                            change_bp = EXCLUDED.change_bp
                    """,
                    (row["date"].date(), row["rate"], row["change_bp"]),
                )
        conn.commit()
    print(f"  → {len(df)} rows saved to fed_rates.")


def load_fomc_meetings():
    """
    Derive FOMC meeting dates as the days where the fed funds rate changed,
    then store in fomc_meetings.
    """
    print("Deriving FOMC meeting dates …")
    df = fetch_fred_series("DFEDTARU")          # Fed funds target rate (upper bound)
    df.rename(columns={"value": "rate"}, inplace=True)
    df["change_bp"] = (df["rate"] - df["rate"].shift(1)) * 100
    meetings = df[df["change_bp"] != 0].dropna().copy()

    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in meetings.iterrows():
                cur.execute(
                    """
                    INSERT INTO fomc_meetings (date, rate, change_bp)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (date) DO UPDATE
                        SET rate = EXCLUDED.rate,
                            change_bp = EXCLUDED.change_bp
                    """,
                    (row["date"].date(), row["rate"], row["change_bp"]),
                )
        conn.commit()
    print(f"  → {len(meetings)} FOMC meetings saved.")


# ── yfinance stock data ───────────────────────────────────────────────────────

def load_stock_prices():
    """Fetch all tickers from Yahoo Finance (free, no key needed) and store in stock_prices."""
    print(f"Fetching stock prices for: {', '.join(TICKERS)} …")
    raw = yf.download(TICKERS, start="2000-01-01", auto_adjust=True, progress=False)

    # yfinance returns a MultiIndex DataFrame — flatten it
    prices = raw["Close"].copy()
    opens  = raw["Open"].copy()
    highs  = raw["High"].copy()
    lows   = raw["Low"].copy()
    vols   = raw["Volume"].copy()

    with get_connection() as conn:
        with conn.cursor() as cur:
            for ticker in TICKERS:
                if ticker not in prices.columns:
                    print(f"  ⚠️  No data found for {ticker}, skipping.")
                    continue
                ticker_df = pd.DataFrame({
                    "date":   prices.index,
                    "open":   opens[ticker].values,
                    "high":   highs[ticker].values,
                    "low":    lows[ticker].values,
                    "close":  prices[ticker].values,
                    "volume": vols[ticker].values,
                }).dropna()

                for _, row in ticker_df.iterrows():
                    cur.execute(
                        """
                        INSERT INTO stock_prices (date, ticker, open, high, low, close, volume)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, ticker) DO UPDATE
                            SET open   = EXCLUDED.open,
                                high   = EXCLUDED.high,
                                low    = EXCLUDED.low,
                                close  = EXCLUDED.close,
                                volume = EXCLUDED.volume
                        """,
                        (row["date"].date(), ticker,
                         row["open"], row["high"], row["low"], row["close"], int(row["volume"])),
                    )
                print(f"  → {len(ticker_df)} rows saved for {ticker}.")
        conn.commit()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Fed Rates Financial Stocks — Data Pipeline ===\n")
    create_tables()
    load_fed_rates()
    load_fomc_meetings()
    load_stock_prices()
    print("\n✅ Pipeline complete.")