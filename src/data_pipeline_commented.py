"""
===========================================================================================
DATA PIPELINE — data_pipeline.py
===========================================================================================

WHAT THIS SCRIPT DOES (the big picture):
    This is the very first step of the entire project. Before we can do any analysis
    or build any AI model, we need data. This script goes out to two sources on the
    internet, downloads the data, cleans it up, and saves it into our database.

    Think of it like a shopping run before you cook a meal. Nothing else works
    until this step is done.

TWO DATA SOURCES:
    1. FRED API  — The Federal Reserve's free public database. We use it to get
                   26 years of interest rate history and to figure out exactly
                   which days the Fed changed rates (FOMC meeting dates).

    2. yfinance  — A free Python library that pulls stock price history from
                   Yahoo Finance. No API key needed. We use it for 6 bank stocks
                   plus SPY (the S&P 500 ETF, which is our market benchmark).

OUTPUT:
    Three tables saved in a PostgreSQL database:
        - fed_rates      → every daily interest rate from 2000 to today
        - fomc_meetings  → only the 31 days when the Fed actually changed rates
        - stock_prices   → daily prices for all 7 tickers going back to 2000
===========================================================================================
"""

# ------------------------------------------------------------------------------------------
# IMPORTS — grabbing the tools we need before we start
# ------------------------------------------------------------------------------------------

import os           # lets us read files and secret keys from the computer's environment
import requests     # lets us make web requests (like a browser, but in code)
import psycopg2     # the connector between Python and our PostgreSQL database
import pandas as pd # the main tool for organizing data into tables (like Excel in code)
import yfinance as yf  # free library for downloading stock prices from Yahoo Finance
from dotenv import load_dotenv  # reads our .env file so passwords stay out of the code

# This line reads our .env file and loads our API key + database password into memory.
# The .env file is never committed to GitHub — it stays private on our machine.
load_dotenv()


# ------------------------------------------------------------------------------------------
# CONFIGURATION — all settings in one place at the top
# ------------------------------------------------------------------------------------------

# Our FRED API key, loaded from the .env file.
# FRED (Federal Reserve Economic Data) is a free government service.
# You register for a free key at fred.stlouisfed.org.
# We never write the actual key here — that would be a security risk.
FRED_API_KEY = os.getenv("FRED_API_KEY")

# All the info needed to connect to our PostgreSQL database.
# os.getenv("KEY", "default") means: read from .env, and if not found, use the default.
DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),   # where the database lives (our own machine)
    "port":     os.getenv("DB_PORT", "5432"),        # the "door number" PostgreSQL listens on
    "dbname":   os.getenv("DB_NAME", "fed_rates_db"),# the name of our specific database
    "user":     os.getenv("DB_USER", "postgres"),    # the database username
    "password": os.getenv("DB_PASSWORD"),            # the database password (from .env)
}

# The 7 stock tickers we care about.
# The first 6 are major financial institutions. SPY is the S&P 500 ETF —
# it represents the overall market and is our benchmark to beat.
TICKERS = ["JPM", "GS", "BAC", "WFC", "C", "MS", "SPY"]


# ------------------------------------------------------------------------------------------
# DATABASE HELPERS — reusable functions for connecting and setting up the database
# ------------------------------------------------------------------------------------------

def get_connection():
    """
    Opens and returns a live connection to our PostgreSQL database.
    Using ** unpacks the DB_CONFIG dictionary as keyword arguments,
    so it's the same as writing psycopg2.connect(host=..., port=..., etc.)
    """
    return psycopg2.connect(**DB_CONFIG)


def create_tables():
    """
    Creates our three database tables — but only if they don't already exist.
    The 'IF NOT EXISTS' part makes this safe to run multiple times without error.

    Think of this like setting up three empty filing cabinets before you start
    storing documents. You only need to do this once, but it's safe to call again.
    """
    # DDL = Data Definition Language — the SQL commands that define structure, not data.
    # This block of SQL creates all three tables in one shot.
    ddl = """
    CREATE TABLE IF NOT EXISTS fed_rates (
        date        DATE PRIMARY KEY,   -- one row per calendar day, no duplicates
        rate        NUMERIC(6, 4),      -- the interest rate (e.g. 5.3300)
        change_bp   NUMERIC(8, 4)       -- how much it changed vs the day before (in basis points)
    );

    CREATE TABLE IF NOT EXISTS fomc_meetings (
        date        DATE PRIMARY KEY,   -- only days when the Fed actually changed rates
        rate        NUMERIC(6, 4),      -- the new rate after the decision
        change_bp   NUMERIC(8, 4)       -- how many basis points it moved (+ = hike, - = cut)
    );

    CREATE TABLE IF NOT EXISTS stock_prices (
        date        DATE,               -- the trading day
        ticker      VARCHAR(10),        -- the stock symbol, e.g. "JPM"
        open        NUMERIC(12, 4),     -- price at market open
        high        NUMERIC(12, 4),     -- highest price of the day
        low         NUMERIC(12, 4),     -- lowest price of the day
        close       NUMERIC(12, 4),     -- price at market close (what we use most)
        volume      BIGINT,             -- how many shares were traded that day
        PRIMARY KEY (date, ticker)      -- a row is unique by date + ticker together
    );
    """
    # Open a connection, run the SQL, then commit (save) the changes.
    # The 'with' keyword ensures the connection is closed cleanly even if an error occurs.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(ddl)
        conn.commit()
    print("✅ Tables ready.")


# ------------------------------------------------------------------------------------------
# FRED DATA — fetching interest rate history from the Federal Reserve
# ------------------------------------------------------------------------------------------

def fetch_fred_series(series_id: str) -> pd.DataFrame:
    """
    This is the core function for talking to the FRED API.
    It takes a series ID (a code that tells FRED which dataset we want),
    makes a web request, and returns a clean pandas DataFrame.

    We call this function twice:
        - Once with "DFF"      → the daily effective federal funds rate
        - Once with "DFEDTARU" → the Fed's target rate upper bound
    """

    # The base URL for FRED's API endpoint (their "front door" for data requests)
    url = "https://api.stlouisfed.org/fred/series/observations"

    # These are the parameters we attach to the request — like search filters.
    # FRED reads these and knows exactly what data to send back.
    params = {
        "series_id":         series_id,      # which dataset to fetch (e.g. "DFF")
        "api_key":           FRED_API_KEY,   # our key proving we're a registered user
        "file_type":         "json",         # we want the response in JSON format
        "observation_start": "2000-01-01",   # only data from the year 2000 onwards
    }

    # Make the actual web request. timeout=30 means: if FRED doesn't respond
    # within 30 seconds, give up and raise an error (don't hang forever).
    resp = requests.get(url, params=params, timeout=30)

    # Check if the request succeeded. If FRED returned an error (like 404 or 500),
    # this line raises an exception immediately so we know something went wrong.
    resp.raise_for_status()

    # FRED sends back JSON — a structured text format. We parse it and pull out
    # just the "observations" list (the actual data rows).
    data = resp.json()["observations"]

    # Convert the list of dictionaries into a pandas DataFrame (a table),
    # keeping only the "date" and "value" columns.
    df = pd.DataFrame(data)[["date", "value"]].copy()

    # The dates come back as plain text strings like "2023-07-26".
    # pd.to_datetime converts them into real date objects so we can do math with them.
    df["date"] = pd.to_datetime(df["date"])

    # The values also come back as text. pd.to_numeric converts them to numbers.
    # errors="coerce" means: if a value can't be converted (e.g. "." for missing data),
    # turn it into NaN (Not a Number) instead of crashing.
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    # Drop any rows where the value is NaN — we don't want missing data in the database.
    df.dropna(inplace=True)

    return df


def load_fed_rates():
    """
    Downloads the daily effective federal funds rate from FRED
    and saves all 9,500+ rows into the fed_rates table.

    Also calculates how much the rate changed each day (in basis points).
    A basis point = 0.01%, so a 0.25% change = 25 basis points.
    This is the standard unit used in finance for rate changes.
    """
    print("Fetching FRED: federal funds rate …")

    # Call our helper above with "DFF" — FRED's code for the daily fed funds rate
    df = fetch_fred_series("DFF")

    # Rename the generic "value" column to something meaningful
    df.rename(columns={"value": "rate"}, inplace=True)

    # Calculate the day-over-day change in basis points.
    # .shift(1) means "the value from the previous row" — subtracting gives us the change.
    # Multiply by 100 to convert from percentage points to basis points.
    # Example: rate goes from 5.25% to 5.50% → change = 0.25 × 100 = 25 basis points
    df["change_bp"] = (df["rate"] - df["rate"].shift(1)) * 100

    # The very first row has no "previous row" to subtract from, so it becomes NaN. Drop it.
    df.dropna(inplace=True)

    # Now save every row to the database.
    with get_connection() as conn:
        with conn.cursor() as cur:
            for _, row in df.iterrows():  # loop through every row in the table
                cur.execute(
                    """
                    INSERT INTO fed_rates (date, rate, change_bp)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (date) DO UPDATE
                        SET rate = EXCLUDED.rate,
                            change_bp = EXCLUDED.change_bp
                    """,
                    # The actual values to insert — %s are placeholders (prevents SQL injection)
                    (row["date"].date(), row["rate"], row["change_bp"]),
                )
                # ON CONFLICT means: if a row for this date already exists, UPDATE it
                # instead of throwing an error. This is called an "upsert" and makes
                # the pipeline safe to re-run without creating duplicate rows.
        conn.commit()  # save all the inserts to disk in one transaction
    print(f"  → {len(df)} rows saved to fed_rates.")


def load_fomc_meetings():
    """
    Figures out which days the Fed actually changed rates, and saves those
    31 days into the fomc_meetings table.

    KEY INSIGHT: We never had a hardcoded calendar of FOMC meetings.
    Instead, we DERIVED the meeting dates from the data itself.

    Logic: The Fed only changes its target rate at official FOMC meetings.
    So any day where DFEDTARU (the target rate) changed from the day before
    must be a meeting day. We just filter for those days.

    This automatically gave us 31 meetings: 20 hikes and 11 cuts.
    """
    print("Deriving FOMC meeting dates …")

    # "DFEDTARU" = Fed Funds Target Rate Upper Bound.
    # Unlike DFF (the effective rate), this one only changes on FOMC meeting days.
    df = fetch_fred_series("DFEDTARU")
    df.rename(columns={"value": "rate"}, inplace=True)

    # Same basis point calculation as above
    df["change_bp"] = (df["rate"] - df["rate"].shift(1)) * 100

    # Filter down to ONLY the rows where change_bp is not zero.
    # These are the FOMC meeting days — the only days rates move.
    meetings = df[df["change_bp"] != 0].dropna().copy()

    # Save those meeting rows to the database (same upsert pattern as above)
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


# ------------------------------------------------------------------------------------------
# STOCK PRICE DATA — downloading from Yahoo Finance via yfinance
# ------------------------------------------------------------------------------------------

def load_stock_prices():
    """
    Downloads daily stock price history for all 7 tickers going back to 2000,
    and saves everything into the stock_prices table.

    yfinance is free and requires no API key — it pulls data from Yahoo Finance.
    We download all tickers in a single call for efficiency.
    """
    print(f"Fetching stock prices for: {', '.join(TICKERS)} …")

    # Download all 7 tickers at once in one API call.
    # auto_adjust=True means prices are adjusted for stock splits and dividends,
    # so a stock's price in 2005 is directly comparable to its price today.
    # progress=False just suppresses a loading bar in the terminal.
    raw = yf.download(TICKERS, start="2000-01-01", auto_adjust=True, progress=False)

    # yfinance returns a "MultiIndex DataFrame" — a table with two levels of column headers.
    # The first level is the price type: Open, High, Low, Close, Volume.
    # The second level is the ticker: JPM, GS, BAC, etc.
    #
    # It looks like this:
    #            Close              Open
    #            JPM    GS   BAC   JPM    GS   BAC
    # 2000-01-03  ...   ...  ...   ...   ...   ...
    #
    # We flatten it by pulling out each price type as its own separate table.
    prices = raw["Close"].copy()   # a table of just closing prices, one column per ticker
    opens  = raw["Open"].copy()
    highs  = raw["High"].copy()
    lows   = raw["Low"].copy()
    vols   = raw["Volume"].copy()

    with get_connection() as conn:
        with conn.cursor() as cur:

            # Process one ticker at a time
            for ticker in TICKERS:

                # Safety check: if yfinance somehow didn't return data for a ticker, skip it
                if ticker not in prices.columns:
                    print(f"  ⚠️  No data found for {ticker}, skipping.")
                    continue

                # Build a clean, flat DataFrame for just this one ticker.
                # .values pulls the raw numbers out of each column.
                ticker_df = pd.DataFrame({
                    "date":   prices.index,           # the date index (trading days)
                    "open":   opens[ticker].values,   # open price for this ticker
                    "high":   highs[ticker].values,   # daily high
                    "low":    lows[ticker].values,    # daily low
                    "close":  prices[ticker].values,  # closing price (most important)
                    "volume": vols[ticker].values,    # number of shares traded
                }).dropna()  # remove any rows with missing data

                # Insert every row for this ticker into the database
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
                        # int() on volume because it must be a whole number (no decimals)
                        (row["date"].date(), ticker,
                         row["open"], row["high"], row["low"], row["close"], int(row["volume"])),
                    )
                print(f"  → {len(ticker_df)} rows saved for {ticker}.")

        conn.commit()  # commit all 7 tickers at once in a single transaction


# ------------------------------------------------------------------------------------------
# MAIN — the entry point that ties everything together
# ------------------------------------------------------------------------------------------

# This condition means: only run the code below if someone runs THIS file directly.
# If another script imports this file, this block is skipped.
# It's a Python best practice for keeping scripts modular and importable.
if __name__ == "__main__":
    print("=== Fed Rates Financial Stocks — Data Pipeline ===\n")

    # Step 1: Make sure the database tables exist (safe to run even if they already do)
    create_tables()

    # Step 2: Download ~9,500 rows of daily Fed funds rate history (2000–today)
    load_fed_rates()

    # Step 3: Derive the 31 FOMC meeting dates by detecting when the target rate changed
    load_fomc_meetings()

    # Step 4: Download ~46,000 rows of stock price history across all 7 tickers
    load_stock_prices()

    print("\n✅ Pipeline complete.")
