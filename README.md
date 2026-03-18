# 📈 Fed Interest Rate Changes & Financial Sector Stock Performance

A machine learning project that explores whether Federal Reserve interest rate decisions can predict short-term stock performance for major financial institutions.

## Team
Isaac Toffel · Alan Chau · Julian Antropow de la Hoz · Lucas Azout  

---

## Overview

This project combines macroeconomic policy data from the Federal Reserve with equity market data to predict whether financial sector stocks will outperform or underperform the S&P 500 in the 30 days following an FOMC rate decision.

---

## Data Sources

- **[FRED API](https://fred.stlouisfed.org/)** — Federal funds rate history and FOMC meeting dates
- **[Yahoo Finance (yfinance)](https://pypi.org/project/yfinance/)** — Daily stock prices for JPM, GS, BAC, WFC, C, MS, and SPY

---

## Project Structure

```
fed-rates-financial-stocks/
├── src/
│   ├── data_pipeline.py   # Data collection → PostgreSQL
│   ├── features.py        # Feature engineering
│   ├── model.py           # Model training & evaluation
│   └── app.py             # Streamlit dashboard
├── notebooks/             # Exploratory analysis
├── data/                  # Saved model (gitignored)
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup

### 1. Clone the repo

```bash
git clone <repository-url>
cd fed-rates-financial-stocks
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up PostgreSQL

```bash
psql -U postgres -h localhost
```
```sql
CREATE DATABASE fed_rates_db;
\q
```

### 5. Configure environment variables

```bash
cp .env.example .env
```

```
FRED_API_KEY=your_fred_api_key
DB_HOST=localhost
DB_PORT=5432
DB_NAME=fed_rates_db
DB_USER=postgres
DB_PASSWORD=your_postgres_password
```

Get a free FRED API key [here](https://fred.stlouisfed.org/docs/api/api_key.html).

---

## Usage

```bash
# Collect data
python src/data_pipeline.py

# Engineer features
python src/features.py

# Train model
python src/model.py

# Launch dashboard
streamlit run src/app.py
```

---

## Results

Three classifiers were trained and evaluated using 5-fold cross-validation. Gradient Boosting performed best with an F1-score of 0.661.

The most predictive features were the rate level before the meeting, pre-meeting stock volatility, and 30-day momentum — suggesting that the broader rate environment matters more than the size or direction of any individual rate change.

---

## Tech Stack

Python · pandas · scikit-learn · yfinance · Streamlit · Plotly · PostgreSQL

---

## License

MIT