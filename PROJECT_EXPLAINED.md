# Fed Rates & Financial Stocks — Project Explained in Plain English

---

## What Is This Project? (The Big Picture)

Imagine you are watching the news and you hear: *"The Federal Reserve just raised interest rates by 0.25%."*
You immediately wonder: *"Should I buy or sell bank stocks right now?"*

That is exactly the question this project tries to answer — using data and machine learning (a type of artificial intelligence).

**In one sentence:**
We built a computer program that looks at historical Federal Reserve interest rate decisions and tries to predict whether bank stocks will perform better than the overall stock market in the 30 days after each decision.

---

## Who Built It?

This was a team project built by four people:
- Isaac Toffel
- Alan Chau
- Julian Antropow de la Hoz
- **Lucas Azout** (you)

---

## Why Does This Matter?

The **Federal Reserve** (the Fed) is the central bank of the United States. One of its most powerful tools is setting interest rates — basically the cost of borrowing money. When rates go up, borrowing becomes more expensive. When rates go down, borrowing becomes cheaper.

**Banks and financial companies are uniquely sensitive to interest rates** because:
- They make money by lending out money
- When rates go up, they can charge more on loans
- When rates go down, their profit margins can shrink

So if you could predict *how bank stocks behave after rate decisions*, you might have a small edge as an investor.

---

## The Stocks We Studied

We looked at 6 major financial companies and compared them to the overall market:

| Ticker | Company |
|--------|---------|
| JPM | JPMorgan Chase |
| GS | Goldman Sachs |
| BAC | Bank of America |
| WFC | Wells Fargo |
| C | Citigroup |
| MS | Morgan Stanley |
| SPY | The S&P 500 (represents the overall stock market — our benchmark) |

**SPY is the measuring stick.** If JPMorgan goes up 5% but SPY goes up 7%, JPMorgan actually *lost* to the market. We are not asking "did the stock go up?" — we are asking "did it beat the market?"

---

## The Data We Used

Think of data like ingredients in a recipe. We needed two main ingredients:

### Ingredient 1: Federal Reserve Interest Rate History
- We downloaded 26 years of interest rate data (2000 to 2026) from a website called **FRED** (Federal Reserve Economic Data — a free government database)
- This gave us the rate on every single day and exactly when the Fed changed it
- Over that time period, the Fed held **31 official meetings** where they changed rates:
  - **20 times** they raised rates (called a "hike")
  - **11 times** they lowered rates (called a "cut")

### Ingredient 2: Stock Price History
- We downloaded daily stock prices for all 7 tickers going back to the year 2000
- That gave us **46,137 rows of data** (about 6,591 days per stock)
- For each day, we have the opening price, closing price, high, low, and trading volume

---

## Where Did We Store All This Data?

All this data was stored in a **database** — think of it like a very organized, powerful spreadsheet that can hold millions of rows and be searched instantly.

The database is called **PostgreSQL**. We organized it into three separate tables (like three separate sheets in Excel):

1. **fed_rates** — Every daily interest rate from 2000–2026
2. **fomc_meetings** — Only the 31 days when the rate actually changed
3. **stock_prices** — Daily prices for all 7 stocks

---

## What Are "Features"? (The Inputs to Our AI)

Before we could teach a computer to make predictions, we had to give it useful information — called **features**. Think of features like clues.

For each of the 31 Fed meetings, and for each of the 6 bank stocks, we calculated 10 clues:

### Clues About the Interest Rate Decision:
1. **What was the rate before the meeting?** (e.g., 5.25%)
2. **How much did it change?** (e.g., went down 25 "basis points" — a basis point is 0.01%, so 25 bps = 0.25%)
3. **How big was the change, ignoring direction?** (e.g., 25 bps whether up or down)
4. **Was it a hike, cut, or hold?**
5. **Was the rate generally low, medium, or high?** (Low = below 2%, Medium = 2–4%, High = above 4%)

### Clues About the Stock's Recent Behavior:
6. **How did the stock perform in the 10 days before the meeting?**
7. **How did the stock perform in the 30 days before the meeting?**
8. **How volatile (jumpy) was the stock in the 30 days before the meeting?** (A volatile stock swings up and down a lot. Measured as annualized standard deviation — just a math formula for how much prices bounce around)
9. **Did the stock beat or lose to SPY in the 10 days before the meeting?**
10. **Did the stock beat or lose to SPY in the 30 days before the meeting?**

All 10 clues together gave our AI the information it needed to make a guess.

---

## What Were We Trying to Predict? (The Answer)

We were predicting one simple yes/no question:

> **"In the 30 trading days after the Fed meeting, did this stock beat SPY?"**

- **1 = Yes, it outperformed** (stock did better than the market)
- **0 = No, it underperformed** (stock did worse than the market)

A great coincidence: out of our 186 examples (31 meetings × 6 stocks), exactly **93 were wins and 93 were losses** — a perfect 50/50 split. This is called "balanced classes" and it makes the AI easier to train fairly.

---

## How the AI Was Trained (The Machine Learning Part)

### What is machine learning?

Imagine teaching a child to recognize cats by showing them thousands of pictures labeled "cat" or "not cat." Eventually, the child learns patterns — cats have pointy ears, whiskers, fur. Machine learning works the same way, but with numbers and math instead of pictures.

We showed our AI 186 historical examples (each FOMC meeting × each stock), along with the answer for each one. The AI learned which combinations of clues tend to lead to outperformance.

### We tested 3 different AI "methods":

**Method 1: Logistic Regression**
Think of this like drawing a single straight line through data to separate wins from losses. Simple, fast, but limited — real-world patterns are rarely a straight line.

**Method 2: Random Forest**
Imagine asking 200 different people (each with slightly different information) to vote on an answer. The majority wins. This is more powerful than one straight line.

**Method 3: Gradient Boosting** ← **THIS WON**
Imagine starting with one bad guesser, then hiring a second person whose only job is to fix the first person's mistakes, then a third person to fix the second person's mistakes... and so on, 200 times. Each round learns from the errors of the round before. This is very powerful.

### How did we measure which method was best?

We used a scoring method called **F1-score**, which is a balanced measure of how often the AI was right. A perfect score would be 1.0 (100%). A random guess would score 0.5 (50%).

| Method | F1-Score |
|--------|---------|
| Logistic Regression | 0.618 |
| Random Forest | 0.623 |
| **Gradient Boosting** | **0.661** ✓ |

Gradient Boosting won with an F1 of **0.661** — meaning it correctly predicted about **66% of cases**, meaningfully better than a coin flip.

### How did we make sure we weren't cheating?

This is called **cross-validation**. Instead of training on all the data and testing on the same data (which would be like giving students the answer sheet before the exam), we:

1. Split the 186 examples into 5 groups
2. Trained on 4 groups, tested on the 1 group left out
3. Repeated this 5 times, each time leaving out a different group
4. Averaged the 5 scores

This gives a much more honest picture of how the AI would perform on data it has never seen before.

---

## The Surprising Finding

Here is the most interesting result — and it is **counter-intuitive**:

> **Whether the Fed raised or cut rates barely mattered.**

When we looked at win rates:
- After **rate hikes**: stocks beat SPY about **48%** of the time (basically a coin flip)
- After **rate cuts**: stocks beat SPY about **51%** of the time (also basically a coin flip)

So the direction of the rate change — the thing everyone talks about on the news — was almost useless as a predictor.

**What actually mattered most?**

1. **Pre-meeting volatility** — Stocks that were already moving around a lot before the meeting tended to outperform afterward
2. **Recent momentum vs. the market** — Stocks that had already been beating SPY in the weeks before the meeting tended to keep doing so
3. **The general level of rates** — Whether we were in a low-rate or high-rate environment mattered more than whether the rate went up or down

**The takeaway:** The *size* or *direction* of the Fed's move matters less than the broader environment the stock is already living in.

---

## The Dashboard (The Visual Interface)

We built an interactive website (called a **Streamlit dashboard**) where anyone can explore the data and make predictions without knowing how to code.

It has 5 pages:

### Page 1: Overview
- Shows key statistics at a glance (how many data points, how many meetings, etc.)
- Displays a chart of the Fed's interest rate from 2000 to today
- Shows a table of recent FOMC meetings

### Page 2: Rate History
- An interactive timeline of interest rates
- You can zoom in, zoom out, and filter by date range
- FOMC meetings are marked with arrows (red = hike, green = cut)
- A histogram showing the distribution of rate change sizes

### Page 3: Stock Performance
- Choose a stock (e.g., JPMorgan) and a rate direction (e.g., only hikes)
- See the "win rate" — how often that stock beat SPY after that type of decision
- Scatter plot of each meeting's result (green dot = outperformed, red = underperformed)

### Page 4: Model Predictions
- Shows the AI's predictions for all historical cases
- Displays how confident the AI was in each prediction
- Bar chart showing which features the AI found most useful

### Page 5: Scenario Testing (The Fun One)
- You can create a *hypothetical* FOMC meeting from scratch
- Use sliders to set: the current rate, the size of the hike/cut, the stock's recent momentum, volatility, etc.
- The AI instantly tells you: **"This stock will likely OUTPERFORM"** or **"UNDERPERFORM"**
- Shows a confidence gauge from 0% to 100%

---

## How the Project Runs (Step by Step)

If someone wanted to re-run this project from scratch, they would do it in 4 steps:

**Step 1:** Run the data collector → Downloads all Fed rate data and stock prices into the database (takes about 5 minutes)

**Step 2:** Run the feature builder → Calculates all 10 clues for each of the 186 meeting-stock combinations (takes about 1 minute)

**Step 3:** Run the model trainer → Tests all 3 AI methods, picks the best one, and saves it (takes about 2 minutes)

**Step 4:** Launch the dashboard → Opens the interactive website at a local address (instant)

---

## Limitations (What This Project Cannot Do)

Being honest about limitations shows scientific maturity. Here are ours:

1. **Small dataset** — Only 31 FOMC meetings means only 186 examples. That is a very small amount of data for an AI. More data would make predictions more reliable.

2. **Missing information** — We only used Fed rate data and stock price history. We did not include:
   - Earnings reports (did the company just have a great quarter?)
   - Fear index (VIX — how nervous is the market overall?)
   - Economic recession indicators
   - What the market *expected* the Fed to do (sometimes the rate change is already "priced in")

3. **Past ≠ future** — Our AI learned from 2000–2026. The economy changes, and patterns that held before may not hold in the future.

4. **66% is not a trading strategy** — While better than random, 66% accuracy is not high enough to reliably make money in the stock market, especially after accounting for trading fees and taxes.

---

## The Technology Stack (What Tools We Used)

| Tool | What It Does | Real-World Analogy |
|------|--------------|--------------------|
| Python | The programming language everything is written in | The language we all spoke |
| FRED API | Source of Fed rate data | A government library we called on the phone |
| yfinance | Source of stock price data | A free stock data service |
| PostgreSQL | The database | A giant organized filing cabinet |
| pandas / numpy | Data manipulation tools | Excel, but 1000x more powerful |
| scikit-learn | The machine learning toolkit | A toolbox of AI algorithms |
| Streamlit | The dashboard framework | The website builder |
| Plotly | The charting library | The tool that makes the charts interactive |

---

## Summary: What You Built

You and your team built a **complete, end-to-end data science project** that:

1. **Collected** 26 years of Federal Reserve interest rate data and stock prices from public sources
2. **Stored** it in a professional database
3. **Engineered** 10 meaningful features per observation
4. **Trained** and compared 3 machine learning models using rigorous cross-validation
5. **Discovered** that pre-meeting volatility and momentum matter more than rate direction
6. **Deployed** an interactive dashboard for non-technical users to explore the data and test predictions

The best model (Gradient Boosting) achieves a **66.1% F1-score** — beating a random baseline of 50% by a meaningful margin.

---

## One-Paragraph Summary for Interviews

*"Our project investigated whether Federal Reserve interest rate decisions can predict short-term outperformance of major financial stocks relative to the S&P 500. We collected 26 years of Fed rate data and stock price history, engineered 10 features capturing the rate environment and each stock's pre-meeting momentum and volatility, and trained three classification models on 186 historical observations. Gradient Boosting performed best with a 66% F1-score using 5-fold cross-validation. Surprisingly, the direction of rate changes barely mattered — the strongest predictors were pre-meeting volatility and momentum relative to the market. We deployed an interactive Streamlit dashboard that lets users explore historical results and test hypothetical scenarios with real-time model predictions."*
