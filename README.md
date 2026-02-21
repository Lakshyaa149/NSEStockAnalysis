# NSEStockAnalysis

NSE stock analysis project that combines:

- 30-day company news sentiment
- 30-day CEO commentary sentiment
- Fundamentals screening (PE, forward PE, PEG, ROE, debt/equity, profit margins)
- Promoter-holding proxy (via insider holding field from Yahoo)
- Ranked output with return-potential categories
- Streamlit UI with separate screens for each report

## Features

1. News Sentiment (All NSE symbols from universe file)
- Fetches recent headlines (lookback configurable)
- Scores sentiment per headline
- Creates company-level and CEO-level sentiment summaries

2. Fundamentals + Sentiment Ranking
- Pulls fundamentals for NSE tickers (`.NS`)
- Scores each stock using valuation + quality + sentiment rules
- Labels stocks as:
  - `High Potential`
  - `Watchlist`
  - `Avoid/Needs Work`

3. UI Dashboard
- `Overview`
- `Company News`
- `CEO Commentary`
- `Fundamentals`
- Buttons to create/refresh reports from UI

## Project Structure

```text
NSEStockAnalysis/
  app.py
  configs/default.yaml
  data/
    raw/
      nse_companies.csv
      sample_prices.csv
    processed/
      ...generated csv reports...
  src/
    news/
    fundamentals/
    features/
    models/
    backtest/
    risk/
    main.py
```

## Setup

```bash
cd "/Users/lakshay/Documents/NSEStockAnalysis"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Reports

### 1) Generate 30-day news reports

```bash
python -m src.main news_report --config configs/default.yaml
```

Outputs:

- `data/processed/company_news_sentiment_summary_30d.csv`
- `data/processed/company_news_sentiment_details_30d.csv`
- `data/processed/ceo_commentary_sentiment_summary_30d.csv`
- `data/processed/ceo_commentary_sentiment_details_30d.csv`

### 2) Generate fundamentals ranking report

```bash
python -m src.main fundamentals_report --config configs/default.yaml
```

Outputs:

- `data/processed/fundamentals_raw.csv`
- `data/processed/fundamentals_ranked_report.csv`
- `data/processed/fundamentals_top_picks.csv`

## Run UI

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit (usually `http://localhost:8501`).

## Configuration

Edit `configs/default.yaml` to control:

- `news.lookback_days`
- `news.limit_per_company`
- `news.max_workers`
- `news.universe_csv`
- `fundamentals.max_workers`
- `fundamentals.max_companies`
- output file paths

## Important Notes

- This is a screening/analysis system, not investment advice.
- Promoter holding is currently a proxy (`heldPercentInsiders`) where exact promoter data is unavailable.
- Data providers may throttle or return partial fields; reruns can improve coverage.
- News links are normalized to openable Google News article links.

## Next Improvements

- Replace sentiment model with finance-tuned FinBERT
- Integrate direct promoter holding from exchange filings/data vendor
- Add transaction cost model and walk-forward portfolio backtest
- Add sector-neutral and risk-budget constraints

