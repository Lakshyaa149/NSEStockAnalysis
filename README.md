# NSEStockAnalysis

NSE stock analysis platform with data pipelines + Streamlit UI.

## What It Does

- 30-day company news sentiment (Google News based)
- 30-day CEO commentary sentiment
- Policy-beneficiary stock detection (government scheme/policy signal from news)
- Fundamentals screening (PE, forward PE, PEG, ROE, debt/equity, margins)
- Ranked stock output with return-potential categories
- UI controls to run train/backtest and refresh reports directly

## Key Features

1. `News Sentiment`
- Fetches company news headlines
- Scores sentiment per headline
- Produces company and CEO sentiment summaries + details

2. `Policy Beneficiary Report`
- Uses policy/scheme keyword categories (infra/capex, PLI, defence, energy transition, etc.)
- Combines keyword coverage + sentiment into policy-benefit score
- Returns likely beneficiary stocks with evidence links

3. `Fundamentals + Sentiment Ranking`
- Pulls NSE fundamentals (`.NS`) using market-data source
- Combines valuation + quality + sentiment factors
- Labels stocks as:
  - `High Potential`
  - `Watchlist`
  - `Avoid/Needs Work`

4. `Modeling`
- `Train`: trains model on technical + sentiment placeholder features
- `Backtest`: computes strategy-level performance (Sharpe)

5. `Integrated UI`
- Screens:
  - `Overview`
  - `Modeling`
  - `Company News`
  - `CEO Commentary`
  - `Policy Beneficiaries`
  - `Fundamentals`
- In-app `Command Logs` and `App Logs`

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
      ...generated report csv files...
  src/
    backtest/
    data/
    features/
    fundamentals/
    models/
    news/
      policy_report.py
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

## Run (CLI)

### 1) News reports

```bash
python -m src.main news_report --config configs/default.yaml
```

Outputs:
- `data/processed/company_news_sentiment_summary_30d.csv`
- `data/processed/company_news_sentiment_details_30d.csv`
- `data/processed/ceo_commentary_sentiment_summary_30d.csv`
- `data/processed/ceo_commentary_sentiment_details_30d.csv`

### 2) Policy beneficiary report

```bash
python -m src.main policy_report --config configs/default.yaml
```

Outputs:
- `data/processed/policy_beneficiary_stocks.csv`
- `data/processed/policy_beneficiary_evidence.csv`

### 3) Fundamentals report

```bash
python -m src.main fundamentals_report --config configs/default.yaml
```

Outputs:
- `data/processed/fundamentals_raw.csv`
- `data/processed/fundamentals_ranked_report.csv`
- `data/processed/fundamentals_top_picks.csv`

### 4) Modeling

```bash
python -m src.main train --config configs/default.yaml
python -m src.main backtest --config configs/default.yaml
```

## Run (UI)

```bash
streamlit run app.py
```

Open the local URL shown by Streamlit (e.g. `http://localhost:8502`).

## Configuration

Edit `configs/default.yaml`:

- `news.*`
  - `lookback_days`, `limit_per_company`, `max_workers`, `universe_csv`, `max_companies`
  - `force_refresh: true` to always overwrite outputs with fresh run
- `policy.*`
  - `source_details_csv`
  - output CSV paths
- `fundamentals.*`
  - `max_workers`, `max_companies`, `top_n`
  - `use_cache: false` for fresh fundamentals each run

## Fresh Data Behavior

Current defaults are fresh-first:

- `news.force_refresh = true`
- `fundamentals.use_cache = false`
- UI reads CSVs directly (no 5-minute cache layer)

## Notes

- This is an analytics/screening tool, not investment advice.
- Promoter holding currently uses insider-holding proxy where direct promoter data is unavailable.
- External providers may throttle or return partial fields.
- Policy-beneficiary output is signal-based and should be validated with fundamentals and risk controls.
