from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import sleep
from typing import Any

import pandas as pd
import yfinance as yf


@dataclass
class FundamentalRow:
    symbol: str
    yahoo_ticker: str
    company_name: str
    industry: str | None
    sector: str | None
    trailing_pe: float | None
    forward_pe: float | None
    peg_ratio: float | None
    price_to_book: float | None
    debt_to_equity: float | None
    return_on_equity: float | None
    return_on_assets: float | None
    profit_margins: float | None
    market_cap: float | None
    promoter_holding_proxy_pct: float | None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
        if pd.isna(x):
            return None
        return x
    except Exception:
        return None


def _yahoo_ticker(symbol: str) -> str:
    return f"{symbol}.NS"


def fetch_single_fundamental(symbol: str, company_name: str, retries: int = 2) -> FundamentalRow:
    ticker = _yahoo_ticker(symbol)

    info = {}
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            if info:
                break
        except Exception as e:
            last_err = e
        sleep(0.15 * (attempt + 1))

    if not info and last_err is not None:
        raise last_err

    trailing_pe = _to_float(info.get("trailingPE"))
    trailing_eps = _to_float(info.get("trailingEps"))
    current_price = _to_float(info.get("currentPrice") or info.get("regularMarketPrice"))
    if trailing_pe is None and trailing_eps and trailing_eps > 0 and current_price:
        trailing_pe = current_price / trailing_eps

    insider = _to_float(info.get("heldPercentInsiders"))
    promoter_proxy = insider * 100.0 if insider is not None else None

    return FundamentalRow(
        symbol=symbol,
        yahoo_ticker=ticker,
        company_name=company_name,
        industry=info.get("industry"),
        sector=info.get("sector"),
        trailing_pe=trailing_pe,
        forward_pe=_to_float(info.get("forwardPE")),
        peg_ratio=_to_float(info.get("pegRatio")),
        price_to_book=_to_float(info.get("priceToBook")),
        debt_to_equity=_to_float(info.get("debtToEquity")),
        return_on_equity=_to_float(info.get("returnOnEquity")),
        return_on_assets=_to_float(info.get("returnOnAssets")),
        profit_margins=_to_float(info.get("profitMargins")),
        market_cap=_to_float(info.get("marketCap")),
        promoter_holding_proxy_pct=promoter_proxy,
    )


def fetch_fundamentals_for_universe(
    universe_df: pd.DataFrame,
    max_workers: int = 16,
) -> pd.DataFrame:
    rows: list[dict] = []
    workers = max(1, min(int(max_workers), 32))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fetch_single_fundamental, row["symbol"], row["company_name"]): row["symbol"]
            for _, row in universe_df.iterrows()
        }
        for fut in as_completed(futures):
            try:
                rows.append(fut.result().__dict__)
            except Exception:
                continue

    return pd.DataFrame(rows)
