from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.news.analyzer import NewsSentimentAnalyzer
from src.news.fetcher import fetch_news_for_queries


def _safe_links(series: pd.Series, top_n: int = 3) -> str:
    links = [x for x in series.tolist() if isinstance(x, str) and x.strip()]
    return " | ".join(list(dict.fromkeys(links))[:top_n])


def _vibe(score: float) -> str:
    if score >= 0.05:
        return "positive vibe"
    if score <= -0.05:
        return "negative vibe"
    return "neutral vibe"


def _summarize(details_df: pd.DataFrame, symbol_col: str = "symbol") -> pd.DataFrame:
    summary = (
        details_df.groupby(symbol_col)
        .agg(
            news_count=("title", "count"),
            positive_count=("sentiment_label", lambda s: int((s == "positive").sum())),
            neutral_count=("sentiment_label", lambda s: int((s == "neutral").sum())),
            negative_count=("sentiment_label", lambda s: int((s == "negative").sum())),
            avg_sentiment=("sentiment_score", "mean"),
            top_links=("link", _safe_links),
        )
        .reset_index()
    )
    summary["vibe"] = summary["avg_sentiment"].apply(_vibe)
    return summary.sort_values(["avg_sentiment", "news_count"], ascending=[False, False]).reset_index(drop=True)


def _existing_non_empty(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(p)
        return df if not df.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _write_output(new_df: pd.DataFrame, csv_path: str, preserve_non_empty: bool) -> pd.DataFrame:
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Fresh mode: always overwrite, even with empty result.
    if not preserve_non_empty:
        new_df.to_csv(csv_path, index=False)
        return new_df

    # Preserve mode: keep old non-empty files if new result is empty.
    if not new_df.empty:
        new_df.to_csv(csv_path, index=False)
        return new_df

    existing = _existing_non_empty(csv_path)
    if not existing.empty:
        return existing

    new_df.to_csv(csv_path, index=False)
    return new_df


def _analyze_items(items: list[dict], output_details_csv: str, preserve_non_empty: bool) -> pd.DataFrame:
    if not items:
        empty = pd.DataFrame(
            columns=[
                "symbol",
                "title",
                "link",
                "published",
                "source",
                "sentiment_label",
                "sentiment_score",
            ]
        )
        return _write_output(empty, output_details_csv, preserve_non_empty)

    analyzer = NewsSentimentAnalyzer()
    rows = []
    for item in items:
        result = analyzer.analyze(item["title"])
        rows.append(
            {
                "symbol": item["symbol"],
                "title": item["title"],
                "link": item["link"],
                "published": item["published"],
                "source": item["source"],
                "sentiment_label": result.label,
                "sentiment_score": result.score,
            }
        )

    details_df = pd.DataFrame(rows)
    return _write_output(details_df, output_details_csv, preserve_non_empty)


def _load_companies(news_cfg: dict) -> list[dict]:
    if news_cfg.get("companies"):
        return news_cfg["companies"]

    universe_csv = news_cfg.get("universe_csv")
    if not universe_csv:
        raise ValueError("Provide either news.companies or news.universe_csv")

    universe = pd.read_csv(universe_csv)
    required = {"symbol", "company_name"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Universe CSV missing required columns: {sorted(missing)}")

    if "ceo_name" not in universe.columns:
        universe["ceo_name"] = ""

    universe = universe.dropna(subset=["symbol", "company_name"])
    universe["symbol"] = universe["symbol"].astype(str).str.strip().str.upper()
    universe["company_name"] = universe["company_name"].astype(str).str.strip()
    universe["ceo_name"] = universe["ceo_name"].fillna("").astype(str).str.strip()
    universe = universe[(universe["symbol"] != "") & (universe["company_name"] != "")]

    max_companies = news_cfg.get("max_companies")
    if max_companies is not None:
        universe = universe.head(int(max_companies))

    return universe[["symbol", "company_name", "ceo_name"]].to_dict(orient="records")


def generate_30d_news_and_ceo_reports(news_cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    lookback_days = int(news_cfg.get("lookback_days", 30))
    limit_per_company = int(news_cfg.get("limit_per_company", 10))
    max_workers = int(news_cfg.get("max_workers", 12))

    # force_refresh=True means always compute fresh and overwrite files.
    # force_refresh=False preserves old non-empty files if new fetch is empty.
    force_refresh = bool(news_cfg.get("force_refresh", True))
    preserve_non_empty = not force_refresh

    companies = _load_companies(news_cfg)

    company_queries: dict[str, str] = {}
    ceo_queries: dict[str, str] = {}

    for c in companies:
        symbol = str(c["symbol"]).strip().upper()
        company_name = str(c["company_name"]).strip()
        ceo_name = str(c.get("ceo_name") or "").strip()

        company_queries[symbol] = f'"{company_name}" NSE stock'
        ceo_topic = f'"{ceo_name}" "{company_name}"' if ceo_name else f'"{company_name}" CEO'
        ceo_queries[symbol] = f"{ceo_topic} commentary OR interview OR says OR guidance"

    company_items_raw = fetch_news_for_queries(
        symbol_to_query=company_queries,
        lookback_days=lookback_days,
        limit_per_symbol=limit_per_company,
        max_workers=max_workers,
    )
    ceo_items_raw = fetch_news_for_queries(
        symbol_to_query=ceo_queries,
        lookback_days=lookback_days,
        limit_per_symbol=limit_per_company,
        max_workers=max_workers,
    )

    company_items = [item.__dict__ for item in company_items_raw]
    ceo_items = [item.__dict__ for item in ceo_items_raw]

    outputs = news_cfg["outputs"]

    company_details = _analyze_items(company_items, outputs["company_details_csv"], preserve_non_empty)
    ceo_details = _analyze_items(ceo_items, outputs["ceo_details_csv"], preserve_non_empty)

    company_summary = _summarize(company_details) if not company_details.empty else pd.DataFrame(
        columns=["symbol", "news_count", "positive_count", "neutral_count", "negative_count", "avg_sentiment", "top_links", "vibe"]
    )
    ceo_summary = _summarize(ceo_details) if not ceo_details.empty else pd.DataFrame(
        columns=["symbol", "news_count", "positive_count", "neutral_count", "negative_count", "avg_sentiment", "top_links", "vibe"]
    )

    company_summary = _write_output(company_summary, outputs["company_summary_csv"], preserve_non_empty)
    ceo_summary = _write_output(ceo_summary, outputs["ceo_summary_csv"], preserve_non_empty)

    return company_summary, ceo_summary
