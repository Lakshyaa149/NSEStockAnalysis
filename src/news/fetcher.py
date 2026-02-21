from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import List
from urllib.parse import quote_plus

import feedparser


@dataclass
class NewsItem:
    symbol: str
    title: str
    link: str
    published: str
    source: str
    published_at: datetime | None


GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"


def _normalize_google_news_link(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    out = url.strip()
    # RSS links often look like /rss/articles/... and may fail to open directly.
    if "news.google.com/rss/articles/" in out:
        out = out.replace("news.google.com/rss/articles/", "news.google.com/articles/")
    return out


def _parse_published(value: str) -> datetime | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def fetch_google_news(symbol: str, query: str, limit: int = 30, lookback_days: int = 30) -> List[NewsItem]:
    q = query.strip()
    if "when:" not in q:
        q = f"{q} when:{lookback_days}d"

    url = GOOGLE_NEWS_RSS.format(query=quote_plus(q))
    feed = feedparser.parse(url)

    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    items: List[NewsItem] = []
    for entry in feed.entries:
        published = getattr(entry, "published", "")
        published_at = _parse_published(published)

        if published_at is not None and published_at < cutoff:
            continue

        source = ""
        if hasattr(entry, "source") and isinstance(entry.source, dict):
            source = entry.source.get("title", "")

        items.append(
            NewsItem(
                symbol=symbol,
                title=getattr(entry, "title", ""),
                link=_normalize_google_news_link(getattr(entry, "link", "")),
                published=published,
                source=source,
                published_at=published_at,
            )
        )
        if len(items) >= limit:
            break

    return items


def fetch_news_for_queries(
    symbol_to_query: dict[str, str],
    lookback_days: int = 30,
    limit_per_symbol: int = 30,
    max_workers: int = 12,
) -> List[NewsItem]:
    all_items: List[NewsItem] = []
    if not symbol_to_query:
        return all_items

    workers = max(1, min(max_workers, 32))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_google_news,
                symbol=symbol,
                query=query,
                limit=limit_per_symbol,
                lookback_days=lookback_days,
            ): symbol
            for symbol, query in symbol_to_query.items()
        }

        for fut in as_completed(futures):
            try:
                all_items.extend(fut.result())
            except Exception:
                continue

    return all_items
