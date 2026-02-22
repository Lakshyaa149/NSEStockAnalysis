from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_POLICY_KEYWORDS: dict[str, list[str]] = {
    "capex_infra": [
        "infrastructure",
        "infra",
        "capex",
        "highway",
        "railway",
        "metro",
        "airport",
        "port",
        "construction",
    ],
    "manufacturing_pli": [
        "pli",
        "production linked incentive",
        "manufacturing",
        "electronics manufacturing",
        "domestic manufacturing",
    ],
    "defence": [
        "defence",
        "defense",
        "defence ministry",
        "military",
        "order win",
    ],
    "energy_transition": [
        "renewable",
        "solar",
        "wind",
        "green hydrogen",
        "battery",
        "ev",
        "energy transition",
    ],
    "banking_credit": [
        "credit growth",
        "msme",
        "fiscal",
        "budget",
        "policy support",
        "rate cut",
    ],
    "agri_rural": [
        "agri",
        "agriculture",
        "rural",
        "fertilizer",
        "irrigation",
        "crop",
    ],
    "healthcare_pharma": [
        "healthcare",
        "pharma",
        "drug policy",
        "medical devices",
    ],
}


def _normalize_keywords(raw: Any) -> dict[str, list[str]]:
    if not isinstance(raw, dict) or not raw:
        return DEFAULT_POLICY_KEYWORDS
    out: dict[str, list[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            out[str(k)] = [str(x).lower().strip() for x in v if str(x).strip()]
    return out or DEFAULT_POLICY_KEYWORDS


def _matches(text: str, keywords: dict[str, list[str]]) -> list[str]:
    t = (text or "").lower()
    hits: list[str] = []
    for cat, words in keywords.items():
        if any(w in t for w in words):
            hits.append(cat)
    return hits


def _bucket(score: float, mentions: int) -> str:
    if mentions >= 5 and score >= 8:
        return "High Policy Benefit"
    if mentions >= 2 and score >= 4:
        return "Potential Beneficiary"
    return "Watch"


def generate_policy_benefit_report(cfg: dict) -> pd.DataFrame:
    source_details_csv = cfg["source_details_csv"]
    outputs = cfg["outputs"]
    keywords = _normalize_keywords(cfg.get("keywords"))

    details_path = Path(source_details_csv)
    if not details_path.exists():
        raise FileNotFoundError(f"Missing source details CSV: {source_details_csv}")

    details = pd.read_csv(details_path)
    required = {"symbol", "title", "link", "published", "source", "sentiment_score"}
    missing = required - set(details.columns)
    if missing:
        raise ValueError(f"Source details CSV missing required columns: {sorted(missing)}")

    if details.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "symbol",
                "scheme_mentions",
                "avg_scheme_sentiment",
                "policy_benefit_score",
                "matched_categories",
                "top_links",
                "benefit_bucket",
            ]
        )
        empty_evidence = pd.DataFrame(
            columns=[
                "symbol",
                "published",
                "title",
                "link",
                "source",
                "sentiment_score",
                "matched_categories",
                "policy_row_score",
            ]
        )
        Path(outputs["summary_csv"]).parent.mkdir(parents=True, exist_ok=True)
        empty_summary.to_csv(outputs["summary_csv"], index=False)
        empty_evidence.to_csv(outputs["evidence_csv"], index=False)
        return empty_summary

    work = details.copy()
    work["sentiment_score"] = pd.to_numeric(work["sentiment_score"], errors="coerce").fillna(0.0)
    work["matched_categories_list"] = work["title"].astype(str).apply(lambda t: _matches(t, keywords))
    work = work[work["matched_categories_list"].map(len) > 0].copy()

    if work.empty:
        empty_summary = pd.DataFrame(
            columns=[
                "symbol",
                "scheme_mentions",
                "avg_scheme_sentiment",
                "policy_benefit_score",
                "matched_categories",
                "top_links",
                "benefit_bucket",
            ]
        )
        Path(outputs["summary_csv"]).parent.mkdir(parents=True, exist_ok=True)
        empty_summary.to_csv(outputs["summary_csv"], index=False)
        work.to_csv(outputs["evidence_csv"], index=False)
        return empty_summary

    work["matched_categories"] = work["matched_categories_list"].apply(lambda x: ", ".join(sorted(set(x))))
    work["keyword_hit_count"] = work["matched_categories_list"].apply(lambda x: len(set(x)))
    # Policy row score: keyword coverage + positive sentiment boost.
    work["policy_row_score"] = work["keyword_hit_count"] * 1.5 + work["sentiment_score"].clip(lower=0.0) * 2.0

    evidence_cols = [
        "symbol",
        "published",
        "title",
        "link",
        "source",
        "sentiment_score",
        "matched_categories",
        "policy_row_score",
    ]
    evidence = work[evidence_cols].sort_values(["policy_row_score", "sentiment_score"], ascending=[False, False])

    grouped = work.groupby("symbol", as_index=False).agg(
        scheme_mentions=("title", "count"),
        avg_scheme_sentiment=("sentiment_score", "mean"),
        policy_benefit_score=("policy_row_score", "sum"),
    )

    cat_map = (
        work.groupby("symbol")["matched_categories_list"]
        .apply(lambda s: ", ".join(sorted(set([c for row in s for c in row]))))
        .reset_index(name="matched_categories")
    )

    link_map = (
        work.groupby("symbol")["link"]
        .apply(lambda s: " | ".join(list(dict.fromkeys([x for x in s if isinstance(x, str) and x]))[:3]))
        .reset_index(name="top_links")
    )

    summary = grouped.merge(cat_map, on="symbol", how="left").merge(link_map, on="symbol", how="left")
    summary["benefit_bucket"] = summary.apply(
        lambda r: _bucket(float(r["policy_benefit_score"]), int(r["scheme_mentions"])), axis=1
    )

    summary = summary.sort_values(["policy_benefit_score", "scheme_mentions", "avg_scheme_sentiment"], ascending=[False, False, False])

    Path(outputs["summary_csv"]).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(outputs["summary_csv"], index=False)
    evidence.to_csv(outputs["evidence_csv"], index=False)

    return summary
