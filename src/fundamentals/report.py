from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.fundamentals.fetcher import fetch_fundamentals_for_universe


def _load_universe(universe_csv: str, max_companies: int | None = None) -> pd.DataFrame:
    u = pd.read_csv(universe_csv)
    required = {"symbol", "company_name"}
    missing = required - set(u.columns)
    if missing:
        raise ValueError(f"Universe CSV missing required columns: {sorted(missing)}")

    u = u[["symbol", "company_name"]].dropna()
    u["symbol"] = u["symbol"].astype(str).str.strip().str.upper()
    u["company_name"] = u["company_name"].astype(str).str.strip()
    u = u[(u["symbol"] != "") & (u["company_name"] != "")].drop_duplicates("symbol")

    if max_companies is not None:
        u = u.head(int(max_companies))

    return u.reset_index(drop=True)


def _load_sentiment(path: str, column_name: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(columns=["symbol", column_name])
    df = pd.read_csv(p)
    if "symbol" not in df.columns or "avg_sentiment" not in df.columns:
        return pd.DataFrame(columns=["symbol", column_name])
    out = df[["symbol", "avg_sentiment"]].copy()
    out.columns = ["symbol", column_name]
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    return out


def _assess_pe(row: pd.Series) -> str:
    pe = row.get("trailing_pe")
    ind_med = row.get("industry_median_pe")
    if pd.isna(pe) or pe is None or pe <= 0:
        return "unavailable_or_loss_making"
    if pd.notna(ind_med) and ind_med > 0:
        ratio = pe / ind_med
        if ratio <= 0.9:
            return "undervalued_vs_industry"
        if ratio <= 1.2:
            return "fair_vs_industry"
        return "overvalued_vs_industry"
    if pe <= 15:
        return "low_absolute_pe"
    if pe <= 25:
        return "mid_absolute_pe"
    return "high_absolute_pe"


def _score_row(row: pd.Series) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []

    pe_assessment = row.get("pe_assessment")
    if pe_assessment in {"undervalued_vs_industry", "low_absolute_pe"}:
        score += 2
        reasons.append("PE favorable")
    elif pe_assessment in {"fair_vs_industry", "mid_absolute_pe"}:
        score += 1
        reasons.append("PE acceptable")

    fpe, tpe = row.get("forward_pe"), row.get("trailing_pe")
    if pd.notna(fpe) and pd.notna(tpe) and fpe > 0 and tpe > 0 and fpe < tpe:
        score += 1
        reasons.append("Forward PE improving")

    peg = row.get("peg_ratio")
    if pd.notna(peg) and peg > 0:
        if peg <= 1.2:
            score += 2
            reasons.append("PEG attractive")
        elif peg <= 1.8:
            score += 1
            reasons.append("PEG reasonable")

    roe = row.get("return_on_equity")
    if pd.notna(roe):
        if roe >= 0.15:
            score += 2
            reasons.append("High ROE")
        elif roe >= 0.10:
            score += 1
            reasons.append("Decent ROE")

    margin = row.get("profit_margins")
    if pd.notna(margin) and margin >= 0.10:
        score += 1
        reasons.append("Healthy margins")

    d2e = row.get("debt_to_equity")
    if pd.notna(d2e):
        if d2e <= 80:
            score += 1
            reasons.append("Manageable debt")
        elif d2e >= 250:
            score -= 1
            reasons.append("High leverage")

    promoter = row.get("promoter_holding_proxy_pct")
    if pd.notna(promoter):
        if promoter >= 45:
            score += 2
            reasons.append("High insider/promoter proxy")
        elif promoter >= 25:
            score += 1
            reasons.append("Moderate insider/promoter proxy")

    c_sent = row.get("company_news_sentiment")
    if pd.notna(c_sent):
        if c_sent >= 0.20:
            score += 2
            reasons.append("Strong company news sentiment")
        elif c_sent >= 0.08:
            score += 1
            reasons.append("Positive company news sentiment")
        elif c_sent <= -0.08:
            score -= 1
            reasons.append("Negative company news sentiment")

    ceo_sent = row.get("ceo_commentary_sentiment")
    if pd.notna(ceo_sent):
        if ceo_sent >= 0.15:
            score += 1
            reasons.append("Positive CEO sentiment")
        elif ceo_sent <= -0.08:
            score -= 1
            reasons.append("Negative CEO sentiment")

    if score >= 8:
        verdict = "High Potential"
    elif score >= 5:
        verdict = "Watchlist"
    else:
        verdict = "Avoid/Needs Work"

    return score, "; ".join(reasons[:6])


def _merge_with_cache(universe: pd.DataFrame, fresh: pd.DataFrame, raw_csv: str) -> pd.DataFrame:
    p = Path(raw_csv)
    if p.exists():
        cached = pd.read_csv(p)
        if "symbol" in cached.columns:
            cached["symbol"] = cached["symbol"].astype(str).str.upper().str.strip()
            fresh["symbol"] = fresh["symbol"].astype(str).str.upper().str.strip()
            merged = pd.concat([cached, fresh], ignore_index=True)
            merged = merged.drop_duplicates(subset=["symbol"], keep="last")
        else:
            merged = fresh
    else:
        merged = fresh

    return universe[["symbol"]].merge(merged, on="symbol", how="left")


def generate_fundamentals_report(cfg: dict) -> pd.DataFrame:
    universe = _load_universe(cfg["universe_csv"], cfg.get("max_companies"))

    fresh = fetch_fundamentals_for_universe(
        universe_df=universe,
        max_workers=cfg.get("max_workers", 16),
    )

    out_cfg = cfg["outputs"]
    Path(out_cfg["raw_fundamentals_csv"]).parent.mkdir(parents=True, exist_ok=True)

    use_cache = bool(cfg.get("use_cache", False))
    if use_cache:
        merged_fund = _merge_with_cache(universe, fresh, out_cfg["raw_fundamentals_csv"])
    else:
        merged_fund = universe[["symbol"]].merge(fresh, on="symbol", how="left")
    merged_fund.to_csv(out_cfg["raw_fundamentals_csv"], index=False)

    df = universe.merge(merged_fund, on="symbol", how="left", suffixes=("", "_dup"))
    if "company_name_dup" in df.columns:
        df = df.drop(columns=["company_name_dup"])

    valid_pe = df[(df["trailing_pe"].notna()) & (df["trailing_pe"] > 0)]
    industry_med = valid_pe.groupby("industry")["trailing_pe"].median().rename("industry_median_pe")
    df = df.merge(industry_med, on="industry", how="left")

    company_sent = _load_sentiment(cfg["company_sentiment_csv"], "company_news_sentiment")
    ceo_sent = _load_sentiment(cfg["ceo_sentiment_csv"], "ceo_commentary_sentiment")
    df = df.merge(company_sent, on="symbol", how="left")
    df = df.merge(ceo_sent, on="symbol", how="left")

    df["pe_assessment"] = df.apply(_assess_pe, axis=1)

    scored = df.apply(_score_row, axis=1)
    df["score"] = [x[0] for x in scored]
    df["reason_summary"] = [x[1] for x in scored]
    df["return_potential_flag"] = np.where(
        df["score"] >= 8,
        "High Potential",
        np.where(df["score"] >= 5, "Watchlist", "Avoid/Needs Work"),
    )

    df = df.sort_values(
        ["score", "company_news_sentiment", "ceo_commentary_sentiment", "market_cap"],
        ascending=[False, False, False, False],
    )

    df.to_csv(out_cfg["ranked_report_csv"], index=False)
    top_n = int(cfg.get("top_n", 50))
    df.head(top_n).to_csv(out_cfg["top_picks_csv"], index=False)

    return df
