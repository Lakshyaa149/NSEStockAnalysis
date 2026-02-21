from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"

COMPANY_SUMMARY = PROCESSED_DIR / "company_news_sentiment_summary_30d.csv"
COMPANY_DETAILS = PROCESSED_DIR / "company_news_sentiment_details_30d.csv"
CEO_SUMMARY = PROCESSED_DIR / "ceo_commentary_sentiment_summary_30d.csv"
CEO_DETAILS = PROCESSED_DIR / "ceo_commentary_sentiment_details_30d.csv"
FUND_RANKED = PROCESSED_DIR / "fundamentals_ranked_report.csv"
FUND_TOP = PROCESSED_DIR / "fundamentals_top_picks.csv"


def _run_command(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(
        args,
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, output.strip()


@st.cache_data(ttl=300)
def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _refresh_news_reports() -> None:
    ok, out = _run_command([
        sys.executable,
        "-m",
        "src.main",
        "news_report",
        "--config",
        str(CONFIG_PATH),
    ])
    st.session_state["last_news_output"] = out[-8000:]
    if ok:
        _load_csv.clear()
        st.success("News reports refreshed.")
    else:
        st.error("News report refresh failed.")


def _refresh_fundamentals_report() -> None:
    ok, out = _run_command([
        sys.executable,
        "-m",
        "src.main",
        "fundamentals_report",
        "--config",
        str(CONFIG_PATH),
    ])
    st.session_state["last_fund_output"] = out[-8000:]
    if ok:
        _load_csv.clear()
        st.success("Fundamentals report refreshed.")
    else:
        st.error("Fundamentals report refresh failed.")


def _kpi(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def _screen_overview() -> None:
    st.subheader("Overview")
    c = _load_csv(COMPANY_SUMMARY)
    ceo = _load_csv(CEO_SUMMARY)
    f = _load_csv(FUND_RANKED)

    col1, col2, col3 = st.columns(3)
    with col1:
        _kpi("Company Sentiment Rows", f"{len(c):,}")
    with col2:
        _kpi("CEO Sentiment Rows", f"{len(ceo):,}")
    with col3:
        _kpi("Fundamentals Rows", f"{len(f):,}")

    st.caption("Use the left navigation to open each screen.")


def _screen_company_news() -> None:
    st.subheader("Company News Sentiment (30D)")
    if st.button("Create/Refresh Company + CEO News Reports", use_container_width=True):
        with st.spinner("Running news_report..."):
            _refresh_news_reports()

    c = _load_csv(COMPANY_SUMMARY)
    d = _load_csv(COMPANY_DETAILS)

    if c.empty:
        st.warning("Company summary report not found. Click refresh.")
        return

    min_news = st.slider("Min News Count", 1, int(c["news_count"].max()), 3)
    vibes = st.multiselect("Vibe", sorted(c["vibe"].dropna().unique().tolist()), default=sorted(c["vibe"].dropna().unique().tolist()))

    view = c[(c["news_count"] >= min_news) & (c["vibe"].isin(vibes))].copy()
    st.dataframe(view, use_container_width=True)

    st.markdown("Top 20 Positive by Avg Sentiment")
    st.dataframe(view.sort_values("avg_sentiment", ascending=False).head(20), use_container_width=True)

    if not d.empty:
        st.markdown("Recent Headlines")
        symbols = sorted(d["symbol"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Symbol", symbols, index=0)
        dv = d[d["symbol"].astype(str) == selected].head(30).copy()
        st.dataframe(
            dv[["symbol", "published", "title", "link", "source", "sentiment_label", "sentiment_score"]],
            use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("News Link")},
        )


def _screen_ceo_commentary() -> None:
    st.subheader("CEO Commentary Sentiment (30D)")
    ceo = _load_csv(CEO_SUMMARY)
    d = _load_csv(CEO_DETAILS)

    if ceo.empty:
        st.warning("CEO summary report not found. Click refresh in Company News screen.")
        return

    min_news = st.slider("Min CEO Mentions", 1, int(max(1, ceo["news_count"].max())), 2)
    view = ceo[ceo["news_count"] >= min_news].copy()
    st.dataframe(view, use_container_width=True)

    st.markdown("Top 20 CEO Sentiment")
    st.dataframe(view.sort_values("avg_sentiment", ascending=False).head(20), use_container_width=True)

    if not d.empty:
        symbols = sorted(d["symbol"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Symbol (CEO)", symbols, index=0)
        dv = d[d["symbol"].astype(str) == selected].head(30).copy()
        st.dataframe(
            dv[["symbol", "published", "title", "link", "source", "sentiment_label", "sentiment_score"]],
            use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("News Link")},
        )


def _screen_fundamentals() -> None:
    st.subheader("Fundamentals + Sentiment Ranking")
    if st.button("Create/Refresh Fundamentals Report", use_container_width=True):
        with st.spinner("Running fundamentals_report..."):
            _refresh_fundamentals_report()

    rank = _load_csv(FUND_RANKED)
    top = _load_csv(FUND_TOP)

    if rank.empty:
        st.warning("Fundamentals report not found. Click refresh.")
        return

    c1, c2, c3 = st.columns(3)
    with c1:
        _kpi("High Potential", f"{(rank['return_potential_flag'] == 'High Potential').sum():,}")
    with c2:
        _kpi("Watchlist", f"{(rank['return_potential_flag'] == 'Watchlist').sum():,}")
    with c3:
        _kpi("Avoid/Needs Work", f"{(rank['return_potential_flag'] == 'Avoid/Needs Work').sum():,}")

    flags = sorted(rank["return_potential_flag"].dropna().unique().tolist())
    selected_flags = st.multiselect("Category", flags, default=flags)
    min_score = st.slider("Minimum Score", int(rank["score"].min()), int(rank["score"].max()), 5)

    view = rank[(rank["return_potential_flag"].isin(selected_flags)) & (rank["score"] >= min_score)].copy()
    cols = [
        "symbol",
        "company_name",
        "score",
        "return_potential_flag",
        "trailing_pe",
        "forward_pe",
        "peg_ratio",
        "promoter_holding_proxy_pct",
        "company_news_sentiment",
        "ceo_commentary_sentiment",
        "reason_summary",
    ]
    show_cols = [c for c in cols if c in view.columns]
    st.dataframe(view[show_cols], use_container_width=True)

    if not top.empty:
        st.markdown("Top Picks File (as generated)")
        st.dataframe(top[show_cols], use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="NSE Stock Analysis", layout="wide")
    st.title("NSE Stock Analysis UI")

    screens = {
        "Overview": _screen_overview,
        "Company News": _screen_company_news,
        "CEO Commentary": _screen_ceo_commentary,
        "Fundamentals": _screen_fundamentals,
    }

    choice = st.sidebar.radio("Screen", list(screens.keys()))
    screens[choice]()

    with st.expander("Command Logs"):
        st.text_area("news_report output", st.session_state.get("last_news_output", ""), height=180)
        st.text_area("fundamentals_report output", st.session_state.get("last_fund_output", ""), height=180)


if __name__ == "__main__":
    main()
