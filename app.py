from __future__ import annotations

import io
import logging
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

from src.main import backtest as run_backtest
from src.main import fundamentals_report as run_fundamentals_report
from src.main import news_report as run_news_report
from src.main import policy_report as run_policy_report
from src.main import train as run_train

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger("nse_ui")

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "default.yaml"
PROCESSED_DIR = ROOT / "data" / "processed"

COMPANY_SUMMARY = PROCESSED_DIR / "company_news_sentiment_summary_30d.csv"
COMPANY_DETAILS = PROCESSED_DIR / "company_news_sentiment_details_30d.csv"
CEO_SUMMARY = PROCESSED_DIR / "ceo_commentary_sentiment_summary_30d.csv"
CEO_DETAILS = PROCESSED_DIR / "ceo_commentary_sentiment_details_30d.csv"
POLICY_SUMMARY = PROCESSED_DIR / "policy_beneficiary_stocks.csv"
POLICY_EVIDENCE = PROCESSED_DIR / "policy_beneficiary_evidence.csv"
FUND_RANKED = PROCESSED_DIR / "fundamentals_ranked_report.csv"
FUND_TOP = PROCESSED_DIR / "fundamentals_top_picks.csv"


def _log(message: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} | {message}"
    LOGGER.info(message)
    logs = st.session_state.get("app_logs", [])
    logs.append(line)
    st.session_state["app_logs"] = logs[-500:]


def _load_config() -> dict:
    _log(f"Loading config from {CONFIG_PATH}")
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_main_function(fn, cfg: dict) -> tuple[bool, str]:
    name = getattr(fn, "__name__", "unknown_fn")
    _log(f"Running function: {name}")
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer), redirect_stderr(buffer):
            fn(cfg)
        _log(f"Function succeeded: {name}")
        return True, buffer.getvalue().strip()
    except Exception as e:
        _log(f"Function failed: {name} ({e})")
        logs = buffer.getvalue().strip()
        if logs:
            logs += "\n"
        logs += f"ERROR: {e}"
        return False, logs


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        _log(f"CSV not found: {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    _log(f"Loaded CSV: {path.name} rows={len(df)}")
    return df


def _refresh_news_reports() -> None:
    _log("Trigger: refresh news reports")
    ok, out = _run_main_function(run_news_report, _load_config())
    st.session_state["last_news_output"] = out[-8000:]
    if ok:
        st.success("News reports refreshed.")
    else:
        st.error("News report refresh failed.")


def _refresh_policy_report() -> None:
    _log("Trigger: refresh policy beneficiary report")
    ok, out = _run_main_function(run_policy_report, _load_config())
    st.session_state["last_policy_output"] = out[-8000:]
    if ok:
        st.success("Policy beneficiary report refreshed.")
    else:
        st.error("Policy beneficiary report refresh failed.")


def _refresh_fundamentals_report() -> None:
    _log("Trigger: refresh fundamentals report")
    ok, out = _run_main_function(run_fundamentals_report, _load_config())
    st.session_state["last_fund_output"] = out[-8000:]
    if ok:
        st.success("Fundamentals report refreshed.")
    else:
        st.error("Fundamentals report refresh failed.")


def _run_training() -> None:
    _log("Trigger: run train")
    ok, out = _run_main_function(run_train, _load_config())
    st.session_state["last_train_output"] = out[-8000:]
    if ok:
        st.success("Training completed.")
    else:
        st.error("Training failed.")


def _run_backtest_job() -> None:
    _log("Trigger: run backtest")
    ok, out = _run_main_function(run_backtest, _load_config())
    st.session_state["last_backtest_output"] = out[-8000:]
    if ok:
        st.success("Backtest completed.")
    else:
        st.error("Backtest failed.")


def _kpi(label: str, value: str) -> None:
    st.metric(label=label, value=value)


def _screen_overview() -> None:
    _log("Screen: Overview")
    st.subheader("Overview")
    c = _load_csv(COMPANY_SUMMARY)
    ceo = _load_csv(CEO_SUMMARY)
    p = _load_csv(POLICY_SUMMARY)
    f = _load_csv(FUND_RANKED)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _kpi("Company Sentiment Rows", f"{len(c):,}")
    with col2:
        _kpi("CEO Sentiment Rows", f"{len(ceo):,}")
    with col3:
        _kpi("Policy Beneficiary Rows", f"{len(p):,}")
    with col4:
        _kpi("Fundamentals Rows", f"{len(f):,}")

    st.caption("Use the left navigation to open each screen.")


def _screen_modeling() -> None:
    _log("Screen: Modeling")
    st.subheader("Modeling (Train + Backtest)")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Run Train", use_container_width=True):
            with st.spinner("Running train..."):
                _run_training()

    with c2:
        if st.button("Run Backtest", use_container_width=True):
            with st.spinner("Running backtest..."):
                _run_backtest_job()


def _screen_company_news() -> None:
    _log("Screen: Company News")
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
    _log("Screen: CEO Commentary")
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


def _screen_policy_beneficiaries() -> None:
    _log("Screen: Policy Beneficiaries")
    st.subheader("Policy / Government Scheme Beneficiary Stocks")
    st.caption("Built from company Google-news headlines using policy/scheme keyword + sentiment scoring.")

    if st.button("Create/Refresh Policy Beneficiary Report", use_container_width=True):
        with st.spinner("Running policy_report..."):
            _refresh_policy_report()

    p = _load_csv(POLICY_SUMMARY)
    e = _load_csv(POLICY_EVIDENCE)

    if p.empty:
        st.warning("Policy beneficiary report not found. Generate company news first, then refresh this page.")
        return

    min_mentions = st.slider("Min Scheme Mentions", 1, int(max(1, p["scheme_mentions"].max())), 2)
    buckets = sorted(p["benefit_bucket"].dropna().unique().tolist())
    selected_buckets = st.multiselect("Benefit Category", buckets, default=buckets)

    view = p[(p["scheme_mentions"] >= min_mentions) & (p["benefit_bucket"].isin(selected_buckets))].copy()
    st.dataframe(view, use_container_width=True)

    st.markdown("Top 25 by Policy Benefit Score")
    st.dataframe(view.sort_values("policy_benefit_score", ascending=False).head(25), use_container_width=True)

    if not e.empty:
        symbols = sorted(e["symbol"].dropna().astype(str).unique().tolist())
        selected = st.selectbox("Symbol (Policy Evidence)", symbols, index=0)
        ev = e[e["symbol"].astype(str) == selected].head(30).copy()
        st.dataframe(
            ev[["symbol", "published", "title", "link", "source", "sentiment_score", "matched_categories", "policy_row_score"]],
            use_container_width=True,
            column_config={"link": st.column_config.LinkColumn("News Link")},
        )


def _screen_fundamentals() -> None:
    _log("Screen: Fundamentals")
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
        "Modeling": _screen_modeling,
        "Company News": _screen_company_news,
        "CEO Commentary": _screen_ceo_commentary,
        "Policy Beneficiaries": _screen_policy_beneficiaries,
        "Fundamentals": _screen_fundamentals,
    }

    choice = st.sidebar.radio("Screen", list(screens.keys()))
    _log(f"Navigation selected: {choice}")
    screens[choice]()

    with st.expander("Command Logs"):
        st.text_area("train output", st.session_state.get("last_train_output", ""), height=140)
        st.text_area("backtest output", st.session_state.get("last_backtest_output", ""), height=140)
        st.text_area("news_report output", st.session_state.get("last_news_output", ""), height=140)
        st.text_area("policy_report output", st.session_state.get("last_policy_output", ""), height=140)
        st.text_area("fundamentals_report output", st.session_state.get("last_fund_output", ""), height=140)

    with st.expander("App Logs"):
        st.text_area("app logs", "\n".join(st.session_state.get("app_logs", [])), height=260)


if __name__ == "__main__":
    main()
