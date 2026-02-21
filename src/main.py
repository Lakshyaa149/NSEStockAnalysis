import argparse
import yaml

from src.data.loader import load_prices
from src.features.technical import add_technical_features
from src.features.sentiment import SentimentProvider
from src.models.train import prepare_training_frame, fit_model, FEATURES
from src.risk.overlay import apply_risk_overlay
from src.backtest.engine import run_backtest
from src.news.report import generate_30d_news_and_ceo_reports
from src.fundamentals.report import generate_fundamentals_report


def build_features(cfg):
    df = load_prices(cfg["data"]["raw_prices_path"])
    df = add_technical_features(df, **cfg["features"])
    df["sentiment"] = SentimentProvider().score(df)
    df.to_csv(cfg["data"]["processed_path"], index=False)
    return df


def train(cfg):
    df = build_features(cfg)
    tf = prepare_training_frame(df, horizon=cfg["model"]["target_horizon"])
    split = int(len(tf) * (1 - cfg["model"]["test_size"]))
    tr, te = tf.iloc[:split], tf.iloc[split:]
    model = fit_model(tr)
    acc = (model.predict(te[FEATURES]) == te["target"]).mean() if len(te) else 0.0
    print(f"Train={len(tr)} Test={len(te)} Accuracy={acc:.4f}")


def backtest(cfg):
    df = build_features(cfg)
    tf = prepare_training_frame(df, horizon=cfg["model"]["target_horizon"])
    model = fit_model(tf.iloc[: max(1, int(len(tf) * 0.7))])
    tf["proba"] = model.predict_proba(tf[FEATURES])[:, 1]
    tf["weight"] = apply_risk_overlay(tf["proba"], cfg["risk"]["max_weight_per_stock"])
    res = run_backtest(tf, "weight")
    print(f"Sharpe={res['sharpe']:.4f}")


def news_report(cfg):
    company_summary, ceo_summary = generate_30d_news_and_ceo_reports(cfg["news"])

    print("\n30-Day Company News Sentiment")
    if company_summary.empty:
        print("No company news found in lookback window.")
    else:
        print(company_summary.head(20).to_string(index=False))

    print("\n30-Day CEO Commentary Sentiment")
    if ceo_summary.empty:
        print("No CEO commentary news found in lookback window.")
    else:
        print(ceo_summary.head(20).to_string(index=False))

    outputs = cfg["news"]["outputs"]
    print("\nSaved files:")
    print(f"- {outputs['company_summary_csv']}")
    print(f"- {outputs['company_details_csv']}")
    print(f"- {outputs['ceo_summary_csv']}")
    print(f"- {outputs['ceo_details_csv']}")


def fundamentals_report(cfg):
    rep = generate_fundamentals_report(cfg["fundamentals"])
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
    show_cols = [c for c in cols if c in rep.columns]
    print("\nFundamentals + Sentiment Ranked Report (Top 30)")
    print(rep[show_cols].head(30).to_string(index=False))
    out = cfg["fundamentals"]["outputs"]
    print("\nSaved files:")
    print(f"- {out['raw_fundamentals_csv']}")
    print(f"- {out['ranked_report_csv']}")
    print(f"- {out['top_picks_csv']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["train", "backtest", "news_report", "fundamentals_report"])
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = yaml.safe_load(open(args.config))

    if args.command == "train":
        train(cfg)
    elif args.command == "backtest":
        backtest(cfg)
    elif args.command == "news_report":
        news_report(cfg)
    else:
        fundamentals_report(cfg)


if __name__ == "__main__":
    main()
