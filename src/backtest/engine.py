import numpy as np

def sharpe_ratio(daily_returns, annualization=252):
    vol = daily_returns.std()
    if vol == 0 or np.isnan(vol):
        return 0.0
    return (daily_returns.mean() / vol) * np.sqrt(annualization)

def run_backtest(df, weights_col="weight"):
    pnl = (df["fwd_ret"].fillna(0) * df[weights_col].fillna(0)).groupby(df["date"]).sum()
    return {"sharpe": sharpe_ratio(pnl), "daily_returns": pnl}
