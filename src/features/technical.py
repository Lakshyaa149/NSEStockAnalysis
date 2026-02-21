import numpy as np
import pandas as pd

def add_technical_features(df: pd.DataFrame, rsi_window=14, ma_fast=10, ma_slow=30) -> pd.DataFrame:
    out = df.copy()
    out["ret_1d"] = out.groupby("symbol")["close"].pct_change()
    out["ma_fast"] = out.groupby("symbol")["close"].transform(lambda s: s.rolling(ma_fast).mean())
    out["ma_slow"] = out.groupby("symbol")["close"].transform(lambda s: s.rolling(ma_slow).mean())
    out["mom_ma"] = out["ma_fast"] / out["ma_slow"] - 1.0

    delta = out.groupby("symbol")["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.groupby(out["symbol"]).transform(lambda s: s.rolling(rsi_window).mean())
    avg_loss = loss.groupby(out["symbol"]).transform(lambda s: s.rolling(rsi_window).mean())

    # Keep RSI finite for monotonic segments by adding epsilon to denominator.
    rs = avg_gain / (avg_loss + 1e-9)
    out["rsi"] = 100 - (100 / (1 + rs))

    return out
