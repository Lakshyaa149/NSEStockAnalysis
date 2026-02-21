from sklearn.ensemble import RandomForestClassifier

FEATURES = ["ret_1d", "mom_ma", "rsi", "sentiment"]

def prepare_training_frame(df, horizon=1):
    out = df.copy()
    out["fwd_ret"] = out.groupby("symbol")["close"].pct_change(horizon).shift(-horizon)
    out["target"] = (out["fwd_ret"] > 0).astype(int)
    return out.dropna(subset=FEATURES + ["target"])

def fit_model(train_df):
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(train_df[FEATURES], train_df["target"])
    return model
