import pandas as pd

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)
