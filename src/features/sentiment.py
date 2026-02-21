import pandas as pd

class SentimentProvider:
    # Placeholder sentiment. Replace with FinBERT later.
    def score(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0.0, index=df.index)
