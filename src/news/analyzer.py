from __future__ import annotations

from dataclasses import dataclass

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


@dataclass
class SentimentResult:
    label: str
    score: float


class NewsSentimentAnalyzer:
    def __init__(self) -> None:
        self._analyzer = SentimentIntensityAnalyzer()

    def analyze(self, text: str) -> SentimentResult:
        compound = self._analyzer.polarity_scores(text).get("compound", 0.0)
        if compound >= 0.05:
            label = "positive"
        elif compound <= -0.05:
            label = "negative"
        else:
            label = "neutral"
        return SentimentResult(label=label, score=compound)
