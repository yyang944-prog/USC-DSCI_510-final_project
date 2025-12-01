"""
Sentiment scoring helpers using VADER.
"""
from __future__ import annotations
from typing import Optional
from nltk.sentiment import SentimentIntensityAnalyzer

_sia: Optional[SentimentIntensityAnalyzer] = None

def _get_sia() -> SentimentIntensityAnalyzer:
    global _sia
    if _sia is None:
        _sia = SentimentIntensityAnalyzer()
    return _sia

def vader_compound(text: str) -> float:
    """
    Return VADER compound score ∈ [-1, 1] for given text (safe for None).
    """
    if not text:
        text = ""
    sia = _get_sia()
    return float(sia.polarity_scores(text)["compound"])
