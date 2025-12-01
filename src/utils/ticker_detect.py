"""
Ticker detection utilities: extract tickers from raw Reddit text.
Rule ensemble:
  1) cashtags like $TSLA (strong)
  2) exact uppercase tickers with finance-context words (medium)
  3) alias/company-name match (medium/weak)
"""
from __future__ import annotations
import re
from typing import Dict, Iterable, List, Set

FIN_CTX = re.compile(r"(stock|stocks|share|shares|call|calls|put|puts|option|earnings|eps|guidance|pt|price target|upgrade|downgrade|buyback|dividend|split|quarter|q[1-4]|revenue|profit|loss)", re.I)
CASHTAG_RE = re.compile(r"\$([A-Z]{1,5})\b")
TICKER_WORD_RE = re.compile(r"\b([A-Z]{1,5})\b")

def extract_tickers(text: str, valid_tickers: Set[str], ambiguous: Set[str]) -> Set[str]:
    """Extract likely tickers using cashtags and finance-context uppercase tokens."""
    if not text:
        return set()
    found: Set[str] = set()
    for m in CASHTAG_RE.findall(text):
        if m in valid_tickers:
            found.add(m)
    if FIN_CTX.search(text or ""):
        for m in TICKER_WORD_RE.findall(text):
            if m in valid_tickers and m not in ambiguous:
                found.add(m)
    return found

def extract_by_alias(text: str, alias_map: Dict[str, Iterable[str]]) -> Set[str]:
    """Alias/company-name based detection (case-insensitive)."""
    if not text:
        return set()
    low = text.lower()
    hits: Set[str] = set()
    for ticker, names in alias_map.items():
        for name in names:
            if name.lower() in low:
                hits.add(ticker); break
    return hits

def detect_ensemble(text: str, valid_tickers: Set[str], ambiguous: Set[str], alias_map: Dict[str, Iterable[str]]) -> List[str]:
    """Union of cashtag/uppercase-with-context and alias matches."""
    return sorted(extract_tickers(text, valid_tickers, ambiguous) | extract_by_alias(text, alias_map))
