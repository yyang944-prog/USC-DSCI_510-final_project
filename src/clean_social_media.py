"""
clean_social_media.py — Reddit social media cleaning + ticker extraction + sentiment + daily aggregation

What this script does
---------------------
1) Read a raw Reddit CSV like the one you posted:
   columns = [kind, subreddit, id, parent_id, created_utc, author, score, num_comments, title, text]

2) Clean text:
   - merge title + text -> content
   - strip URLs/HTML entities/punctuation noise, lowercase
   - keep $cashtags

3) Fuzzy ticker extraction (multi-strategy):
   - Cashtags: $TSLA, $AAPL ...
   - Uppercase tickers in a universe (optional CSV with tickers & company names)
   - Company name aliases (Tesla -> TSLA, Amazon -> AMZN, Disney -> DIS, NVIDIA -> NVDA, etc.)
     If you provide a universe CSV, aliases can be taken from a column as well.

4) Sentiment:
   - VADER compound score ∈ [-1, 1]
   - If NLTK lexicon missing, download automatically (once)
   - If VADER fails, fallback to TextBlob polarity scaled to [-1, 1]

5) Discussion volume & daily aggregation:
   - explode to one row per (post/comment × matched_ticker)
   - compute per (ticker, date_utc):
       * n_posts, n_comments
       * n_total
       * mean_compound
       * score_weighted_compound (optional, weight by Reddit score)
       * median_compound

6) Save outputs (separate from price cleaning):
   - data/processed/socialmediadataclean_rows.csv        (row level, exploded by ticker)
   - data/processed/socialmediadataclean_daily.csv       (daily aggregated by ticker)
   - data/processed/socialmediadataclean_top_tickers.csv (overall top discussed tickers)

Recommended upstreams
---------------------
- Your Reddit collector output (e.g., data/raw/reddit_posts.csv)
- (Optional) Ticker universe CSV:
    Required columns: ticker,company
    Optional column:  aliases  (semicolon-separated aliases, e.g., "Tesla;Tesla Inc;TSLA")
    If omitted, a default alias mapping of top US mega-caps is used.

Usage
-----
python src/clean_social_media.py \
  --in data/raw/reddit_posts.csv \
  --out-rows data/processed/socialmediadataclean_rows.csv \
  --out-daily data/processed/socialmediadataclean_daily.csv \
  --out-top data/processed/socialmediadataclean_top_tickers.csv \
  --ticker-universe data/reference/ticker_universe.csv

Minimal run (no universe provided, uses defaults):
python src/clean_social_media.py \
  --in data/raw/reddit_posts.csv

Notes
-----
- This script is SOCIAL MEDIA ONLY. Keep price cleaning in a separate script.
- Later you can left-join daily sentiment onto prices by (ticker, date_utc).
"""

from __future__ import annotations
import argparse
import os
import re
import html
import json
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
import pandas as pd

# --------- utils ---------
def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def to_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(path); df.to_csv(path, index=False)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="input raw reddit csv")
    ap.add_argument("--out-rows", default="data/processed/socialmediadataclean_rows.csv")
    ap.add_argument("--out-daily", default="data/processed/socialmediadataclean_daily.csv")
    ap.add_argument("--out-top", default="data/processed/socialmediadataclean_top_tickers.csv")
    ap.add_argument("--ticker-universe", default=None,
                    help="optional CSV with columns: ticker,company[,aliases]; aliases separated by ';'")
    ap.add_argument("--min_token_len", type=int, default=2, help="min token length for uppercase ticker tokens")
    return ap.parse_args()

# --------- text clean ---------
URL_RE = re.compile(r'https?://\S+|www\.\S+', flags=re.IGNORECASE)
NONALNUM_RE_KEEP_DOLLAR = re.compile(r"[^A-Za-z0-9\$\s]+")
MULTISPACE_RE = re.compile(r"\s+")

def clean_text_keep_cashtag(s: str) -> str:
    if not isinstance(s, str): s = "" if s is None else str(s)
    s = html.unescape(s)
    s = URL_RE.sub(" ", s)
    s = s.replace("&gt;", " ").replace("&lt;", " ").replace("&amp;", " ")
    s = NONALNUM_RE_KEEP_DOLLAR.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()
    return s

# --------- ticker universe & alias building ---------
DEFAULT_ALIAS = {
    # ticker : aliases (lowercase)
    "AMZN": ["amazon", "prime"],
    "AAPL": ["apple", "iphone", "ipad", "macbook"],
    "MSFT": ["microsoft", "windows", "azure"],
    "GOOG": ["google", "alphabet"],
    "META": ["facebook", "instagram", "whatsapp", "meta"],
    "NFLX": ["netflix"],
    "DIS":  ["disney", "hulu", "espn", "disney+"],
    "NVDA": ["nvidia", "geforce"],
    "TSLA": ["tesla", "elon"],
    "AMD":  ["amd", "ryzen", "radeon"],
    "GME":  ["gamestop"],
    "AMC":  ["amc"],
}

def load_universe(path: str | None) -> tuple[Set[str], Dict[str, List[str]]]:
    """
    Returns:
      (ticker_set, alias_dict_lowercase)
    """
    if path is None or not os.path.isfile(path):
        # default
        return set(DEFAULT_ALIAS.keys()), {k: [a.lower() for a in v] for k, v in DEFAULT_ALIAS.items()}

    df = pd.read_csv(path)
    need = {"ticker","company"}
    if not need.issubset(set(x.lower() for x in df.columns)):
        # try case-insensitive columns
        cols = {c.lower(): c for c in df.columns}
        if not need.issubset(set(cols.keys())):
            raise ValueError("ticker_universe must have columns: ticker, company[, aliases]")
        ticker_col = cols["ticker"]; company_col = cols["company"]
        aliases_col = cols.get("aliases")
    else:
        # exact match
        ticker_col, company_col = "ticker", "company"
        aliases_col = "aliases" if "aliases" in df.columns else None

    tickers = set(df[ticker_col].astype(str).str.upper().str.strip())
    alias_dict: Dict[str, List[str]] = {}
    for _, r in df.iterrows():
        t = str(r[ticker_col]).upper().strip()
        names = [str(r[company_col]).lower()]
        if aliases_col and pd.notna(r[aliases_col]):
            extra = [a.strip().lower() for a in str(r[aliases_col]).split(";") if a.strip()]
            names.extend(extra)
        alias_dict[t] = list(sorted(set(names)))
    return tickers, alias_dict

# --------- extraction ---------
CASH_TAG_RE = re.compile(r"\$([A-Z]{1,5})(?![A-Za-z])")

def extract_tickers_for_row(text_clean: str,
                            ticker_set: Set[str],
                            alias_dict: Dict[str, List[str]],
                            min_token_len: int = 2) -> List[str]:
    """
    Strategy:
      1) cashtags: $TSLA    -> TSLA
      2) plain uppercase tokens in universe: 'TSLA', 'NVDA'
      3) alias matches: 'tesla' -> TSLA (word-boundary, lowercase)
    """
    found: Set[str] = set()
    # 1) cashtags
    for m in CASH_TAG_RE.findall(text_clean):
        t = m.upper()
        if t in ticker_set:
            found.add(t)

    # 2) uppercase plain tokens
    tokens = text_clean.split()
    for tok in tokens:
        if not tok: continue
        if tok.startswith("$"):  # cashtags already handled
            continue
        if tok.isupper() and len(tok) >= min_token_len and len(tok) <= 5 and tok.isalpha():
            up = tok.upper()
            if up in ticker_set:
                found.add(up)

    # 3) aliases
    low = text_clean.lower()
    for t, aliases in alias_dict.items():
        for a in aliases:
            # use word boundary; allow '+' in disney+ by normalizing '+' -> ' plus '
            a_norm = a.replace("+", r"\+")
            if re.search(rf"\b{a_norm}\b", low):
                found.add(t)

    return sorted(found)

# --------- sentiment ---------
def get_sentiment_series(texts: pd.Series) -> pd.Series:
    """
    Try VADER; if lexicon missing, download once; if VADER unavailable, fallback to TextBlob.
    Returns a float series in [-1, 1].
    """
    try:
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        try:
            _ = SentimentIntensityAnalyzer()
        except Exception:
            nltk.download("vader_lexicon")
        sia = SentimentIntensityAnalyzer()
        return texts.fillna("").astype(str).apply(lambda x: sia.polarity_scores(x)["compound"])
    except Exception:
        try:
            from textblob import TextBlob
            def blob_score(t: str) -> float:
                p = TextBlob(t).sentiment.polarity  # [-1,1]
                return float(np.clip(p, -1.0, 1.0))
            return texts.fillna("").astype(str).apply(blob_score)
        except Exception:
            # final fallback: zeros
            return pd.Series(np.zeros(len(texts)), index=texts.index, dtype=float)

# --------- main pipeline ---------
def main():
    args = parse_args()
    df = pd.read_csv(args.inp)

    # Basic sanity
    needed_cols = {"kind","subreddit","id","created_utc","title","text","score","num_comments"}
    if not needed_cols.issubset(df.columns):
        raise ValueError(f"Input missing columns. Need at least: {sorted(needed_cols)}")

    # Merge text & clean
    df["title"] = df["title"].fillna("")
    df["text"]  = df["text"].fillna("")
    df["content"] = (df["title"].astype(str) + " " + df["text"].astype(str)).str.strip()
    df["content_clean"] = df["content"].apply(clean_text_keep_cashtag)

    # Basic filters: remove empty after cleaning
    df = df[df["content_clean"].str.len() > 0].copy()

    # Normalize time (UTC date)
    df["created_utc"] = pd.to_numeric(df["created_utc"], errors="coerce")
    df = df[pd.notna(df["created_utc"])].copy()
    df["date_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True).dt.date

    # Sentiment
    df["compound"] = get_sentiment_series(df["content_clean"])

    # Load ticker universe & aliases (or defaults)
    ticker_set, alias_dict = load_universe(args.ticker_universe)

    # Extract tickers per row
    tickers_list = []
    for s in df["content_clean"].tolist():
        tickers_list.append(extract_tickers_for_row(s, ticker_set, alias_dict, min_token_len=args.min_token_len))
    df["tickers"] = tickers_list

    # Keep rows with at least one matched ticker
    df = df[df["tickers"].map(len) > 0].copy()

    # Explode by ticker
    df_ex = df.explode("tickers").rename(columns={"tickers":"ticker"})
    df_ex["is_post"] = (df_ex["kind"].astype(str).str.lower() == "post").astype(int)
    df_ex["is_comment"] = (df_ex["kind"].astype(str).str.lower() == "comment").astype(int)

    # Row-level output
    out_rows = args.out_rows
    to_csv(df_ex[[
        "date_utc","ticker","kind","subreddit","id","parent_id","author","score","num_comments",
        "compound","title","text","content_clean"
    ]], out_rows)

    # Daily aggregation per ticker
    def _weighted_mean(x):
        w = x["score"].fillna(0).astype(float).values
        v = x["compound"].astype(float).values
        w = np.clip(w, 0, None)  # negative weights set to 0
        if w.sum() == 0:
            # avoid division by zero — fall back to unweighted mean
            return float(np.nanmean(v)) if len(v) else np.nan
        return float(np.average(v, weights=w))


    g = df_ex.groupby(["date_utc","ticker"], as_index=False).agg(
        n_posts=("is_post","sum"),
        n_comments=("is_comment","sum"),
        n_total=("ticker","count"),
        mean_compound=("compound","mean"),
        median_compound=("compound","median"),
        score_weighted_compound=("compound", lambda s: _weighted_mean(df_ex.loc[s.index, ["score","compound"]]))
    )
    out_daily = args.out_daily
    to_csv(g, out_daily)

    # Top discussed tickers overall
    top = df_ex.groupby("ticker", as_index=False).agg(
        total_mentions=("ticker","count"),
        unique_authors=("author", lambda s: s.fillna("").nunique()),
        across_days=("date_utc", "nunique"),
        mean_compound=("compound","mean")
    ).sort_values("total_mentions", ascending=False)
    out_top = args.out_top
    to_csv(top, out_top)

    print(f"[OK] Wrote row-level to {out_rows}")
    print(f"[OK] Wrote daily    to {out_daily}")
    print(f"[OK] Wrote toplist  to {out_top}")

if __name__ == "__main__":
    main()
