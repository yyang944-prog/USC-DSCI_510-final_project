"""
get_data.py  — Robust Reddit collector (with progress logs) + Yahoo prices

Overview
--------
This script collects social-media posts/comments from Reddit and daily prices
from Yahoo Finance (yfinance), matching the DSCI 510 "Data Collection" stage.

Reddit backends (auto-chosen unless flags override):
1) PRAW (OAuth, best): requires env variables
   - REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, (optional) REDDIT_USER_AGENT
2) Pullpush mirror (pushshift-like): NO key required, **submissions only** (comments often unavailable)
3) Reddit Public JSON (search/top/new/comments): NO key required, may be rate-limited

Backend selection rules
-----------------------
- If credentials exist -> try **PRAW**; on failure, fallback to Pullpush -> Public JSON
- If no credentials:
    - default: try **Pullpush**; if empty -> **Public JSON**
    - with `--no-pullpush`: **skip Pullpush** and go **Public JSON** directly

New in this version
-------------------
- **--no-pullpush** flag to disable Pullpush explicitly.
- Progress logs every ~10 seconds: backend, subreddit, posts/comments counts.
- Gentle rate limiting + retry for public endpoints.

Outputs
-------
- Reddit: CSV with columns
  (kind, subreddit, id, parent_id, created_utc, author, score, num_comments, title, text).
- Prices: CSV with columns
  (date, ticker, Open, High, Low, Close, Adj Close, Volume).

Usage
-----
# Reddit (three months)
python -u src/get_data.py reddit \
  --start 2025-08-15 --end 2025-11-15 \
  --subs r/stocks r/wallstreetbets \
  --posts-per-sub 1500 --comments-per-post 200 \
  --out data/raw/reddit_posts.csv

# Force Public JSON (skip Pullpush)
python -u src/get_data.py reddit \
  --start 2025-05-15 --end 2025-11-15 \
  --subs r/stocks r/wallstreetbets r/investing r/StockMarket r/options r/RobinHood \
         r/pennystocks r/shortsqueeze r/daytrading r/techstocks \
  --posts-per-sub 2000 --comments-per-post 300 \
  --out data/raw/reddit_posts_full_6mo.csv \
  --no-pullpush

# Prices
python src/get_data.py prices \
  --start 2025-08-15 --end 2025-11-15 \
  --tickers TSLA NVDA AAPL AMD GME \
  --out data/raw/prices.csv

Environment (optional for PRAW)
-------------------------------
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export REDDIT_USER_AGENT="dsci510-project-bot"

Notes
-----
- If you see 403/429 on public JSON, try again later, split the date window, or switch networks.
- Progress logs print roughly every 10 seconds (not on every item).
- For VADER later, remember:
  python -c "import nltk; nltk.download('vader_lexicon')"
"""
from __future__ import annotations
import argparse, os, time, datetime as dt
from typing import List, Dict, Any, Optional
import pandas as pd
import requests

# ---------------------- Small I/O helpers ----------------------
def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def to_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(path); df.to_csv(path, index=False)

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_r = sub.add_parser("reddit", help="collect Reddit posts and top-level comments")
    ap_r.add_argument("--start", required=True, type=str, help="YYYY-MM-DD")
    ap_r.add_argument("--end", required=True, type=str, help="YYYY-MM-DD")
    ap_r.add_argument("--subs", nargs="+",
                      default=["r/stocks","r/wallstreetbets"],
                      help="e.g. r/stocks r/wallstreetbets r/investing ...")
    ap_r.add_argument("--posts-per-sub", type=int, default=500, help="max posts per subreddit")
    ap_r.add_argument("--comments-per-post", type=int, default=150, help="max comments per post")
    ap_r.add_argument("--out", required=True, type=str, help="output CSV path")
    ap_r.add_argument("--no-pullpush", action="store_true",
                      help="Skip Pullpush mirror; go straight to Reddit Public JSON if no PRAW creds")

    ap_p = sub.add_parser("prices", help="download prices via yfinance")
    ap_p.add_argument("--start", required=True, type=str)
    ap_p.add_argument("--end", required=True, type=str)
    ap_p.add_argument("--tickers", nargs="+", required=True)
    ap_p.add_argument("--out", required=True, type=str)
    return ap.parse_args()

# ---------------------- Progress logger ----------------------
class Progress:
    """Lightweight progress reporter that prints every ~10 seconds."""
    def __init__(self, tag: str, interval_sec: float = 10.0) -> None:
        self.tag = tag
        self.interval = interval_sec
        self.last = time.time()
        self.count_posts = 0
        self.count_comments = 0
        self.current_sub = ""
        self.backend = ""

    def set_backend(self, name: str) -> None:
        self.backend = name
        self.heartbeat("switch-backend")

    def set_sub(self, sub: str) -> None:
        self.current_sub = sub
        self.heartbeat("switch-sub")

    def add_posts(self, n: int) -> None:
        self.count_posts += n
        self._maybe_print()

    def add_comments(self, n: int) -> None:
        self.count_comments += n
        self._maybe_print()

    def heartbeat(self, note: str = "") -> None:
        now = time.time()
        if now - self.last >= self.interval:
            print(f"[{self.tag}] backend={self.backend} sub={self.current_sub} "
                  f"posts={self.count_posts} comments={self.count_comments} {note} ...", flush=True)
            self.last = now

    def _maybe_print(self) -> None:
        now = time.time()
        if now - self.last >= self.interval:
            print(f"[{self.tag}] backend={self.backend} sub={self.current_sub} "
                  f"posts={self.count_posts} comments={self.count_comments} ...", flush=True)
            self.last = now

# ---------------------- Mode 1: PRAW (OAuth) ----------------------
def reddit_via_praw(start_ts: int, end_ts: int, subs: List[str],
                    posts_per_sub: int, comments_per_post: int,
                    prog: Progress) -> pd.DataFrame:
    import praw
    from praw.models import MoreComments

    cid = os.getenv("REDDIT_CLIENT_ID")
    csec = os.getenv("REDDIT_CLIENT_SECRET")
    ua = os.getenv("REDDIT_USER_AGENT", "dsci510-project-bot")
    if not cid or not csec:
        raise RuntimeError("No Reddit credentials")
    reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)

    rows: List[Dict[str, Any]] = []
    prog.set_backend("PRAW")
    for sub_name in subs:
        prog.set_sub(sub_name)
        sub = reddit.subreddit(sub_name.replace("r/",""))
        search_iter = sub.search(query="*", sort="new", limit=posts_per_sub)
        for s in search_iter:
            created = int(s.created_utc or 0)
            if not (start_ts <= created <= end_ts):
                continue
            rows.append(dict(
                kind="post", subreddit=sub_name, id=s.id, parent_id=None,
                created_utc=created, author=str(s.author) if s.author else None,
                score=int(s.score) if s.score is not None else None,
                num_comments=int(s.num_comments) if s.num_comments is not None else None,
                title=s.title or "", text=s.selftext or ""
            ))
            prog.add_posts(1)
            # comments
            try:
                s.comments.replace_more(limit=0)
                count = 0
                for c in s.comments.list():
                    if isinstance(c, MoreComments):
                        continue
                    c_created = int(c.created_utc or 0)
                    if not (start_ts <= c_created <= end_ts):
                        continue
                    rows.append(dict(
                        kind="comment", subreddit=sub_name, id=c.id, parent_id=c.parent_id,
                        created_utc=c_created, author=str(c.author) if c.author else None,
                        score=int(c.score) if c.score is not None else None,
                        num_comments=None, title="", text=c.body or ""
                    ))
                    count += 1
                    prog.add_comments(1)
                    if count >= comments_per_post:
                        break
            except Exception:
                pass
            time.sleep(0.2)
    return pd.DataFrame(rows)

# ---------------------- Mode 2: Pullpush mirror ----------------------
PULLPUSH = "https://api.pullpush.io/reddit/search"
UA = {"User-Agent": "dsci510-project-bot/1.0 (+https://example.edu/)"}

def _pp_get(kind: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(f"{PULLPUSH}/{kind}/", params=params, headers=UA, timeout=20)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def reddit_via_pullpush(start_ts: int, end_ts: int, subs: List[str],
                        posts_per_sub: int, comments_per_post: int,
                        prog: Progress) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prog.set_backend("Pullpush")
    for sub in subs:
        name = sub.replace("r/","")
        prog.set_sub(sub)
        fetched = 0
        after = start_ts
        while fetched < posts_per_sub:
            js = _pp_get("submission", {
                "subreddit": name,
                "after": after,
                "before": end_ts,
                "sort": "desc",
                "sort_type": "created_utc",
                "size": min(100, posts_per_sub - fetched)
            })
            if not js or not js.get("data"):
                break
            created_list = []
            for d in js["data"]:
                created = int(d.get("created_utc", 0))
                if not (start_ts <= created <= end_ts):
                    continue
                pid = d.get("id")
                rows.append(dict(
                    kind="post", subreddit=f"r/{name}", id=pid, parent_id=None,
                    created_utc=created, author=d.get("author"),
                    score=d.get("score"), num_comments=d.get("num_comments"),
                    title=d.get("title") or "", text=d.get("selftext") or ""
                ))
                fetched += 1
                prog.add_posts(1)
                # NOTE: Pullpush comments endpoint is often unavailable → comments not fetched here
                created_list.append(created)
            after = min(created_list) - 1 if created_list else after
            if after <= start_ts:
                break
            time.sleep(0.4)
    return pd.DataFrame(rows)

# ---------------------- Mode 3: Public JSON ----------------------
HEADERS = UA
FIN_KEYWORDS = ["stock","stocks","price","prices","buy","sell","calls","puts","option","earnings","guidance","split"]

def _safe_get(url: str, params: Optional[Dict[str, str]] = None,
              retries: int = 3, sleep_sec: float = 3.0,
              prog: Optional[Progress] = None) -> Optional[Dict[str, Any]]:
    for i in range(retries):
        try:
            if prog: prog.heartbeat(f"GET {url.split('/r/')[-1]} try={i+1}/{retries}")
            r = requests.get(url, headers=HEADERS, params=params, timeout=20)
            if r.status_code == 429:
                if prog: print("[rate-limit] sleep 60s ...", flush=True)
                time.sleep(60.0); continue
            if r.status_code != 200:
                if prog: prog.heartbeat(f"status={r.status_code}, retry in {sleep_sec}s")
                time.sleep(sleep_sec); continue
            return r.json()
        except Exception:
            if prog: prog.heartbeat(f"exception, retry in {sleep_sec}s")
            time.sleep(sleep_sec)
    return None

def _fetch_comments_public(post_id: str, limit: int, prog: Progress) -> List[Dict[str, Any]]:
    url = f"https://www.reddit.com/comments/{post_id}.json"
    js = _safe_get(url, prog=prog)
    rows: List[Dict[str, Any]] = []
    if not js or not isinstance(js, list) or len(js) < 2:
        return rows
    try:
        children = js[1]["data"]["children"]
    except Exception:
        return rows
    for ch in children:
        data = ch.get("data", {})
        if data.get("body") is None:
            continue
        rows.append(dict(
            kind="comment", subreddit=data.get("subreddit_name_prefixed"),
            id=data.get("id"), parent_id=data.get("parent_id"),
            created_utc=int(data.get("created_utc", 0)), author=data.get("author"),
            score=data.get("score"), num_comments=None, title="", text=data.get("body") or ""
        ))
        prog.add_comments(1)
        if len(rows) >= limit:
            break
    return rows

def _collect_from_listing(sub: str, endpoint: str, params: Dict[str, str],
                          start_ts: int, end_ts: int, max_posts: int,
                          comments_per_post: int, prog: Progress) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    after: Optional[str] = None
    while len(out) < max_posts:
        q = dict(params)
        if after:
            q["after"] = after
        js = _safe_get(f"https://www.reddit.com/r/{sub}/{endpoint}", q, prog=prog)
        if not js:
            break
        children = js.get("data", {}).get("children", [])
        if not children:
            break
        for ch in children:
            d = ch.get("data", {})
            created = int(d.get("created_utc", 0))
            if not (start_ts <= created <= end_ts):
                continue
            out.append(dict(
                kind="post", subreddit=d.get("subreddit_name_prefixed"),
                id=d.get("id"), parent_id=None, created_utc=created, author=d.get("author"),
                score=d.get("score"), num_comments=d.get("num_comments"),
                title=d.get("title") or "", text=d.get("selftext") or ""
            ))
            prog.add_posts(1)
            comments = _fetch_comments_public(d.get("id",""), comments_per_post, prog)
            out.extend(comments)
            if len(out) >= max_posts:
                break
        after = js.get("data", {}).get("after")
        if not after:
            break
        time.sleep(0.4)
    return out

def reddit_via_public(start_ts: int, end_ts: int, subs: List[str],
                      posts_per_sub: int, comments_per_post: int,
                      prog: Progress) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    prog.set_backend("PublicJSON")
    for sub in subs:
        prog.set_sub(sub)
        fetched = 0
        # A) search.json with timestamp + finance keywords
        for kw in [""] + FIN_KEYWORDS:
            q = f"timestamp:{start_ts}..{end_ts} {kw}".strip()
            chunk = _collect_from_listing(
                sub=sub.replace("r/",""),
                endpoint="search.json",
                params={"q": q, "restrict_sr": "on", "sort": "new", "limit": "100"},
                start_ts=start_ts, end_ts=end_ts,
                max_posts=min(posts_per_sub - fetched, 300),
                comments_per_post=comments_per_post, prog=prog
            )
            rows.extend(chunk); fetched += len(chunk)
            if fetched >= posts_per_sub:
                break
            time.sleep(0.5)
        # B) top.json (year/month)
        if fetched < posts_per_sub:
            for t in ["year","month"]:
                chunk = _collect_from_listing(
                    sub=sub.replace("r/",""),
                    endpoint="top.json",
                    params={"t": t, "limit": "100"},
                    start_ts=start_ts, end_ts=end_ts,
                    max_posts=posts_per_sub - fetched,
                    comments_per_post=comments_per_post, prog=prog
                )
                rows.extend(chunk); fetched += len(chunk)
                if fetched >= posts_per_sub:
                    break
                time.sleep(0.5)
        # C) new.json
        if fetched < posts_per_sub:
            chunk = _collect_from_listing(
                sub=sub.replace("r/",""),
                endpoint="new.json",
                params={"limit": "100"},
                start_ts=start_ts, end_ts=end_ts,
                max_posts=posts_per_sub - fetched,
                comments_per_post=comments_per_post, prog=prog
            )
            rows.extend(chunk); fetched += len(chunk)
    return pd.DataFrame(rows)

# ---------------------- command handlers ----------------------
def cmd_reddit(args: argparse.Namespace) -> None:
    start_ts = int(dt.datetime.fromisoformat(args.start).timestamp())
    end_ts = int(dt.datetime.fromisoformat(args.end).timestamp())
    has_creds = bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))
    prog = Progress(tag="reddit", interval_sec=10.0)

    if has_creds:
        try:
            print("Using PRAW (OAuth)…", flush=True)
            df = reddit_via_praw(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)
        except Exception as e:
            print(f"PRAW failed: {e}. Trying Pullpush mirror…", flush=True)
            if args.no_pullpush:
                print("Skip Pullpush due to --no-pullpush; using Reddit Public JSON…", flush=True)
                df = reddit_via_public(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)
            else:
                df = reddit_via_pullpush(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)
                if df.empty:
                    print("Pullpush empty. Falling back to Reddit Public JSON…", flush=True)
                    df = reddit_via_public(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)
    else:
        if args.no_pullpush:
            print("No credentials. --no-pullpush set → using Reddit Public JSON…", flush=True)
            df = reddit_via_public(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)
        else:
            print("No credentials. Trying Pullpush mirror first…", flush=True)
            df = reddit_via_pullpush(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)
            if df.empty:
                print("Pullpush empty. Falling back to Reddit Public JSON…", flush=True)
                df = reddit_via_public(start_ts, end_ts, args.subs, args.posts_per_sub, args.comments_per_post, prog)

    df = df.drop_duplicates(subset=["kind","id"])
    to_csv(df, args.out)
    print(f"Wrote {len(df)} rows to {args.out}", flush=True)

def cmd_prices(args: argparse.Namespace) -> None:
    """
    Download daily stock prices for a list of tickers (yfinance) and
    normalize columns so the final CSV always has:
    ['date','ticker','Open','High','Low','Close','Adj Close','Volume'].

    Handles:
    - MultiIndex columns like ('TSLA','Open') by flattening
    - Suffix/Prefix around field names like 'TSLA_Open', 'Open_TSLA'
    - 'AdjClose' vs 'Adj Close' vs missing Adj Close
    """
    import yfinance as yf
    import pandas as pd
    import numpy as np
    import re

    start, end = args.start, args.end
    all_frames = []

    for t in args.tickers:
        # Ask yfinance to keep 'Adj Close' and avoid column groups by ticker
        df = yf.download(
            t,
            start=start,
            end=end,
            interval="1d",
            progress=False,
            auto_adjust=False,   # keep 'Adj Close'
            group_by="column",   # prefer columns like 'Open','Close',...
            actions=False,
        )

        # If MultiIndex columns -> flatten to strings
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join([str(x) for x in col if x and str(x) != ""]).strip("_")
                for col in df.columns.values
            ]

        # Bring Date back as a column
        df = df.reset_index()
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "date"})
        else:
            df = df.rename(columns={df.columns[0]: "date"})

        # Build a robust mapper that finds the real column behind each target field
        # It will accept variants like 'Open', 'TSLA_Open', 'Open_TSLA', 'AdjClose', etc.
        target_keys = {
            "open": "Open",
            "high": "High",
            "low":  "Low",
            "close": "Close",
            "adjclose": "Adj Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }

        found = {}
        for col in df.columns:
            col_norm = col.lower().replace(" ", "")
            # Split by non-alphas to capture suffix/prefix variants
            parts = re.split(r"[^a-z]+", col_norm)
            # Check last and first token (covers 'tsla_open' and 'open_tsla')
            candidates = []
            if parts:
                candidates.append(parts[-1])
                candidates.append(parts[0])
            for c in candidates:
                if c in target_keys and target_keys[c] not in found:
                    found[target_keys[c]] = col

        # If 'Adj Close' truly missing, fallback to Close
        if "Adj Close" not in found and "Close" in found:
            df["Adj Close"] = df[found["Close"]]
            found["Adj Close"] = "Adj Close"

        # Add ticker column
        df["ticker"] = t

        # Assemble final frame with whatever we found, preserving order
        final_cols = ["date", "ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        take = ["date", "ticker"]
        for k in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            if k in found:
                take.append(found[k])
            else:
                # create empty column if missing
                df[k] = np.nan
                take.append(k)

        out_df = df[take].copy()
        # Normalize column names to canonical
        out_df.columns = final_cols

        # Ensure date is tz-naive datetime
        out_df["date"] = pd.to_datetime(out_df["date"]).dt.tz_localize(None)

        all_frames.append(out_df)

    out = pd.concat(all_frames, ignore_index=True)
    ensure_dir(args.out)
    out.to_csv(args.out, index=False)
    print(f"Wrote prices for {len(args.tickers)} tickers to {args.out} (rows={len(out)})", flush=True)


if __name__ == "__main__":
    ns = parse_args()
    if ns.cmd == "reddit":
        cmd_reddit(ns)
    elif ns.cmd == "prices":
        cmd_prices(ns)
