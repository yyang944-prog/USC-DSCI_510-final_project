"""
merge_for_model.py
------------------
Merge weekly social-media features with stock prices (daily → weekly),
and construct modeling targets including:
- next_week_return: next week's percentage return (per ticker)
- abnormal_volume: relative volume vs trailing N weeks' average (excludes current week)
- z_return: standardized next_week_return using expanding mean/std (no look-ahead)

Usage
-----
python src/merge_for_model.py \
  --social  data/processed/socialmediadataclean_weekly_6mo.csv \
  --prices  data/raw/prices_6mo.csv \
  --out     data/processed/socialmedia_price_panel_6mo.csv \
  --vol-window 8
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def ensure_dir(path: str) -> None:
    """Create parent directory of `path` if it does not exist.
    
    Parameters
    ----------
    path : str
        File path for which the parent directory should be ensured.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def build_weekly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily prices to weekly frequency aligned to Monday.
    
    Aggregation rules:
      - weekly_close: last 'Adj Close' of the week
      - weekly_volume: sum of 'Volume' in the week
      - weekly_return: pct change of `weekly_close` from the previous week (per ticker)
    
    Parameters
    ----------
    prices : pd.DataFrame
        Daily price table with at least ['date','ticker','Adj Close','Volume'].
    
    Returns
    -------
    pd.DataFrame
        Weekly price table with columns:
        ['ticker','week_start','weekly_close','weekly_volume','weekly_return'].
    """
    df = prices.copy()
    df = df.rename(columns={"Adj Close": "Adj_Close"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.sort_values(["ticker", "date"])
    df["week_start"] = df["date"].dt.to_period("W-MON").apply(lambda p: p.start_time.normalize())
    weekly = (
        df.groupby(["ticker", "week_start"], as_index=False)
        .agg(weekly_close=("Adj_Close", "last"), weekly_volume=("Volume", "sum"))
        .sort_values(["ticker", "week_start"])
    )
    weekly["weekly_return"] = weekly.groupby("ticker")["weekly_close"].pct_change()
    return weekly


def add_targets(weekly_prices: pd.DataFrame, vol_window: int = 8) -> pd.DataFrame:
    """Add next-week return, abnormal volume, and standardized return (z-score).
    
    Definitions
    ----------
    - next_week_return: lead(+1) of weekly_return per ticker.
    - abnormal_volume: relative deviation from the trailing `vol_window`-week average
      (excluding the current week), i.e. Volume_t / mean(Volume_{t-N:t-1}) - 1.
    - z_return: standardized next_week_return using expanding (past-only) mean/std.
    
    Parameters
    ----------
    weekly_prices : pd.DataFrame
        Weekly price table from `build_weekly_prices()`.
    vol_window : int, default=8
        Rolling window length (in weeks) for abnormal volume baseline.
    
    Returns
    -------
    pd.DataFrame
        Copy of input with new columns:
        ['next_week_return','abnormal_volume','z_return'].
    """
    df = weekly_prices.copy()
    df["next_week_return"] = df.groupby("ticker")["weekly_return"].shift(-1)
    vol = df.groupby("ticker")["weekly_volume"]
    trailing_mean = vol.transform(lambda s: s.rolling(vol_window, min_periods=1).mean().shift(1))
    df["abnormal_volume"] = df["weekly_volume"] / trailing_mean - 1
    grp = df.groupby("ticker")["next_week_return"]
    mean_prev = grp.transform(lambda s: s.expanding(min_periods=5).mean().shift(1))
    std_prev = grp.transform(lambda s: s.expanding(min_periods=5).std(ddof=0).shift(1))
    df["z_return"] = (df["next_week_return"] - mean_prev) / std_prev
    return df


def merge_social_with_prices(social_weekly: pd.DataFrame, weekly_targets: pd.DataFrame) -> pd.DataFrame:
    """Merge weekly social-media features with weekly price targets.
    
    The merge key is ['ticker','week_start']. Social columns are renamed
    for clarity and consistency with the project report.
    
    Parameters
    ----------
    social_weekly : pd.DataFrame
        Weekly social-media features (from aggregate_weekly.py).
    weekly_targets : pd.DataFrame
        Weekly price data with targets (from add_targets()).
    
    Returns
    -------
    pd.DataFrame
        Merged panel suitable for modeling.
    """
    social = social_weekly.copy()
    social["week_start"] = pd.to_datetime(social["week_start"]).dt.tz_localize(None)
    rename_map = {
        "n_total_week": "mentions_week",
        "mean_compound_w": "sentiment_mean_w",
        "score_weighted_compound_w": "sentiment_score_weighted_w",
    }
    social = social.rename(columns=rename_map)
    merged = social.merge(weekly_targets, on=["ticker", "week_start"], how="left")
    return merged


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed arguments with attributes: social, prices, out, vol_window.
    """
    ap = argparse.ArgumentParser(description="Merge weekly social features with prices and create targets.")
    ap.add_argument("--social", required=True, help="Path to weekly social CSV (from aggregate_weekly.py).")
    ap.add_argument("--prices", required=True, help="Path to daily prices CSV (from get_data.py).")
    ap.add_argument("--out", required=True, help="Output CSV path for the merged modeling panel.")
    ap.add_argument("--vol-window", type=int, default=8, help="Rolling window (weeks) for abnormal volume baseline.")
    return ap.parse_args()


def main() -> None:
    """Entry point: load data, build weekly targets, merge, and save."""
    args = parse_args()
    social = pd.read_csv(args.social)
    prices = pd.read_csv(args.prices)
    weekly_prices = build_weekly_prices(prices)
    weekly_targets = add_targets(weekly_prices, vol_window=args.vol_window)
    merged = merge_social_with_prices(social, weekly_targets)
    merged = merged[merged["next_week_return"].notna()].copy()
    ensure_dir(args.out)
    merged.to_csv(args.out, index=False)
    print(f"[OK] Wrote merged panel: {args.out} (shape={merged.shape})")
    

if __name__ == "__main__":
    main()
