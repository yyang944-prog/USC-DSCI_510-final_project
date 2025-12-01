"""
aggregate_weekly.py - Convert daily social-media aggregates to weekly features (per ticker x week).

Input
-----
- data/processed/socialmediadataclean_daily_6mo.csv  (from clean_social_media.py)
  Required columns:
    ['date_utc','ticker','n_posts','n_comments','n_total',
     'mean_compound','median_compound','score_weighted_compound']

Output
------
- data/processed/socialmediadataclean_weekly_6mo.csv

Aggregation design
------------------
- week_start: Monday of each ISO week (W-MON)
- Sum: n_posts, n_comments, n_total
- Weighted means (weight = daily n_total, avoid zero-sum): mean_compound_w, score_weighted_compound_w
- Median-of-medians: median_compound_week (simple median over daily medians within week)
- days_covered: count of distinct days contributing to the week

Usage
-----
python src/aggregate_weekly.py   --in  data/processed/socialmediadataclean_daily_6mo.csv   --out data/processed/socialmediadataclean_weekly_6mo.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="daily-level CSV path")
    ap.add_argument("--out", dest="outp", required=True, help="weekly-level CSV path")
    return ap.parse_args()

def safe_weighted_mean(values, weights):
    v = pd.Series(values, dtype=float)
    w = pd.Series(weights, dtype=float).clip(lower=0).fillna(0.0)
    if float(w.sum()) == 0.0:
        return float(v.mean()) if len(v) else np.nan
    return float(np.average(v, weights=w))

def main():
    args = parse_args()
    df = pd.read_csv(args.inp)
    needed = {'date_utc','ticker','n_posts','n_comments','n_total',
              'mean_compound','median_compound','score_weighted_compound'}
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing columns: {missing}")

    df['date_utc'] = pd.to_datetime(df['date_utc']).dt.tz_localize(None)
    week_start = df['date_utc'].dt.to_period('W-MON').apply(lambda p: p.start_time)
    df['week_start'] = week_start.dt.normalize()

    def agg_func(x):
        w = x['n_total'].clip(lower=0).fillna(0.0).values
        return pd.Series({
            'n_posts_week': int(x['n_posts'].sum()),
            'n_comments_week': int(x['n_comments'].sum()),
            'n_total_week': int(x['n_total'].sum()),
            'mean_compound_w': safe_weighted_mean(x['mean_compound'].values, w),
            'score_weighted_compound_w': safe_weighted_mean(x['score_weighted_compound'].values, w),
            'median_compound_week': float(x['median_compound'].median()),
            'days_covered': int(x['date_utc'].nunique()),
        })

    weekly = df.groupby(['ticker','week_start'], as_index=False).apply(agg_func)

    ensure_dir(args.outp)
    weekly.to_csv(args.outp, index=False)
    print(f"[OK] Wrote weekly file: {args.outp} (shape={weekly.shape})")

if __name__ == "__main__":
    main()
