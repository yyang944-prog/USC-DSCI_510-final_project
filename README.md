# DSCI510_FinalProject — Do Social Media Sentiments Drive U.S. Stock Market Reactions?

This repository builds an **end‑to‑end, reproducible pipeline** that collects Reddit discussions about US stocks, cleans and labels the texts (tickers + sentiment), aggregates to **weekly features**, merges with **Yahoo Finance** prices, and produces a modeling panel for **return/volume** analysis and visualizations.

## Team
- Mengyue Niu — USC ID: 5926-7082-41 - USC Email: mengyuen@usc.edu - Github Username: MengyueNiu
- Yichen Yang — USC ID: 4403-8375-34 - USC Email: yyang944@usc.edu - Github Username: yyang944-prog

## Environment (create a clean virtualenv)
```bash
conda create -n dsci510 python=3.11 -y
conda activate dsci510

pip install --upgrade pip
pip install -r requirements.txt

python -c "import nltk; nltk.download('vader_lexicon')"
```

## Folder Layout
```
DSCI510_FinalProject/
├── data/
│   ├── raw/
│   │   ├── reddit_posts.csv
│   │   ├── reddit_posts_small.csv
│   │   ├── reddit_posts_full.csv
│   │   ├── reddit_posts_full_6mo.csv         # final Reddit raw (6 months)
│   │   └── prices_6mo.csv                    # final daily stock data (6 months)
│   ├── processed/
│   │   ├── socialmediadataclean_rows_6mo.csv   # rows: post/comment × ticker (with sentiment)
│   │   ├── socialmediadataclean_daily_6mo.csv  # daily: (date, ticker) aggregation
│   │   ├── socialmediadataclean_weekly_6mo.csv # weekly features (final social input for merge)
│   │   ├── socialmediadataclean_top_tickers_6mo.csv
│   │   ├── socialmedia_price_panel_6mo.csv     # merged panel (weekly) — final for analysis
│   │   └── Dataset_Documentation.txt
│   └── reference/
│       └── ticker_universe.csv               # ticker, company, aliases (matching dictionary)
├── results/
│   ├── final_report.pdf
│   ├── visualization.ipynb
│   ├── correlation_matrices.xlsx             # Excel file containing correlation matrices (Sentiment vs. Returns, Volume vs. Mentions)
│   ├── descriptive_statistics.xlsx           # Excel file with summary statistics (Overall and per-ticker performance)
│   ├── fig1_correlation_heatmaps.png         # Heatmap visualization of the correlation matrices
│   ├── fig2_price_vs_sentiment.png           # Dual-axis time series plot comparing stock price trends with sentiment scores
│   ├── fig3_enhanced_sentiment_comparison.png # Violin/Box plots showing return distributions across sentiment categories
│   ├── fig4A_overall_sentiment_distribution.png # Histogram showing the distribution of sentiment scores across all data
│   ├── fig4B_ticker_sentiment_distributions.png # Histograms showing sentiment score distributions for individual tickers
│   ├── fig5_discussion_vs_trading_volume.png # Time series plot comparing online discussion volume vs. abnormal trading volume
│   ├── fig6_volume_vs_mentions_by_ticker.png # Scatter plots with regression lines showing the relationship between mentions and volume
│   ├── hypothesis_testing.txt                # Text output of T-test results (High vs. Low sentiment weeks)
│   └── strategy_simulation.txt               # Text output of the simple sentiment-based trading strategy simulation results
├── src/
│   ├── utils/
│   │   ├── io_utils.py
│   │   ├── sentiment.py
│   │   └── ticker_detect.py
│   ├── get_data.py
│   ├── clean_social_media.py
│   ├── aggregate_weekly.py
│   ├── merge_for_model.py
│   ├── run_analysis.py
│   └── visualize_results.py
├── README.md
├── requirements.txt
└── project_proposal.pdf
```


## 1) Data Collection — `src/get_data.py`

### Purpose
- **Reddit**: collect posts + top‑level comments from multiple subreddits over a date range, saving text, meta, and scores.
- **Prices**: download **daily OHLCV** from Yahoo Finance for selected tickers.

### Reddit (no API key path used in this project)
We implemented a **credential‑aware** fetcher:
- If Reddit API credentials are available, use **PRAW**;
- Else automatically fall back to **public JSON** and a mirror endpoint;
- We used the **fallback** route in this project.

**Command (6 months window, multi‑subreddit):**
```bash
python src/get_data.py reddit \
  --start 2025-05-15 --end 2025-11-15 \
  --subs r/stocks r/wallstreetbets r/investing r/StockMarket r/options r/RobinHood \
         r/pennystocks r/shortsqueeze r/daytrading r/techstocks \
  --posts-per-sub 2000 --comments-per-post 300 \
  --out data/raw/reddit_posts_full_6mo.csv
```

**Output**
- `data/raw/reddit_posts_full_6mo.csv` — raw posts/comments with columns like:
  `kind, subreddit, id, parent_id, created_utc, author, score, num_comments, title, text`.

### Prices (Yahoo Finance via yfinance)
Based on the cleaned Reddit dataset  `socialmediadataclean_top_tickers_6mo.csv`, select the top ten stocks with the highest discussion volume for data collection. We fixed column compatibility to always emit canonical OHLCV (including `Adj Close`).

**Command**
```bash
python src/get_data.py prices \
  --start 2025-05-15 --end 2025-11-15 \
  --tickers TSLA GOOG NVDA MSFT GME PLTR AMZN WMT AAPL INTC \
  --out data/raw/prices_6mo.csv
```

**Output**
- `data/raw/prices_6mo.csv` — daily prices with columns:
  `date, ticker, Open, High, Low, Close, Adj Close, Volume`.

---

## 2) Social Cleaning & Sentiment — `src/clean_social_media.py`

### Purpose
- De‑duplicate, remove ads/spam, normalize time to UTC date;
- **Ticker detection** using `reference/ticker_universe.csv` (tickers + aliases);
- **Sentiment** via VADER (`compound` in [-1,1], plus neg/neu/pos);
- Multi‑level outputs: **rows** (exploded by ticker), **daily**, and **top tickers**.

**Command**
```bash
python src/clean_social_media.py \
  --in data/raw/reddit_posts_full_6mo.csv \
  --ticker-universe data/reference/ticker_universe.csv \
  --out-rows  data/processed/socialmediadataclean_rows_6mo.csv \
  --out-daily data/processed/socialmediadataclean_daily_6mo.csv \
  --out-top   data/processed/socialmediadataclean_top_tickers_6mo.csv
```

**Outputs**
- `socialmediadataclean_rows_6mo.csv` — row‑level; each text × matched ticker; includes `compound`.
- `socialmediadataclean_daily_6mo.csv` — `(date_utc, ticker)`; fields: `n_posts, n_comments, n_total, mean_compound, median_compound, score_weighted_compound`.
- `socialmediadataclean_top_tickers_6mo.csv` — ranked by mentions, unique authors, active days, mean sentiment.

---

## 3) Daily → Weekly Social Aggregation — `src/aggregate_weekly.py`

### Purpose
- Convert **daily** social stats to **weekly** (week starts on **Monday**) to improve signal‑to‑noise.

**Command**
```bash
python src/aggregate_weekly.py \
  --in  data/processed/socialmediadataclean_daily_6mo.csv \
  --out data/processed/socialmediadataclean_weekly_6mo.csv
```

**Output**
- `socialmediadataclean_weekly_6mo.csv` — weekly per `(ticker, week_start)`:
  totals and average sentiment (e.g., `n_total_week`, `mean_compound_w`, `score_weighted_compound_w`).

---

## 4) Merge + Targets (Weekly Panel) — `src/merge_for_model.py`

### Purpose
- Aggregate **daily prices** to weekly and compute **weekly_return**;
- Construct modeling targets (per ticker):
  - `next_week_return` — lead(+1) of weekly return;
  - `abnormal_volume` — deviation from trailing average **(window configurable)**;
  - `z_return` — standardized next_week_return using expanding past mean/std;
- Merge social features with price targets by `(ticker, week_start)`.

**Command**
```bash
python src/merge_for_model.py \
  --social data/processed/socialmediadataclean_weekly_6mo.csv \
  --prices data/raw/prices_6mo.csv \
  --out    data/processed/socialmedia_price_panel_6mo.csv \
  --vol-window 8
```

**Output**
- `socialmedia_price_panel_6mo.csv` — final weekly panel with columns such as:
  `ticker, week_start, mentions_week, sentiment_mean_w, sentiment_score_weighted_w,
   weekly_close, weekly_volume, weekly_return, next_week_return, abnormal_volume, z_return`.

---

## 5) Analysis — `src/run_analysis.py`

### Purpose
- **Descriptive Statistics:** Summarizes key metrics (sentiment, volume, returns) overall and by ticker.
- **Correlation Analysis:** Computes Pearson correlation matrices to test relationships between sentiment/returns and discussion/trading volume.
- **Hypothesis Testing:** Performs T-tests to check if "high sentiment" weeks yield significantly higher returns than "low sentiment" weeks.
- **Strategy Simulation:** Simulates a simple sentiment-based trading strategy to quantify potential economic value.

**Command**
```bash
python src/run_analysis.py \
  --in  data/processed/socialmedia_price_panel_6mo.csv \
  --out-dir results
```

**Outputs**
- results/`descriptive_statistics.xlsx` — General statistics and ticker performance ranking.
- results/`correlation_matrices.xlsx` — Correlation tables for sentiment-returns and discussion_volume-transaction_volume.
- results/`hypothesis_testing.txt` — T-test results comparing high vs. low sentiment groups.
- results/`strategy_simulation.txt` — Performance comparison of sentiment strategy vs. buy-and-hold.

---

## 6) Visualization — `src/visualize_results.py`

### Purpose
- **Correlation Analysis:** Generates heatmaps to visualize the strength of relationships between sentiment/returns and discussion/trading volume.
- **Time Series Trends:** Plots dual-axis charts for Stock Price vs. Sentiment Score and Discussion Volume vs. Abnormal Volume over time.
- **Statistical Distributions:** Visualizes the distribution of sentiment scores (overall and per ticker) using histograms.
- **Relationship Plots:** Creates box/violin plots for returns across sentiment categories and scatter plots with regression lines for discussion vs. trading volume.

**Command**
```bash
python src/visualize_results.py \
  --in  data/processed/socialmedia_price_panel_6mo.csv \
  --out-dir results
```

**Outputs**
- results/`fig1_correlation_heatmaps.png` — Heatmaps of correlation matrices.
- results/`fig2_price_vs_sentiment.png` — Dual-axis time series plots per ticker.
- results/`fig3_enhanced_sentiment_comparison.png` — Returns distribution by sentiment category.
- results/`fig4A_overall_sentiment_distribution.png` — Overall sentiment histogram.
- results/`fig4B_ticker_sentiment_distributions.png` — Per-ticker sentiment histograms.
- results/`fig5_discussion_vs_trading_volume.png` — Time series of mentions vs. abnormal volume.
- results/`fig6_volume_vs_mentions_by_ticker.png` — Scatter plots with regression lines.

---

## Notes & Assumptions

- **Reddit access**: If API credentials are missing or rate‑limited, the script automatically uses public JSON endpoints and a mirror. All requests send a custom `User-Agent` string.
- **Ticker universe**: You can extend `data/reference/ticker_universe.csv` with new tickers or aliases (e.g., common names, cashtags like `$TSLA`). The cleaner will pick them up automatically.
- **Why weekly?** Reddit posts are sparse and bursty at daily level; weekly aggregation increases signal‑to‑noise and aligns naturally with the “next‑week” hypothesis.
- **Return choice**: Use **Adjusted Close** for return/abnormal calculations to reflect true investor P/L after corporate actions.

---

## End‑to‑End Quick Run

```bash
# 1) Collect Reddit data (6 months)
python src/get_data.py reddit \
  --start 2025-05-15 --end 2025-11-15 \
  --subs r/stocks r/wallstreetbets r/investing r/StockMarket r/options r/RobinHood \
         r/pennystocks r/shortsqueeze r/daytrading r/techstocks \
  --posts-per-sub 2000 --comments-per-post 300 \
  --out data/raw/reddit_posts_full_6mo.csv

# 2) Clean social media data + sentiment + daily aggregation
python src/clean_social_media.py \
  --in data/raw/reddit_posts_full_6mo.csv \
  --ticker-universe data/reference/ticker_universe.csv \
  --out-rows  data/processed/socialmediadataclean_rows_6mo.csv \
  --out-daily data/processed/socialmediadataclean_daily_6mo.csv \
  --out-top   data/processed/socialmediadataclean_top_tickers_6mo.csv

# 3) Daily → Weekly
python src/aggregate_weekly.py \
  --in  data/processed/socialmediadataclean_daily_6mo.csv \
  --out data/processed/socialmediadataclean_weekly_6mo.csv

# 4) Collect stock prices (6 months)
python src/get_data.py prices \
  --start 2025-05-15 --end 2025-11-15 \
  --tickers TSLA GOOG NVDA MSFT GME PLTR AMZN WMT AAPL INTC \
  --out data/raw/prices_6mo.csv

# 5) Merge social × prices + targets
python src/merge_for_model.py \
  --social data/processed/socialmediadataclean_weekly_6mo.csv \
  --prices data/raw/prices_6mo.csv \
  --out    data/processed/socialmedia_price_panel_6mo.csv \
  --vol-window 8

# 6) Analysis
python src/run_analysis.py \
  --in  data/processed/socialmedia_price_panel_6mo.csv \
  --out-dir results

# 7) Visualization
python src/visualize_results.py \
  --in  data/processed/socialmedia_price_panel_6mo.csv \
  --out-dir results
```

---