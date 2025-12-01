"""
run_analysis.py
---------------
Analyze the processed social media and price panel data to answer project questions.
Outputs include:
- Descriptive statistics (Console & Excel)
- Correlation matrices (Console & Excel)
- Hypothesis testing results (T-tests, Console & Txt)
- Simple trading strategy simulation (Console & Txt)

Usage
-----
# Run from project root:
python src/run_analysis.py \
  --in data/processed/socialmedia_price_panel_6mo.csv \
  --out-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load CSV and perform initial cleaning."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    # Convert to date format
    df['week_start'] = pd.to_datetime(df['week_start'])
    # Clean the data: Remove the rows where the key fields are empty
    original_len = len(df)
    df_clean = df.dropna(subset=['mentions_week', 'sentiment_mean_w', 'weekly_volume', 'next_week_return', 'abnormal_volume'])
    print(f"Data Loaded. Rows: {original_len} -> Cleaned: {len(df_clean)}")
    return df_clean


def perform_descriptive_stats(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Calculate and save descriptive statistics."""
    print("\n[1] Descriptive Statistics")
    stats_cols = ['mentions_week', 'days_covered', 'sentiment_mean_w', 'sentiment_score_weighted_w', 'median_compound_week', 'weekly_close', 'weekly_volume', 'abnormal_volume', 'next_week_return']
    desc_stats = df_clean[stats_cols].describe().round(4)
    print(desc_stats)

    # Check the average performance of each Ticker
    print("\nAverage Performance by Ticker")
    ticker_stats = df_clean.groupby('ticker')[stats_cols].mean()
    sorted_ticker_stats = ticker_stats.sort_values('mentions_week', ascending=False).round(4)
    print(sorted_ticker_stats)
    
    out_path = Path(out_dir) / 'descriptive_statistics.xlsx'
    with pd.ExcelWriter(out_path) as writer:
        desc_stats.to_excel(writer, sheet_name='Overall_Descriptive_Stats')
        sorted_ticker_stats.to_excel(writer, sheet_name='Ticker_Average_Performance')


def perform_correlation_analysis(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Calculate and save correlation matrices."""
    print("\n[2.1] Sentiment vs Returns Correlation")
    sentiment_corr_cols = ['sentiment_mean_w', 'sentiment_score_weighted_w', 'median_compound_week', 'next_week_return']
    sentiment_corr_matrix = df_clean[sentiment_corr_cols].corr().round(4)
    print(sentiment_corr_matrix)

    print("\n[2.2] Discussion Volume vs Trading Volume Correlation")
    volume_corr_cols = ['mentions_week', 'days_covered', 'weekly_volume', 'abnormal_volume']
    volume_corr_matrix = df_clean[volume_corr_cols].corr().round(4)
    print(volume_corr_matrix)

    # Save the two correlation matrices to different worksheets in the same Excel file
    out_path = Path(out_dir) / 'correlation_matrices.xlsx'
    with pd.ExcelWriter(out_path) as writer:
        sentiment_corr_matrix.to_excel(writer, sheet_name='Sentiment_Returns_Corr')
        volume_corr_matrix.to_excel(writer, sheet_name='Discussion_Volume_Corr')

    # Extract the key correlation coefficients
    sentiment_return_corr = sentiment_corr_matrix.loc['sentiment_score_weighted_w', 'next_week_return']
    mentions_abnormal_vol_corr = volume_corr_matrix.loc['mentions_week', 'abnormal_volume']
    print(f"\nKey Correlation Coefficients:")
    print(f"Sentiment Score vs Next Week Return: {sentiment_return_corr:.4f}")
    print(f"Mentions Week vs Abnormal Volume: {mentions_abnormal_vol_corr:.4f}")


def perform_hypothesis_testing(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Perform T-tests on sentiment and returns and save to txt."""
    # Objective: To verify whether the yield rate during the "high mood" week is significantly higher than that during the "low mood" week
    out_path = Path(out_dir) / 'hypothesis_testing.txt'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        title = "[3] Hypothesis Testing (T-Test)"
        print("\n" + title)
        f.write(title + "\n")
        
        high_sentiment = df_clean[df_clean['sentiment_score_weighted_w'] > 0.5]['next_week_return']
        low_sentiment = df_clean[df_clean['sentiment_score_weighted_w'] < -0.5]['next_week_return']
        t_stat, p_val = stats.ttest_ind(high_sentiment, low_sentiment, equal_var=False)
        
        line1 = f"High Sentiment Weeks (N={len(high_sentiment)}) vs Low Sentiment Weeks (N={len(low_sentiment)})"
        line2 = f"T-statistic: {t_stat:.4f}, P-value: {p_val:.4f}"
        
        print(line1)
        print(line2)
        f.write(line1 + "\n")
        f.write(line2 + "\n")
    
        # Calculate the returns for each stock
        ticker_results = []
        for ticker in df_clean['ticker'].unique():
            ticker_data = df_clean[df_clean['ticker'] == ticker]
            if len(ticker_data) > 2:
                high_ret = ticker_data[ticker_data['sentiment_score_weighted_w'] > 0.5]['next_week_return'].mean()
                low_ret = ticker_data[ticker_data['sentiment_score_weighted_w'] < -0.5]['next_week_return'].mean()
                ticker_results.append(high_ret - low_ret)
        
        # Paired t-test: To test whether the difference is significant and not zero
        t_stat_paired, p_val_paired = stats.ttest_1samp([x for x in ticker_results if not np.isnan(x)], 0)
        
        line3 = f"\nPaired T-test (controlling for ticker): T-statistic: {t_stat_paired:.4f}, P-value: {p_val_paired:.4f}"
        print(line3)
        f.write(line3 + "\n")


def perform_strategy_simulation(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Simulate a simple sentiment-based trading strategy and save to txt."""
    # Objective: To calculate how much profit we would make if we only held stocks when our emotions were positive？
    out_path = Path(out_dir) / 'strategy_simulation.txt'
    
    with open(out_path, 'w', encoding='utf-8') as f:
        title = "[4] Sentiment Strategy Simulation"
        print("\n" + title)
        f.write(title + "\n")
        
        # Re-define high_sentiment locally as it's needed for the mean calculation
        high_sentiment = df_clean[df_clean['sentiment_score_weighted_w'] > 0.5]['next_week_return']
        
        avg_market_return = df_clean['next_week_return'].mean()
        avg_strategy_return = high_sentiment.mean()
        
        line1 = f"Average Weekly Return (All Data): {avg_market_return*100:.2f}%"
        line2 = f"Average Weekly Return (High Sentiment Weeks Only): {avg_strategy_return*100:.2f}%"
        
        print(line1)
        print(line2)
        f.write(line1 + "\n")
        f.write(line2 + "\n")
        
        improvement = avg_strategy_return - avg_market_return
        line3 = f">> Strategy Improvement: {improvement*100:.2f} percentage points"
        print(line3)
        f.write(line3 + "\n")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description="Analyze social media and price panel data.")
    # UPDATED: Added --in as an alias for --input to match README instructions
    ap.add_argument("--in", "--input", dest="input", default="data/processed/socialmedia_price_panel_6mo.csv", help="Path to input CSV.")
    ap.add_argument("--out-dir", default="results", help="Directory to save output Excel files.")
    return ap.parse_args()


def main() -> None:
    """Entry point: load data, run analysis steps."""
    args = parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.out_dir)

    # 1. Load Data
    df_clean = load_and_clean_data(args.input)

    # 2. Descriptive Statistics
    perform_descriptive_stats(df_clean, args.out_dir)

    # 3. Correlation Analysis
    perform_correlation_analysis(df_clean, args.out_dir)

    # 4. Hypothesis Testing
    perform_hypothesis_testing(df_clean, args.out_dir)

    # 5. Strategy Simulation
    perform_strategy_simulation(df_clean, args.out_dir)


if __name__ == "__main__":
    main()