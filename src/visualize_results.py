"""
visualize_results.py
--------------------
Generate visualizations for the social media and price panel data analysis.
Outputs include:
- Figure 1: Correlation Analysis Heatmaps
- Figure 2: Stock Price vs. Social Sentiment Trends
- Figure 3: Sentiment Score vs. Next Week Returns
- Figure 4: Sentiment Distribution Analysis (Overall & per Ticker)
- Figure 5: Online Discussion Volume vs. Abnormal Trading Volume
- Figure 6: Discussion Volume vs. Trading Volume by Ticker

Usage
-----
# Run from project root:
python src/visualize_results.py \
  --in data/processed/socialmedia_price_panel_6mo.csv \
  --out-dir results
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """Load CSV, convert dates, sort, and perform initial cleaning."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    df['week_start'] = pd.to_datetime(df['week_start'])
    df = df.sort_values('week_start')
    # Clean the data: Remove the rows where the key fields are empty
    original_len = len(df)
    df_clean = df.dropna(subset=['mentions_week', 'sentiment_mean_w', 'weekly_volume', 'next_week_return', 'abnormal_volume'])
    print(f"Data Loaded. Rows: {original_len} -> Cleaned: {len(df_clean)}")
    return df_clean


def perform_correlation_heatmap(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Figure 1: Generate Correlation Heatmaps."""
    print("Generating Figure 1: Correlation Heatmaps...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Sentiment vs Returns Correlation
    sentiment_corr_cols = ['sentiment_mean_w', 'sentiment_score_weighted_w', 'median_compound_week', 'next_week_return']
    sentiment_corr_matrix = df_clean[sentiment_corr_cols].corr().round(3)
    sns.heatmap(sentiment_corr_matrix, annot=True, cmap='coolwarm', center=0, 
                fmt=".3f", linewidths=0.5, ax=ax1, cbar_kws={'shrink': 0.8})
    ax1.set_title('(a) Sentiment vs Returns Correlation', fontsize=14, fontweight='bold', pad=20)

    # Discussion Volume vs Trading Volume Correlation
    volume_corr_cols = ['mentions_week', 'days_covered', 'weekly_volume', 'abnormal_volume']
    volume_corr_matrix = df_clean[volume_corr_cols].corr().round(3)
    sns.heatmap(volume_corr_matrix, annot=True, cmap='viridis', center=0, 
                fmt=".3f", linewidths=0.5, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('(b) Discussion Volume vs Trading Volume Correlation', fontsize=14, fontweight='bold', pad=20)

    # Set the general title
    fig.suptitle('Figure 1: Correlation Analysis Heatmaps', fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    out_path = Path(out_dir) / 'fig1_correlation_heatmaps.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def perform_price_vs_sentiment(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Figure 2: Stock Price vs. Social Sentiment Trends."""
    print("Generating Figure 2: Price vs Sentiment Trends...")
    
    # Get all the unique Tickers
    unique_tickers = df_clean['ticker'].unique()
    # Create a 5x2 subgraph grid
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 25))
    fig.suptitle('Figure 2: Stock Price vs. Social Sentiment Trends', fontsize=20, y=0.98)

    # Draw charts for each ticker
    for idx, ticker in enumerate(unique_tickers):
        if idx >= rows * cols: break
        row = idx // cols
        col = idx % cols
        ticker_data = df_clean[df_clean['ticker'] == ticker].sort_values('week_start')
        ax1 = axes[row, col]
        # the stock price on the left
        color1 = '#1f77b4'
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Stock Price ($)', color=color1, fontsize=10)
        ax1.plot(ticker_data['week_start'], ticker_data['weekly_close'], color=color1, linewidth=2.5, label='Price')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        # the sentiment score on the right
        ax2 = ax1.twinx()
        color2 = '#ff7f0e'
        ax2.set_ylabel('Sentiment Score', color=color2, fontsize=10)
        ax2.plot(ticker_data['week_start'], ticker_data['sentiment_score_weighted_w'], color=color2, linestyle='--', marker='o', markersize=3, linewidth=1.5, label='Sentiment')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.axhline(0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
        # Set the Title
        ax1.set_title(f'{ticker}', fontsize=14, fontweight='bold', pad=10)
        # Add Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    out_path = Path(out_dir) / 'fig2_price_vs_sentiment.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def perform_sentiment_vs_returns(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Figure 3: Sentiment Score vs. Next Week Returns."""
    print("Generating Figure 3: Sentiment vs Returns Comparison...")

    # Define Emotion categories function
    def get_sentiment_label(score):
        if score > 0.05: return 'Positive'
        if score < -0.05: return 'Negative'
        return 'Neutral'

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Figure 3: Comparison of Next Week Returns Across Different Sentiment Measurement Methods', fontsize=16, fontweight='bold', y=0.95)
    sentiment_metrics = ['sentiment_mean_w', 'sentiment_score_weighted_w', 'median_compound_week']
    titles = ['Mean Sentiment Score', 'Weighted Sentiment Score', 'Median Compound Score']

    for idx, (metric, title) in enumerate(zip(sentiment_metrics, titles)):
        temp_df = df_clean.copy()
        temp_df['sentiment_label'] = temp_df[metric].apply(get_sentiment_label)   
        # Create violin plot + boxplot combination
        sns.violinplot(x='sentiment_label', y='next_week_return', data=temp_df, order=['Negative', 'Neutral', 'Positive'], palette='pastel', ax=axes[idx])
        sns.boxplot(x='sentiment_label', y='next_week_return', data=temp_df, order=['Negative', 'Neutral', 'Positive'], width=0.3, boxprops={'alpha': 0.7}, ax=axes[idx])
        # Add zero line
        axes[idx].axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='Zero Return')

        # Calculate and display mean returns
        means = temp_df.groupby('sentiment_label')['next_week_return'].mean()
        for i, category in enumerate(['Negative', 'Neutral', 'Positive']):
            if category in means:
                mean_val = means[category]
                color = 'green' if mean_val > 0 else 'red' if mean_val < 0 else 'black'
                axes[idx].text(i, axes[idx].get_ylim()[0] + (axes[idx].get_ylim()[1] - axes[idx].get_ylim()[0]) * 0.05, 
                            f'Mean: {mean_val:.4f}', ha='center', va='bottom', 
                            fontsize=9, fontweight='bold', color=color,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.9))
        
        # Set labels and title
        axes[idx].set_title(f'({chr(97+idx)}) {title}', fontsize=12, fontweight='bold', pad=15)
        axes[idx].set_ylabel('Next Week Return' if idx == 0 else '', fontsize=11)
        axes[idx].set_xlabel('Sentiment Category', fontsize=11)
        # Add correlation info
        corr = temp_df[metric].corr(temp_df['next_week_return'])
        axes[idx].text(0.5, 0.95, f'Correlation: {corr:.3f}', transform=axes[idx].transAxes, ha='center', va='top', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    out_path = Path(out_dir) / 'fig3_enhanced_sentiment_comparison.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def perform_sentiment_distribution(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Figure 4: Sentiment Distribution Analysis (Overall and by Ticker)."""
    print("Generating Figure 4: Sentiment Distributions...")

    # --- Figure 4A: Overall Mood histogram ---
    plt.figure(figsize=(12, 6))
    sns.histplot(x=df_clean['sentiment_score_weighted_w'], bins=25, kde=True, color='steelblue', alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=2, alpha=0.8, label='Neutral Line')
    plt.axvline(df_clean['sentiment_score_weighted_w'].mean(), color='red', linestyle='-', linewidth=2, alpha=0.8, label=f'Mean: {df_clean["sentiment_score_weighted_w"].mean():.3f}')

    mean_val = df_clean['sentiment_score_weighted_w'].mean()
    std_val = df_clean['sentiment_score_weighted_w'].std()
    positive_pct = (df_clean['sentiment_score_weighted_w'] > 0).mean() * 100

    plt.text(0.02, 0.95, f'Mean: {mean_val:.3f}\nStd: {std_val:.3f}\nPositive: {positive_pct:.1f}%', 
            transform=plt.gca().transAxes, fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.title('Figure 4A: Overall Sentiment Score Distribution (All Tickers Combined)', fontsize=14, fontweight='bold')
    plt.xlabel('Weighted Sentiment Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    out_path_a = Path(out_dir) / 'fig4A_overall_sentiment_distribution.png'
    plt.savefig(out_path_a, dpi=300, bbox_inches='tight')
    plt.close('all')

    # --- Figure 4B: Ticker's Emotion Histogram (5×2) ---
    unique_tickers = df_clean['ticker'].unique()
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 20))

    for idx, ticker in enumerate(unique_tickers):
        if idx >= rows * cols: break
        row = idx // cols
        col = idx % cols
        
        ticker_data = df_clean[df_clean['ticker'] == ticker]
        ax = axes[row, col]
        
        # Draw the histogram of a single ticker
        sns.histplot(ticker_data['sentiment_score_weighted_w'], bins=15, kde=True, color='purple', alpha=0.7, ax=ax)
        # Add reference lines and statistical information
        ax.axvline(0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(ticker_data['sentiment_score_weighted_w'].mean(), color='red', linestyle='-', linewidth=1.5, alpha=0.7)
        ticker_mean = ticker_data['sentiment_score_weighted_w'].mean()
        ticker_std = ticker_data['sentiment_score_weighted_w'].std()
        ticker_positive = (ticker_data['sentiment_score_weighted_w'] > 0).mean() * 100
        n_obs = len(ticker_data)
        
        stats_text = f'Mean: {ticker_mean:.3f}\nStd: {ticker_std:.3f}\nPos: {ticker_positive:.1f}%\nn={n_obs}'
        ax.text(0.65, 0.85, stats_text, transform=ax.transAxes, fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_title(f'{ticker}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Figure 4B: Sentiment Score Distribution by Individual Ticker', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    out_path_b = Path(out_dir) / 'fig4B_ticker_sentiment_distributions.png'
    plt.savefig(out_path_b, dpi=300, bbox_inches='tight')
    plt.close('all')


def perform_discussion_vs_trading_volume(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Figure 5: Online Discussion Volume vs. Abnormal Trading Volume."""
    print("Generating Figure 5: Discussion vs Abnormal Volume...")

    # Get all the unique Tickers
    unique_tickers = df_clean['ticker'].unique()
    # Create a 5x2 subgraph grid
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 25))
    fig.suptitle('Figure 5: Online Discussion Volume vs Abnormal Trading Volume', fontsize=20, y=0.98)

    # Draw charts for each ticker
    for idx, ticker in enumerate(unique_tickers):
        if idx >= rows * cols: break
        row = idx // cols
        col = idx % cols
        ticker_data = df_clean[df_clean['ticker'] == ticker].sort_values('week_start')
        ax1 = axes[row, col]
        
        # Online discussion volume on the left (mentions_week)
        color1 = '#2E86AB'
        ax1.set_xlabel('Date', fontsize=10)
        ax1.set_ylabel('Online Mentions', color=color1, fontsize=10)
        ax1.plot(ticker_data['week_start'], ticker_data['mentions_week'], color=color1, linewidth=2.5, label='Online Mentions')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.tick_params(axis='x', rotation=45, labelsize=8)
        
        # Abnormal trading volume on the right
        ax2 = ax1.twinx()
        color2 = '#A23B72'
        ax2.set_ylabel('Abnormal Trading Volume', color=color2, fontsize=10)
        ax2.plot(ticker_data['week_start'], ticker_data['abnormal_volume'], color=color2, linestyle='--', marker='s', markersize=3, linewidth=1.5, label='Abnormal Volume')
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # Add correlation coefficient to title
        correlation = ticker_data['mentions_week'].corr(ticker_data['abnormal_volume'])
        
        # Set the Title with correlation info
        ax1.set_title(f'{ticker} (r = {correlation:.3f})', fontsize=14, fontweight='bold', pad=10)
        
        # Add Legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    out_path = Path(out_dir) / 'fig5_discussion_vs_trading_volume.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def perform_volume_scatter(df_clean: pd.DataFrame, out_dir: str) -> None:
    """Figure 6: Scatter Plot Volume vs Mentions by Ticker."""
    print("Generating Figure 6: Discussion Volume vs Trading Volume Scatter...")

    unique_tickers = df_clean['ticker'].unique()
    rows, cols = 5, 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 20))
    fig.suptitle('Figure 6: Discussion Volume vs. Trading Volume by Ticker', fontsize=16, fontweight='bold', y=0.98)

    for idx, ticker in enumerate(unique_tickers):
        if idx >= rows * cols: break
        row = idx // cols
        col = idx % cols
        
        ticker_data = df_clean[df_clean['ticker'] == ticker]
        ax = axes[row, col]
        
        # Draw scatter plots and regression lines
        sns.regplot(x='mentions_week', y='weekly_volume', data=ticker_data, scatter_kws={'alpha':0.6, 's':40}, line_kws={'color':'red', 'linewidth':2, 'label': 'Trend Line'}, ax=ax)

        # Set titles and tags
        correlation = ticker_data['mentions_week'].corr(ticker_data['weekly_volume'])
        ax.set_title(f'{ticker} (r = {correlation:.3f})', fontsize=12, fontweight='bold')
        ax.set_xlabel('Weekly Mentions')
        ax.set_ylabel('Weekly Trading Volume')
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.6)
        # Add sample size information
        n_obs = len(ticker_data)
        ax.text(0.05, 0.95, f'n = {n_obs}', transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(top=0.96)
    out_path = Path(out_dir) / 'fig6_volume_vs_mentions_by_ticker.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description="Visualize social media and price panel data analysis.")
    # UPDATED: Added --in as an alias and set dest='input' so code logic doesn't need to change
    ap.add_argument("--in", "--input", dest="input", default="data/processed/socialmedia_price_panel_6mo.csv", help="Path to input CSV.")
    ap.add_argument("--out-dir", default="results", help="Directory to save output images.")
    return ap.parse_args()


def main() -> None:
    """Entry point: load data, set styles, run visualization steps."""
    args = parse_args()
    
    # Global Style Settings
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (12, 7)
    
    # Ensure output directory exists
    ensure_dir(args.out_dir)

    # 1. Load Data
    df_clean = load_and_clean_data(args.input)

    # 2. Generate Visualizations
    perform_correlation_heatmap(df_clean, args.out_dir)
    perform_price_vs_sentiment(df_clean, args.out_dir)
    perform_sentiment_vs_returns(df_clean, args.out_dir)
    perform_sentiment_distribution(df_clean, args.out_dir)
    perform_discussion_vs_trading_volume(df_clean, args.out_dir)
    perform_volume_scatter(df_clean, args.out_dir)
    
    print(f"\nAll visualizations saved to: {args.out_dir}")


if __name__ == "__main__":
    main()