import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA_10, MA_20, MA_50"""
    df['MA_10'] = df['close'].rolling(10).mean()
    df['MA_20'] = df['close'].rolling(20).mean()
    df['MA_50'] = df['close'].rolling(50).mean()
    logger.info("Computed moviging averages")
    return df

def compute_daily_return_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """Compute daily return and 20-day rolling volatility"""
    df['Daily_Return'] = df['close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(20).std()
    logger.info("Computed daily return and volatility")
    return df

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Compute Relative Strength Index"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = -delta.where(delta < 0, 0).rolling(period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    logger.info("Computed RSI")
    return df

def compute_macd(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MACD and Signal line"""
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    logger.info("Computed MACD")
    return df


def compute_bollinger_bands(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Bollinger Bands"""
    df['BB_Middle'] = df['close'].rolling(20).mean()
    df['BB_Upper'] = df['BB_Middle'] + 2 * df['close'].rolling(20).std()
    df['BB_Lower'] = df['BB_Middle'] - 2 * df['close'].rolling(20).std()
    logger.info("Computed Bollinger Bands")
    return df


def merge_sentiment(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge sentiment features into price DataFrame"""
    price_df['date'] = pd.to_datetime(price_df['date'])
    sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])

    df = price_df.merge(sentiment_df, on='date', how='left')

    sentiment_cols = [
        'avg_sentiment', 'weighted_sentiment',
        'total_engagement', 'tweet_count',
        'positive_count', 'negative_count'
    ]

    # If sentiment DB is empty, fill all with neutral 0 values
    for col in sentiment_cols:
        if col not in df.columns or df[col].isna().all():
            df[col] = 0.0

    # Forward fill then backfill then fill any remaining with 0
    df[sentiment_cols] = df[sentiment_cols].ffill().bfill().fillna(0.0)

    logger.info(f"Merged sentiment into price data — shape: {df.shape}")
    return df


def run_feature_engineering(
    price_df: pd.DataFrame,
    sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline"""
    logger.info("Starting feature engineering pipeline...")

    df = price_df.copy()
    df = df.sort_values('date').reset_index(drop=True)

    # Compute all technical indicators
    df = compute_moving_averages(df)
    df = compute_daily_return_volatility(df)
    df = compute_rsi(df)
    df = compute_macd(df)
    df = compute_bollinger_bands(df)

    # Merge sentiment
    df = merge_sentiment(df, sentiment_df)

    # Drop NaN rows from rolling calculations
    df = df.dropna().reset_index(drop=True)

    logger.info(f"Feature engineering complete — shape: {df.shape}")
    logger.info(f"Columns: {df.columns.tolist()}")
    return df


def get_feature_columns() -> list:
    """Returns the exact feature list used during model training"""
    return [
        'close', 'MA_10', 'MA_20', 'MA_50',
        'Daily_Return', 'Volatility', 'RSI',
        'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower',
        'avg_sentiment', 'weighted_sentiment', 'total_engagement'
    ]

if __name__ == "__main__":
    #Quick test
    import yfinace as yf
    tsla = yf.download('TSLA', start='2024-01-01')
    if isinstance(tsla.columns, pd.MultiIndex):
        tsla.columns = tsla.columns.get_level_values(0)
    tsla = tsla.reset_index()
    tsla.columns = [c.lower() for c in tsla.columns]

    #Dummy sentiment
    sentiment = pd.DataFrame({
        'date': tsla['date'],
        'avg_sentiment': 0.2,
        'weighted_sentiment': 100.0,
        'total_engagement': 500,
        'tweet_count': 3,
        'positive_count': 2,
        'negative_count': 1
    })

    result = run_feature_engineering(tsla, sentiment)
    print(result.tail())
    print("Features:", get_feature_columns())