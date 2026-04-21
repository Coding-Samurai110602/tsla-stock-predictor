import finnhub
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_finnhub_client():
    """Initialize Finnhub client"""
    return finnhub.Client(api_key=os.getenv('FINNHUB_API_KEY'))


def get_db_engine():
    url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)


def fetch_latest_tsla_price():
    """Fetch today's TSLA price from Finnhub"""
    try:
        logger.info("Fetching latest TSLA price from Finnhub...")
        client = get_finnhub_client()
        quote = client.quote('TSLA')

        if not quote or quote.get('c') == 0:
            logger.warning("No data returned from Finnhub")
            return None

        # Finnhub quote fields:
        # c = current price, h = high, l = low, o = open, v = volume
        from datetime import date as date_type
        data = {
            'date': date_type.today(),
            'open': float(quote['o']),
            'high': float(quote['h']),
            'low': float(quote['l']),
            'close': float(quote['c']),
            'volume': int(quote.get('v', 0))
        }
        logger.info(f"Fetched TSLA price for {data['date']}: ${data['close']:.2f}")
        return data

    except Exception as e:
        logger.error(f"Error fetching TSLA price from Finnhub: {e}")
        return None


def fetch_historical_tsla(start_date='2010-06-29'):
    """Load TSLA history from local CSV — no network call needed"""
    try:
        logger.info("Loading TSLA history from CSV...")
        df = pd.read_csv('/app/df_tsla.csv')

        # Keep only price columns needed for DB
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date']).dt.date
        df = df[df['date'] >= pd.to_datetime(start_date).date()]
        df = df.dropna()
        df = df.sort_values('date').reset_index(drop=True)

        logger.info(f"Loaded {len(df)} rows from CSV")
        return df

    except Exception as e:
        logger.error(f"Error loading CSV: {e}")
        return None


def store_price_data(data: dict):
    """Store single day price data to PostgreSQL"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO price_history (date, open, high, low, close, volume)
                VALUES (:date, :open, :high, :low, :close, :volume)
                ON CONFLICT (date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
            """), data)
            conn.commit()
        logger.info(f"Stored price data for {data['date']}")
        return True

    except Exception as e:
        logger.error(f"Error storing price data: {e}")
        return False


def store_historical_prices(df: pd.DataFrame):
    """Bulk store historical price data"""
    try:
        engine = get_db_engine()
        records = df[['date', 'open', 'high', 'low', 'close', 'volume']].to_dict('records')

        with engine.connect() as conn:
            for record in records:
                conn.execute(text("""
                    INSERT INTO price_history (date, open, high, low, close, volume)
                    VALUES (:date, :open, :high, :low, :close, :volume)
                    ON CONFLICT (date) DO NOTHING
                """), record)
            conn.commit()

        logger.info(f"Stored {len(records)} historical price records")
        return True

    except Exception as e:
        logger.error(f"Error storing historical prices: {e}")
        return False


def get_last_n_days_prices(n: int = 100):
    """Fetch last N days of price data from DB"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, open, high, low, close, volume
                FROM price_history
                ORDER BY date DESC
                LIMIT :n
            """), {'n': n})
            rows = result.fetchall()

        df = pd.DataFrame(rows, columns=['date', 'open', 'high', 'low', 'close', 'volume'])
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"Retrieved {len(df)} price records from DB")
        return df

    except Exception as e:
        logger.error(f"Error fetching prices from DB: {e}")
        return None


if __name__ == "__main__":
    data = fetch_latest_tsla_price()
    if data:
        print("Latest TSLA data:", data)
        store_price_data(data)