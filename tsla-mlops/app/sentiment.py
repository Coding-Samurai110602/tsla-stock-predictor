import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from sqlalchemy import create_engine, text
import os
import logging
from datetime import datetime, date
from dotenv import load_dotenv
import requests

def get_db_engine():
    url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#TSLA relevant keywords
TSLA_KEYWORDS = [
    'tesla', 'tsla', 'model s', 'model 3', 'model x', 'model y',
    'cybertruck', 'electric', 'ev', 'autopilot', 'fsd', 'giga',
    'production', 'delivery', 'stock', 'short', 'shareholders',
    'battery', 'supercharger', 'roadster', 'semi', 'powerwall'
]

def load_finbert():
    """Load FinBERT model and pipeline"""
    try:
        logger.info("Loading FinBERT model...")
        tokenizer = BertTokenizer.from_pretrained('ProsusAI/finbert')
        model = BertForSequenceClassification.from_pretrained('ProsusAI/finbert')

        finbert_pipeline = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=-1,   #CPU
            batch_size=32,
            truncation=True,
            max_length=512
        )
        logger.info("FinBERT loaded successfully")
        return finbert_pipeline

    except Exception as e:
        logger.error(f"Error loading FinBERT: {e}")
        return None

def filter_tsla_tweets(tweets: list) -> list:
    """Fileter tweets to only TSLA relevant ones"""
    pattern = '|'.join(TSLA_KEYWORDS)
    filtered = [
        t for t in tweets
        if any(kw in str(t.get('text', '')).lower() for kw in TSLA_KEYWORDS)
    ]
    logger.info(f"Filtered {len(filtered)} TSLA-relevant tweets from {len(tweets)}")
    return filtered

def score_tweets(tweets: list, finbert_pipeline) -> list:
    """Run FinBERT sentiment scoring on tweets"""
    if not tweets:
        return []

    texts = [str(t.get('text', '')) for t in tweets]

    try:
        results = finbert_pipeline(texts)
        scored = []
        for tweet, result in zip(tweets, results):
            label = result['label']
            score = result['score']

            #Convert to compound score like VADER
            if label == "positive":
                compound = score
            elif label == "negative":
                compound=-score
            else:
                compound=0.0

            scored.append({
                **tweet,
                'sentiment_score': compound,
                'sentiment_label': label
            })
        return scored
    
    except Exception as e:
        logger.error(f"Error scoring tweets: {e}")
        return []


def aggregate_daily_sentiment(scored_tweets: list, target_date: date) -> dict:
    """Aggregate scored tweets into daily sentiment metrics"""
    if not scored_tweets:
        return {
            'date': target_date,
            'avg_sentiment': 0.0,
            'weighted_sentiment': 0.0,
            'total_engagement': 0,
            'tweet_count': 0,
            'positive_count': 0,
            'negative_count': 0
        }

    df = pd.DataFrame(scored_tweets)

    #Filling missing engagement with 1
    df['like_count'] = df.get('like_count', pd.Series([1] * len(df))).fillna(1)
    df['retweet_count'] = df.get('retweet_count', pd.Series([1] * len(df))).fillna(1)

    #Weighted sentiment
    df['engagement'] = df['like_count'] + df['retweet_count'] * 2
    df['weighted_sentiment'] = df['sentiment_score'] * df['engagement']

    return {
        'date': target_date,
        'avg_sentiment': float(df['sentiment_score'].mean()),
        'weighted_sentiment': float(df['weighted_sentiment'].sum()),
        'total_engagement': int(df['engagement'].sum()),
        'tweet_count': len(df),
        'positive_count': int((df['sentiment_label'] == 'positive').sum()),
        'negative_count': int((df['sentiment_label'] == 'negative').sum())
    }

def store_sentiment(sentiment: dict):
    """Store daily sentiment to PostgreSQL"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO sentiment_history (
                    date, avg_sentiment, weighted_sentiment,
                    total_engagement, tweet_count,
                    positive_count, negative_count
                )
                VALUES (
                    :date, :avg_sentiment, :weighted_sentiment,
                    :total_engagement, :tweet_count,
                    :positive_count, :negative_count
                )
                ON CONFLICT (date) DO UPDATE SET
                    avg_sentiment = EXCLUDED.avg_sentiment,
                    weighted_sentiment = EXCLUDED.weighted_sentiment,
                    total_engagement = EXCLUDED.total_engagement,
                    tweet_count = EXCLUDED.tweet_count,
                    positive_count = EXCLUDED.positive_count,
                    negative_count = EXCLUDED.negative_count
            """), sentiment)
            conn.commit()
        logger.info(f"Stored sentiment for {sentiment['date']}")
        return True
        
    except Exception as e:
        logger.error(f"Error storing sentiment: {e}")
        return False


def get_latest_sentiment():
    """Fetch most recent sentiment from DB"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT *
                FROM sentiment_history
                ORDER BY date DESC
                LIMIT 1
            """))
            row = result.fetchone()
            
        if row:
            return dict(row._mapping)
        return None
        
    except Exception as e:
        logger.error(f"Error fetching sentiment: {e}")
        return None
        
        
def get_last_n_days_sentiment(n: int = 100):
    """Fetch last N days of sentiment from DB"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, avg_sentiment, weighted_sentiment,
                       total_engagement, tweet_count,
                       positive_count, negative_count
                FROM sentiment_history
                ORDER BY date DESC
                LIMIT :n
            """), {'n': n})
            rows = result.fetchall()

        df = pd.DataFrame(
            rows,
            columns=[
                'date', 'avg_sentiment', 'weighted_sentiment',
                'total_engagement', 'tweet_count',
                'positive_count', 'negative_count'
            ]
        )
        df = df.sort_values('date').reset_index(drop=True)
        logger.info(f"Retrived {len(df)} sentiment records from DB")
        return df

    except Exception as e:
        logger.error(f"Error fetching sentiment history: {e}")
        return None


if __name__ == "__main__":
    #Testing with sample tweets
    sample_tweets = [
        {'text': 'Tesla deliveries hit record high this quarter!', 'like_count': 5000, 'retweet_count': 1200},
        {'text': 'FSD update is incredible, full autonomy soon', 'like_count': 8000, 'retweet_count': 2000},
        {'text': 'Tesla production numbers missed expectations', 'like_count': 3000, 'retweet_count': 800},
    ]

    finbert = load_finbert()
    if finbert:
        filtered = filter_tsla_tweets(sample_tweets)
        scored = score_tweets(filtered, finbert)
        sentiment = aggregate_daily_sentiment(scored, date.today())
        print("Daily sentiment:", sentiment)