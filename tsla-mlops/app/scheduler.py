import logging
import os
from datetime import datetime, date, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from dotenv import load_dotenv

from data_ingestion import (
    fetch_latest_tsla_price,
    store_price_data,
    get_last_n_days_prices
)
from sentiment import (
    load_finbert,
    get_last_n_days_sentiment
)
from feature_engineering import run_feature_engineering
from predict import (
    load_artifacts,
    predict_next_day,
    store_prediction,
    cache_prediction,
    update_actual_price,
    log_to_mlflow
)

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load heavy models once at startup
finbert_pipeline = None
lstm_model = None
scaler = None


def initialize_models():
    """Load all models into memory at startup"""
    global finbert_pipeline, lstm_model, scaler
    logger.info("Initializing models...")
    finbert_pipeline = load_finbert()
    lstm_model, scaler = load_artifacts()
    logger.info("All models loaded and ready")

def run_daily_pipeline():
    """
    Full end of day pipeline which runs at 4:05 PM EST every market day
    1. Fetch today's price
    2. Get latest sentiment
    3. Feature engineering
    4. Predict tomorrow's price
    5. Store + cache results
    6. Update yesterday's actual price
    """
    logger.info(f"Starting daily pipeline at {datetime.now()}")

    try:
        # Step 1 — Fetch today's price from Finnhub
        price_data = fetch_latest_tsla_price()
        if not price_data:
            logger.error("Failed to fetch price data — aborting pipeline")
            return
        store_price_data(price_data)
        current_price = float(price_data['close'])

        # Step 2 — Update yesterday's actual price
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        update_actual_price(yesterday, current_price)

        # Step 3 — Get last 100 days price + sentiment from DB
        price_df = get_last_n_days_prices(200)
        sentiment_df = get_last_n_days_sentiment(100)

        if price_df is None or sentiment_df is None:
            logger.error("Failed to fetch historical data — aborting pipeline")
            return

        # Step 4 — Feature engineering
        df = run_feature_engineering(price_df, sentiment_df)
        if df is None or len(df) < 60:
            logger.error("Not enough data for prediction — aborting pipeline")
            return

        # Step 5 — Predict tomorrow's price
        prediction = predict_next_day(df, lstm_model, scaler, current_price)

        if not prediction:
            logger.error("Prediction failed — aborting pipeline")
            return

        # Step 6 — Store and cache
        store_prediction(prediction)
        cache_prediction(prediction)
        log_to_mlflow(prediction)

        logger.info(
            f"Pipeline complete: Tomorrow's prediction: "
            f"${prediction['predicted_price']} "
            f"({prediction['predicted_dir']}) "
            f"with {prediction['confidence']}% confidence"
        )

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")

def run_backfill():
    """
    Run once on startup to fill DB with historical data
    Only runs if price_history table is empty
    """
    from data_ingestion import fetch_historical_tsla, store_historical_prices
    from sentiment import get_last_n_days_sentiment
    from sqlalchemy import create_engine, text

    logger.info("Checking if backfill is needed...")

    try:
        engine = create_engine(
            f"postgresql://{os.getenv('POSTGRES_USER')}:"
            f"{os.getenv('POSTGRES_PASSWORD')}@"
            f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
            f"{os.getenv('POSTGRES_PORT', '5432')}/"
            f"{os.getenv('POSTGRES_DB')}"
        )

        with engine.connect() as conn:
            result = conn.execute(text("SELECT COUNT(*) FROM price_history"))
            count = result.scalar()

        if count == 0:
            logger.info("DB is empty - running historical backfill...")
            price_df = fetch_historical_tsla()
            if price_df is not None:
                store_historical_prices(price_df)
                logger.info("Historical price backfill complete")
        else:
            logger.info(f"DB already has {count} price records = skipping backfill")

    except Exception as e:
        logger.error(f"Backfill error: {e}")


#Schedule instance
scheduler = BackgroundScheduler(timezone="America/New_York")

def run_weekend_sentiment_collection():
    """
    Collect and store Elon tweets on weekends
    No price prediction, just sentiment  accumulation
    So Monday's pipeline has Sat + Sun sentiment  ready
    """
    logger.info(f"Running weekend sentiment collection at {datetime.now()}")

    try:
        from sentiment import get_last_n_days_sentiment
        sentiment_df = get_last_n_days_sentiment(1)
        logger.info("Weekend sentiment collected and stored")

    except Exception as e:
        logger.error(f"Weekend sentiment collection failed: {e}")



def start_scheduler():
    """Start the APScheduler with end of day job"""
    initialize_models()
    run_backfill()

    #Run every weekday at 4:05 PM EST (after market close)
    scheduler.add_job(
        run_daily_pipeline,
        trigger=CronTrigger(
            day_of_week='mon-fri',
            hour=16,
            minute=5,
            timezone="America/New_York"
        ),
        id='daily_pipeline',
        name='End of Day TSLA Prediction Pipeline',
        replace_existing=True
    )

    #Collect weekend tweets Sat + Sun at 8 PM EST
    scheduler.add_job(
        run_weekend_sentiment_collection,
        trigger=CronTrigger(
            day_of_week='sat,sun',
            hour=20,
            minute=0,
            timezone="America/New_York"
        ),
        id='weekend_sentiment',
        name='Weekend Sentiment Collection',
        replace_existing=True
    )

    scheduler.start()
    logger.info("Scheduler started - "
                "daily pipeline runs at 4:05 PM EST (Mon-Fri), "
                "sentiment collection runs at 8:00 PM EST (Sat-Sun)"
            )


def stop_scheduler():
    """Gracefully stop the scheduler"""
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler stopped")

if __name__ == "__main__":
    #Test - run the pipeline immediately
    initalize_models()
    run_daily_pipeline()