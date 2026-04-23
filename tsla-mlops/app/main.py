import logging
import os
from datetime import date, datetime, timedelta
from typing import Optional
from contextlib import asynccontextmanager

import redis
import json
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from prometheus_fastapi_instrumentator import Instrumentator

from scheduler import start_scheduler, stop_scheduler, run_daily_pipeline
from data_ingestion import fetch_historical_tsla, store_historical_prices
import subprocess

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', 6379)),
    decode_responses=True
)

# --- DATABASE ---
def get_db_engine():
    url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)


# --- LIFESPAN ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start scheduler on startup, stop on shutdown"""
    logger.info("Starting TSLA Predictor API...")
    start_scheduler()
    yield
    logger.info("Shutting down TSLA Predictor API...")
    stop_scheduler()


# --- APP ---
app = FastAPI(
    title="TSLA Price Predictor API",
    description="End of day TSLA price prediction using LSTM + FinBERT sentiment",
    version="1.0.0",
    lifespan=lifespan
)

#CORS - allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/dashboard", 
    StaticFiles(directory="/app/frontend", html=True), 
    name="frontend"
)

#Prometheus metrics
Instrumentator().instrument(app).expose(app)


# --- PYDANTIC MODELS ---
class PredictionResponse(BaseModel):
    date: str
    predicted_price: float
    predicted_dir: str
    confidence: float
    current_price: float
    price_change: float
    price_change_pct: float
    created_at: str

class ManualTriggerResponse(BaseModel):
    message: str
    triggered_at: str


# --- ROUTES ---
@app.get("/health")
def health_check():
    """Health check endpoint for Docker"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "tsla-predictor"
    }

@app.get("/api/prediction/latest", response_model=PredictionResponse)
def get_latest_prediction():
    """
    Get today's prediction
    Checks Redis cache first, fall back to PostgreSQL
    """
    #Try Redis cache first
    try:
        cached = redis_client.get("latest_prediction")
        if cached:
            logger.info("Reurning cached prediction")
            return json.loads(cached)
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}")

    #Fallback to PostgreSQL
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, predicted_price, predicted_dir,
                    confidence, created_at
                FROM predictions
                ORDER BY date DESC
                LIMIT 1
            """))
            row = result.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="No predictions found yet. Wait for market close."
            )

        return {
                "date": str(row[0]),
                "predicted_price": row[1],
                "predicted_dir": row[2],
                "confidence": row[3],
                "current_price": 0.0,
                "price_change": 0.0,
                "price_change_pct": 0.0,
                "created_at": str(row[4])
            }   

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching predictions: {e}")
        raise HTTPException(status_code=500, detail=str(e))    

@app.get("/api/prediction/history")
def get_prediction_history(limit: int = 30):
    """Get past N predictions with actual prices and accuracy"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, predicted_price, predicted_dir,
                       confidence, actual_price, was_correct
                FROM predictions
                ORDER BY date DESC
                LIMIT :limit
            """), {'limit': limit})
            rows = result.fetchall()

        return {
            "predictions": [
                {
                    "date": str(r[0]),
                    "predicted_price": r[1],
                    "predicted_dir": r[2],
                    "confidence": r[3],
                    "actual_price": r[4],
                    "was_correct": r[5]
                }
                for r in rows
            ],
            "total": len(rows)
        }

    except Exception as e:
        logger.error(f"Error fetching prediction history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/price/history")
def get_price_history(days: int = 90):
    """Get last N days of TSLA price history"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, open, high, low, close, volume
                FROM price_history
                ORDER BY date DESC
                LIMIT :days
            """), {'days': days})
            rows = result.fetchall()

        return {
            "prices": [
                {
                    "date": str(r[0]),
                    "open": r[1],
                    "high": r[2],
                    "low": r[3],
                    "close": r[4],
                    "volume": r[5]
                }
                for r in rows
            ],
            "total": len(rows)
        }

    except Exception as e:
        logger.error(f"Error fetching price history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sentiment/latest")
def get_latest_sentiment():
    """Get most recent sentiment analysis"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, avg_sentiment, weighted_sentiment,
                    total_engagement, tweet_count,
                    positive_count, negative_count
                FROM sentiment_history
                ORDER BY date DESC
                LIMIT 1
            """))
            row = result.fetchone()

        if not row:
            raise HTTPException(
                status_code=404,
                detail="No sentiment data found"
            )

        return {
            "date": str(row[0]),
            "avg_sentiment": row[1],
            "weighted_sentiment": row[2],
            "total_engagement": row[3],
            "tweet_count": row[4],
            "positive_count": row[5],
            "negative_count": row[6],
            "sentiment_label": (
                "Positive" if row[1] > 0.05
                else "Negative" if row[1] < -0.05
                else "Neutral"
            )
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sentiment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/accuracy")
def get_model_accuracy():
    """Get overall model directional accuracy over time"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT
                    COUNT(*) as total_predictions,
                    SUM(CASE WHEN was_correct THEN 1 ELSE 0 END) as correct,
                    ROUND(
                        AVG(CASE WHEN was_correct THEN 1.0 ELSE 0.0 END) * 100,
                        2
                    ) as accuracy_pct
                FROM predictions
                WHERE was_correct IS NOT NULL
            """))
            row = result.fetchone()

        #Also getting model metrics
        with engine.connect() as conn:
            metrics = conn.execute(text("""
                SELECT rmse, mae, dir_accuracy, model_version
                FROM model_metrics
                ORDER BY created_at DESC
                LIMIT 1
            """))
            m = metrics.fetchone()

        return {
            "total_predictions": row[0],
            "correct_predictions": row[1],
            "live_accuracy_pct": float(row[2]) if row[2] else 0.0,
            "model_rmse": m[0] if m else 23.99,
            "model_mae": m[1] if m else 18.10,
            "model_dir_accuracy": m[2] if m else 50.97,
            "model_version": m[3] if m else "finbert_v1"
        }

    except Exception as e:
        logger.error(f"Error fetching accuracy: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prediction/trigger", response_model=ManualTriggerResponse)
def trigger_prediction(background_tasks: BackgroundTasks):
    """
    Manually trigger the prediction pipeline
    Usefule for testing without waiting for the market close
    """
    background_tasks.add_task(run_daily_pipeline)
    return {
        "message": "Prediction pipeline triggered successfully",
        "triggered_at": datetime.now().isoformat()
    }

@app.get("/api/sentiment/history")
def get_sentiment_history(days: int = 30):
    """Get last N days of sentiment history"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT date, avg_sentiment, weighted_sentiment,
                       total_engagement, tweet_count
                FROM sentiment_history
                ORDER BY date DESC
                LIMIT :days
            """), {"days" : days})
            rows = result.fetchall()

        return {
            "sentiment" : [
                {
                    "date" : str(r[0]),
                    "avg_sentiment": r[1],
                    "weigted_sentiment": r[2],
                    "total_engagement": r[3],
                    "tweet_count": r[4]
                }
                for r in rows
            ],
            "total": len(rows)
        }

    except Exception as e:
        logger.error(f"Error fetching sentiment history: {e}")
        raise HTTPException(status_code=500, detail=str(0))

@app.post("/api/admin/seed-prices")
def seed_prices():
    df = fetch_historical_tsla()
    if df is None:
        raise HTTPException(status_code=500, detail="Failed to load CSV")
    success = store_historical_prices(df)
    return {"status": "done", "rows": len(df), "stored": success}


@app.post("/api/admin/seed-sentiment")
def seed_sentiment_bg(background_tasks: BackgroundTasks):
    def run_seed():
        subprocess.run(["python3", "/app/seed_sentiment.py"])
    background_tasks.add_task(run_seed)
    return {"status": "started", "note": "Sentiment seeding running in background, takes ~30 min"}