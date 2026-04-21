import numpy as np
import pandas as pd
import pickle
import logging
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine, text
from datetime import date, datetime
import os
import redis
import json
from dotenv import load_dotenv
from feature_engineering import get_feature_columns, run_feature_engineering

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 60
CACHE_KEY = "latest_prediction"
CACHE_TTL = 86400

#Redis client
redis_client = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    port=int(os.getenv('REDIS_PORT', '6379')),
    decode_responses=True
)

def get_db_engine():
    url = (
        f"postgresql://{os.getenv('POSTGRES_USER')}:"
        f"{os.getenv('POSTGRES_PASSWORD')}@"
        f"{os.getenv('POSTGRES_HOST', 'localhost')}:"
        f"{os.getenv('POSTGRES_PORT', '5432')}/"
        f"{os.getenv('POSTGRES_DB')}"
    )
    return create_engine(url)

def load_artifacts():
    """Load LSTM model and scaler"""
    try:
        logger.info("Loading model artifacts...")
        model = load_model('models/tsla_finbert_model.keras')
        with open('models/scaler_sentiment.pkl', 'rb') as f:
            scaler = pickle.load(f)
        logger.info("Model and scaler loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"Error loading artifacts: {e}")
        return None, None

def prepare_sequence(df: pd.DataFrame, scaler) -> np.ndarray:
    """Prepare 60-day sequence for LSTM input"""
    try:
        features = get_feature_columns()
        data = df[features].values

        if len(data) < SEQUENCE_LENGTH:
            logger.error(f"Not enough data: need {SEQUENCE_LENGTH}, got {len(data)}")
            return None

        #Scale data
        scaled = scaler.transform(data)

        #Take last 60 days
        sequence = scaled[-SEQUENCE_LENGTH:]
        sequence = sequence.reshape(1, SEQUENCE_LENGTH, len(features))
        logger.info(f"Prepared sequence shape: {sequence.shape}")
        return sequence

    except Exception as e:
        logger.error(f"Error preparing sequence: {e}")
        return None


def inverse_transform_price(scaled_price: float, scaler) -> float:
    """Inverse transform scaled price back to USD"""
    features = get_feature_columns()
    dummy = np.zeros((1, len(features)))
    dummy[0, 0] = scaled_price
    return float(scaler.inverse_transform(dummy)[0, 0])


def calculate_confidence(model, sequence: np.ndarray, n_iterations: int = 10) -> float:
    """
    Monte Carlo Dropout for confidence estimation
    Runs model N times with dropout active to get prediction variance
    """
    predictions = []
    for _ in range(n_iterations):
        pred = model(sequence, training=True) #training=True keeps the droupout active
        predictions.append(float(pred.numpy()[0, 0]))

    std = np.std(predictions)
    mean = np.mean(predictions)

    #Normalize confidence: lower std = higher confidence
    confidence = max(0.0, min(1.0, 1.0 - (std / (abs(mean) + 1e-8))))
    return round(confidence * 100, 2)


def predict_next_day(
    df: pd.DataFrame,
    model,
    scaler,
    current_price: float
) -> dict:
    """Run full prediction pipeline for next trading day"""
    try:
        #Prepare sequence
        sequence = prepare_sequence(df, scaler)
        if sequence is None:
            return None

        #Make prediction
        scaled_pred = float(model.predict(sequence, verbose=0)[0, 0])
        predicted_price = inverse_transform_price(scaled_pred, scaler)

        #Direction and confidence
        direction = 'UP' if predicted_price > current_price else 'DOWN'
        confidence = calculate_confidence(model, sequence)

        result = {
            'date': date.today().isoformat(),
            'predicted_price': round(predicted_price, 2),
            'predicted_dir': direction,
            'confidence': confidence,
            'current_price': round(current_price, 2),
            'price_change': round(predicted_price - current_price, 2),
            'price_change_pct': round(
                (predicted_price - current_price) / current_price * 100, 2),
            'created_at': datetime.now().isoformat()
        }

        logger.info(f"Prediction: ${predicted_price: .2f} ({direction}) "
                    f"with {confidence: .1f}% confidence")
        return result

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return None


def store_prediction(prediction: dict):
    """Store prediction to PostgreSQL"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO predictions (
                    date, predicted_price, predicted_dir, confidence
                )
                VALUES (:date, :predicted_price, :predicted_dir, :confidence)
                ON CONFLICT (date) DO UPDATE SET
                    predicted_price = EXCLUDED.predicted_price,
                    predicted_dir = EXCLUDED.predicted_dir,
                    confidence = EXCLUDED.confidence
            """), {
                'date': prediction['date'],
                'predicted_price': prediction['predicted_price'],
                'predicted_dir': prediction['predicted_dir'],
                'confidence': prediction['confidence']
            })
            conn.commit()
        logger.info(f"Stored prediction for {prediction['date']}")
        return True

    except Exception as e:
        logger.error(f"Error storing prediction: {e}")
        return False

def cache_prediction(prediction: dict):
    """Cache latest prediction in Redis"""
    try:
        redis_client.setex(
            CACHE_KEY,
            CACHE_TTL,
            json.dumps(prediction)
        )
        logger.info("Prediciton cached in Redis")
        return True
    except Exception as e:
        logger.error(f"Error caching prediction: {e}")
        return False

def get_cached_prediction() -> dict:
    """Retrive cached prediction from Redis"""
    try:
        cached = redis_client.get(CACHE_KEY)
        if cached:
            logger.info("Returning cached prediciton")
            return json.loads(cached)
            return None
    except Exception as e:
        logger.error(f"Error retrieving cached prediction: {e}")
        return None

def update_actual_price(target_date: str, actual_price: float):
    """Update actual price and accuracy for previous prediction"""
    try:
        engine = get_db_engine()
        with engine.connect() as conn:
            # Get prediction for that date
            result = conn.execute(text("""
                SELECT predicted_price, predicted_dir
                FROM predictions
                WHERE date = :date
            """), {'date': target_date})
            row = result.fetchone()

            if row:
                pred_price = row[0]
                pred_dir = row[1]

                #Check if direction was correct
                was_correct = (
                    (pred_dir == 'UP' and actual_price > pred_price) or
                    (pred_dir == 'DOWN' and actual_price < pred_price)
                )

                conn.execute(text("""
                    UPDATE predictions
                    SET actual_price = :actual_price,
                        was_correct = :was_correct
                    WHERE date = :date
                """), {
                    'actual_price': actual_price,
                    'was_correct': was_correct,
                    'date': target_date
                })
                conn.commit()
                logger.info(f"Updated actual price for {target_date}: "
                            f"${actual_price: .2f} - Correct: {was_correct}")

            return True

    except Exception as e:
        logger.error(f"Error updating actual price: {e}")
        return False


def log_to_mlflow(prediction: dict, rmse: float = None, mae: float = None):
    """Log prediction metrics to MLflow"""
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5001'))
        mlflow.set_experiment('tsla_daily_predicitons')

        with mlflow.start_run():
            mlflow.log_param("model_version", "finbert_v1")
            mlflow.log_param("sequence_length", "SEQUENCE_LENGTH")
            mlflow.log_metric("predicted_price", prediction['predicted_price'])
            mlflow.log_metric("confidence", prediction['confidence'])
            if rmse:
                mlflow.log_metric("rmse", rmse)
            if mae:
                mlflow.log_metric("mae", mae)

        logger.info("Logged to MLflow")
        return True

    except Exception as e:
        logger.error(f"Error logging to MLflow: {e}")
        return False


if __name__ == "__main__":
    #Test prediction pipeline
    import yfinace as yf
    from data_ingestion import fetch_historical_tsla
    from sentiment import get_last_n_days_sentiment

    #Load artifacts
    model, scaler = load_artifacts()
    if model and scaler:
        #Get price data
        price_df = fetch_historical_tsla()

        #Get sentiment data
        sentiment_df = get_last_n_days_sentiment(100)

        if price_df is not None and sentiment_df is not None:
            #Feature engineering
            df = run_feature_engineering(price_df, sentiment_df)

            #Predict
            current_price = float(price_df.iloc[-1]['close'])
            prediction = predict_next_day(df, model, scaler, current_price)

            if prediction:
                print("Prediction:", prediction)
                store_prediction(prediction)
                cache_prediction(prediction)
                log_to_mlflow(prediction)

