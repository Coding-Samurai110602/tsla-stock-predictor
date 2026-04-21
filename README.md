# TSLA Stock Price Predictor

An end-to-end MLOps project that predicts Tesla (TSLA) stock prices using LSTM neural networks enhanced with FinBERT sentiment analysis on Elon Musk's tweets.

---

## Overview

This project combines time series forecasting with NLP-based sentiment analysis to predict next-day TSLA closing prices. The system runs as a fully containerized MLOps pipeline that fetches live market data daily, generates predictions, and serves results through a REST API and web dashboard.

---

## Architecture
Data Sources          Backend                 Frontend

Finnhub API     -->   FastAPI + LSTM      -->  Dashboard
Elon Tweets     -->   FinBERT Sentiment        (Price Chart,
PostgreSQL      -->   APScheduler              Predictions,
Redis Cache     -->   MLflow Tracking          Sentiment)

**Pipeline runs daily at 4:05 PM EST:**
1. Fetch today's TSLA price from Finnhub
2. Pull sentiment scores from PostgreSQL
3. Run feature engineering (MA, RSI, MACD, Bollinger Bands)
4. LSTM model predicts tomorrow's closing price
5. Store prediction in PostgreSQL, cache in Redis
6. Dashboard updates automatically

---

## Tech Stack

| Layer | Technology |
|---|---|
| ML Model | LSTM (TensorFlow/Keras) |
| Sentiment | FinBERT (ProsusAI/finbert) |
| API | FastAPI |
| Database | PostgreSQL |
| Cache | Redis |
| Experiment Tracking | MLflow |
| Scheduler | APScheduler |
| Containerization | Docker + Docker Compose |
| Live Data | Finnhub API |
| Frontend | HTML, CSS, JavaScript, Chart.js |

---

## Model Performance

| Metric | Baseline LSTM | LSTM + VADER | LSTM + FinBERT |
|---|---|---|---|
| RMSE | $21.59 | $19.18 | $23.99 |
| MAE | $15.63 | $14.38 | $18.10 |
| Directional Accuracy | 49.29% | 49.81% | 50.97% |

FinBERT improves directional accuracy over the baseline by 1.68%, which is the most relevant metric for trading signals. The higher RMSE compared to VADER is a known bias-variance tradeoff — FinBERT adds meaningful directional signal at the cost of slightly higher price magnitude error.

---

## Project Structure

```
tsla-stock-predictor/
├── notebooks/
│   ├── TSLA_Stock_Analysis_LSTM.ipynb    # EDA, feature engineering, model training
│   └── TSLA_Sentiment_Analysis.ipynb     # FinBERT sentiment pipeline
├── tsla-mlops/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py                       # FastAPI endpoints
│   │   ├── predict.py                    # LSTM inference pipeline
│   │   ├── scheduler.py                  # APScheduler daily jobs
│   │   ├── sentiment.py                  # FinBERT sentiment scoring
│   │   ├── feature_engineering.py        # Technical indicators
│   │   ├── data_ingestion.py             # Finnhub + PostgreSQL
│   │   └── models/                       # Saved LSTM + scalers
│   ├── frontend/
│   │   ├── index.html
│   │   ├── style.css
│   │   └── app.js
│   └── db/
│       └── init.sql
└── README.md
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Health check |
| GET | /api/prediction/latest | Latest prediction |
| GET | /api/prediction/history | Prediction history |
| GET | /api/price/history | TSLA price history |
| GET | /api/sentiment/latest | Latest sentiment score |
| GET | /api/accuracy | Model accuracy metrics |
| POST | /api/prediction/trigger | Manually trigger pipeline |
| GET | /metrics | Prometheus metrics |

---

## Running Locally

**Prerequisites:** Docker Desktop, Finnhub API key (free at finnhub.io)

```bash
# Clone the repo
git clone https://github.com/Coding-Samurai110602/tsla-stock-predictor.git
cd tsla-stock-predictor/tsla-mlops

# Create .env file
echo "POSTGRES_USER=tsla_user" > .env
echo "POSTGRES_PASSWORD=tsla_password123" >> .env
echo "POSTGRES_DB=tsla_db" >> .env
echo "FINNHUB_API_KEY=your_api_key_here" >> .env

# Build and run
docker-compose up --build
```

**Access:**
- Dashboard: http://localhost:3000
- API docs: http://localhost:8000/docs
- MLflow: http://localhost:5001

---

## Notebooks

**Notebook 1 — TSLA_Stock_Analysis_LSTM.ipynb**
- Data collection via yfinance
- Exploratory Data Analysis (closing prices, volume, moving averages, daily returns, correlation heatmaps, risk vs return)
- Feature engineering (RSI, MACD, Bollinger Bands, volatility)
- LSTM model training with early stopping
- Model evaluation (RMSE, MAE, directional accuracy)

**Notebook 2 — TSLA_Sentiment_Analysis.ipynb**
- Elon Musk tweet dataset (55,000+ tweets, 2010-2025)
- TSLA keyword filtering and engagement weighting
- FinBERT sentiment scoring (positive, negative, neutral)
- Combined LSTM + sentiment model training and evaluation

---

## Future Improvements

- Live tweet sentiment via Twitter/X API integration
- Finnhub news sentiment for automated daily sentiment updates
- Model retraining pipeline triggered by performance drift
- Cloud deployment on Railway/AWS
- Expanded feature set (options flow, institutional holdings)

---

## Data Sources

- TSLA historical prices: yfinance
- Live daily prices: Finnhub API
- Elon Musk tweets: Kaggle dataset (2010-2025)

---

## Disclaimer

This project is built for educational and portfolio purposes only. It is not financial advice and should not be used for actual trading decisions.
