-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- TSLA daily price history
CREATE TABLE IF NOT EXISTS price_history (
    id              SERIAL PRIMARY KEY,
    date            DATE NOT NULL UNIQUE,
    open            FLOAT NOT NULL,
    high            FLOAT NOT NULL,
    low             FLOAT NOT NULL,
    close           FLOAT NOT NULL,
    volume          BIGINT NOT NULL,
    created_at      TIMESTAMP DEFAULT NOW()
);

-- Daily sentiment scores
CREATE TABLE IF NOT EXISTS sentiment_history (
    id                  SERIAL PRIMARY KEY,
    date                DATE NOT NULL UNIQUE,
    avg_sentiment       FLOAT,
    weighted_sentiment  FLOAT,
    total_engagement    BIGINT,
    tweet_count         INT DEFAULT 0,
    positive_count      INT DEFAULT 0,
    negative_count      INT DEFAULT 0,
    created_at          TIMESTAMP DEFAULT NOW()
);

-- Model predictions
CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    date             DATE NOT NULL UNIQUE,
    predicted_price  FLOAT NOT NULL,
    predicted_dir    VARCHAR(10) NOT NULL,
    confidence       FLOAT NOT NULL,
    actual_price     FLOAT,
    was_correct      BOOLEAN,
    created_at       TIMESTAMP DEFAULT NOW()
);

-- Model performance tracking
CREATE TABLE IF NOT EXISTS model_metrics (
    id            SERIAL PRIMARY KEY,
    date          DATE NOT NULL,
    rmse          FLOAT,
    mae           FLOAT,
    dir_accuracy  FLOAT,
    model_version VARCHAR(50),
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Indexes for fast querying
CREATE INDEX IF NOT EXISTS idx_price_date ON price_history(date);
CREATE INDEX IF NOT EXISTS idx_sentiment_date ON sentiment_history(date);
CREATE INDEX IF NOT EXISTS idx_prediction_date ON predictions(date);

-- Seed model metrics with current results
INSERT INTO model_metrics (date, rmse, mae, dir_accuracy, model_version)
VALUES (CURRENT_DATE, 23.99, 18.10, 50.97, 'finbert_v1')
ON CONFLICT DO NOTHING;