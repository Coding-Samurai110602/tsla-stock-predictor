import pandas as pd
from sentiment import load_finbert, score_tweets, aggregate_daily_sentiment, store_sentiment

print("Loading FinBERT...")
finbert = load_finbert()

print("Loading tweets...")
df = pd.read_csv('/app/all_musk_posts.csv', low_memory=False)
print(f"Loaded {len(df)} tweets")

df = df[['fullText', 'createdAt', 'isRetweet', 'likeCount', 'retweetCount']]
df['createdAt'] = pd.to_datetime(df['createdAt'], utc=True)
df['date'] = df['createdAt'].dt.date
df = df[df['isRetweet'] != True]
df = df.sort_values('createdAt').reset_index(drop=True)

tsla_keywords = [
    'tesla', 'tsla', 'model s', 'model 3', 'model x', 'model y',
    'cybertruck', 'electric', 'ev', 'autopilot', 'fsd', 'giga',
    'production', 'delivery', 'stock', 'short', 'shareholders',
    'battery', 'supercharger', 'roadster', 'semi', 'powerwall'
]
pattern = '|'.join(tsla_keywords)
df_tsla = df[df['fullText'].str.lower().str.contains(pattern, na=False)].copy()
print(f"TSLA tweets: {len(df_tsla)}")

df_tsla['likeCount'] = df_tsla['likeCount'].fillna(1)
df_tsla['retweetCount'] = df_tsla['retweetCount'].fillna(1)

tweets_list = df_tsla[['fullText', 'likeCount', 'retweetCount', 'date']].rename(
    columns={'fullText': 'text', 'likeCount': 'like_count', 'retweetCount': 'retweet_count'}
).to_dict('records')

print("Scoring tweets with FinBERT (this will take a while)...")
scored = score_tweets(tweets_list, finbert)
print(f"Scored {len(scored)} tweets")

dates = df_tsla['date'].unique()
print(f"Aggregating {len(dates)} days...")

count = 0
for d in sorted(dates):
    day_tweets = [t for t in scored if t['date'] == d]
    if day_tweets:
        sentiment = aggregate_daily_sentiment(day_tweets, d)
        store_sentiment(sentiment)
        count += 1
        if count % 100 == 0:
            print(f"Stored {count} days...")

print(f"Done! Stored sentiment for {count} days")
