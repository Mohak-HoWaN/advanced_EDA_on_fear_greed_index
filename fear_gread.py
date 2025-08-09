import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
df = pd.read_csv('fear_greed_index.csv')
df['date'] = pd.to_datetime(df['date'])
df.sort_values('date', inplace=True)

# 2. Overview: Missing Values & Data Types
print(df.info())
print("Missing values per column:")
print(df.isnull().sum())
print("Unique Classifications:", df['classification'].unique())
print(df.head())

# 3. Sentiment Count Plot
plt.figure(figsize=(6,4))
sns.countplot(x="classification", data=df)
plt.title("Count of Fear vs Greed Days")
plt.show()

# 4. Time Series Plot
plt.figure(figsize=(12,5))
sns.lineplot(x='Date', y=df['Classification'].apply(lambda x: 1 if x.lower() == 'greed' else 0))
plt.title("Sentiment (Greed=1, Fear=0) Over Time")
plt.ylabel("Sentiment Label")
plt.xlabel("Date")
plt.show()

# 5. Rolling Window Analysis
window = 30  # 30-day rolling average
df['Sentiment_num'] = df['Classification'].apply(lambda x: 1 if x.lower() == 'greed' else 0)
df['Rolling_Greed'] = df['Sentiment_num'].rolling(window=window).mean()

plt.figure(figsize=(12,5))
plt.plot(df['Date'], df['Rolling_Greed'])
plt.title(f"{window}-Day Rolling Average of Greed Sentiment")
plt.ylabel("Average Greed Level")
plt.xlabel("Date")
plt.show()

# 6. Streaks and Pattern Discovery
streaks = []
current_streak = {'type': None, 'length': 0, 'start': None, 'end': None}
for idx, row in df.iterrows():
    sentiment = row['Classification']
    if current_streak['type'] is None or sentiment != current_streak['type']:
        if current_streak['type'] is not None:
            streaks.append(current_streak.copy())
        current_streak = {'type': sentiment, 'length': 1, 'start': row['Date'], 'end': row['Date']}
    else:
        current_streak['length'] += 1
        current_streak['end'] = row['Date']
if current_streak['type'] is not None:
    streaks.append(current_streak.copy())

streaks_df = pd.DataFrame(streaks)
print("\nLongest Fear Streak:", streaks_df[streaks_df['type']=='Fear']['length'].max())
print("Longest Greed Streak:", streaks_df[streaks_df['type']=='Greed']['length'].max())

# 7. Show Periods of Extreme Sentiment (Optional Highlight)
extreme_periods = streaks_df[streaks_df['length'] >= 5]  # e.g., streaks >=5 days
print("\nExtreme Sentiment Streaks (>=5 days):")
print(extreme_periods)

# Optional: Save aggregated results for dashboard use
summary = df['Classification'].value_counts()
summary.to_csv("sentiment_summary.csv")

