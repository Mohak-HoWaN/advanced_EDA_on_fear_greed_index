import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Load and preprocess data
df = pd.read_csv('fear_greed_index.csv')
df.columns = df.columns.str.lower()
df['date'] = pd.to_datetime(df['date'])

# Map sentiments to simplified categories
sentiment_map = {'extreme fear': 'Fear', 'fear': 'Fear',
                 'extreme greed': 'Greed', 'greed': 'Greed',
                 'neutral': 'Neutral'}
df['sentiment'] = df['classification'].str.lower().map(sentiment_map)

# Numeric sentiment for plotting
sentiment_num_map = {'Fear': 0, 'Neutral': 0.5, 'Greed': 1}
df['sentiment_num'] = df['sentiment'].map(sentiment_num_map)

# Rolling average for Greed sentiment
df['rolling_greed'] = df['sentiment_num'].rolling(window=30).mean()

# Identify streaks of sentiments
streaks = []
current_streak = {'type': None, 'length': 0, 'start': None, 'end': None}
for idx, row in df.iterrows():
    sentiment = row['sentiment']
    if current_streak['type'] != sentiment:
        if current_streak['type'] is not None:
            streaks.append(current_streak.copy())
        current_streak = {'type': sentiment, 'length': 1, 'start': row['date'], 'end': row['date']}
    else:
        current_streak['length'] += 1
        current_streak['end'] = row['date']
if current_streak['type'] is not None:
    streaks.append(current_streak.copy())

streaks_df = pd.DataFrame(streaks)
longest_fear = streaks_df[streaks_df['type'] == 'Fear']['length'].max()
longest_greed = streaks_df[streaks_df['type'] == 'Greed']['length'].max()
extreme_periods = streaks_df[streaks_df['length'] >= 5]

# Streamlit dashboard layout
st.title('Advanced EDA: Fear & Greed Index Analysis')

selected = st.sidebar.multiselect('Choose Sentiments', ['Fear', 'Neutral', 'Greed'], default=['Fear', 'Neutral', 'Greed'])
filtered_df = df[df['sentiment'].isin(selected)]

# Plot 1: Sentiment counts
counts = filtered_df['sentiment'].value_counts().reindex(['Fear', 'Neutral', 'Greed']).fillna(0)
fig1, ax1 = plt.subplots()
sns.barplot(x=counts.index, y=counts.values, palette=['red', 'gray', 'green'], ax=ax1)
ax1.set_title('Sentiment Day Counts')
ax1.set_ylabel('Count')

# Plot 2: Sentiment time series
fig2, ax2 = plt.subplots(figsize=(10, 3))
sns.lineplot(x='date', y='sentiment_num', data=filtered_df, ax=ax2)
ax2.set_title('Sentiment Time Series (Greed=1, Neutral=0.5, Fear=0)')
ax2.set_ylabel('Sentiment Level')

# Plot 3: Rolling Greed Average
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(df['date'], df['rolling_greed'], color='blue')
ax3.set_title('30-Day Rolling Average of Greed Sentiment')
ax3.set_ylabel('Rolling Average')

st.pyplot(fig1)
st.pyplot(fig2)
st.pyplot(fig3)

# Show streak summaries
st.subheader('Longest Sentiment Streaks')
st.write(f'Longest Fear Streak: {longest_fear} days')
st.write(f'Longest Greed Streak: {longest_greed} days')

st.subheader('Extreme Sentiment Periods (Streaks >= 5 days)')
st.dataframe(extreme_periods.sort_values(by='length', ascending=False))