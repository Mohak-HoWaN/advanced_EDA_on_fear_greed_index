import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- Page Configuration ---
st.set_page_config(
    page_title="Fear & Greed Index Dashboard",
    page_icon="📊",
    layout="wide"
)

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
st.title('Fear & Greed Index: Step-by-Step Analysis')
st.markdown('Welcome! This dashboard helps you understand how market sentiment (Fear & Greed) changes over time and what it might mean for trading. Each section has a plot, a simple description, and a key insight.')
selected = st.sidebar.multiselect('Choose Sentiments', ['Fear', 'Neutral', 'Greed'], default=['Fear', 'Neutral', 'Greed'])
filtered_df = df[df['sentiment'].isin(selected)]

# Update CSS to fix hyperlink styling and duplication
st.markdown(
    """
    <style>
    html {
        scroll-behavior: smooth;
    }

    /* Target links inside the sidebar */
    section[data-testid="stSidebar"] a {
        text-decoration: none !important;
        color: inherit !important;
    }

    section[data-testid="stSidebar"] a:hover {
        text-decoration: none !important;
        color: inherit !important;
    }

    /* Make anchors scroll to the right position */
    a[id] {
        scroll-margin-top: 90px; /* adjust if your top bar is bigger */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add markdown portion back and ensure proper anchoring for all points
st.sidebar.markdown("""
<ul>
    <li><a href="#sentiment-counts">Sentiment Counts</a></li>
    <li><a href="#sentiment-time-series">Sentiment Time Series</a></li>
    <li><a href="#rolling-greed-average">Rolling Greed Average</a></li>
    <li><a href="#sentiment-volatility">Sentiment Volatility</a></li>
    <li><a href="#sentiment-transition-matrix">Transition Matrix</a></li>
    <li><a href="#longest-streaks-and-extreme-sentiment-periods">Longest Streaks</a></li>
    <li><a href="#summary--key-takeaways">Summary</a></li>
</ul>
""", unsafe_allow_html=True)

# Add anchors before all section titles
st.markdown('<a id="sentiment-counts"></a><h2>1. How Often is the Market in Fear, Neutral, or Greed?</h2>', unsafe_allow_html=True)
st.markdown('This bar chart shows how many days the market was classified as Fear, Neutral, or Greed.')
counts = filtered_df['sentiment'].value_counts().reindex(['Fear', 'Neutral', 'Greed']).fillna(0)
fig1, ax1 = plt.subplots()
sns.barplot(x=counts.index, y=counts.values, palette=['red', 'gray', 'green'], ax=ax1)
ax1.set_title('Sentiment Day Counts')
ax1.set_ylabel('Count')
st.pyplot(fig1)
st.info(f"Most common sentiment: {counts.idxmax()} ({int(counts.max())} days). This can help you identify the prevailing mood in the market and adjust your strategy accordingly.")

# 2. Sentiment time series
st.markdown('<a id="sentiment-time-series"></a><h2>2. How Does Sentiment Change Over Time?</h2>', unsafe_allow_html=True)
st.markdown('This line plot shows how market sentiment moves between Fear, Neutral, and Greed over time.')
fig2, ax2 = plt.subplots(figsize=(10, 3))
sns.lineplot(x='date', y='sentiment_num', data=filtered_df, ax=ax2)
ax2.set_title('Sentiment Time Series (Greed=1, Neutral=0.5, Fear=0)')
ax2.set_ylabel('Sentiment Level')
st.pyplot(fig2)
st.info('Sentiment often stays in one state for several days before switching. Watch for sudden jumps! If you notice a rapid change, it may signal a shift in market regime or an opportunity for contrarian trades.')

# 3. Rolling Greed Average
st.markdown('<a id="rolling-greed-average"></a><h2>3. Trend: 30-Day Rolling Average of Greed</h2>', unsafe_allow_html=True)
st.markdown('This plot smooths out daily changes to show the overall trend in market sentiment.')
fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(df['date'], df['rolling_greed'], color='blue')
ax3.set_title('30-Day Rolling Average of Greed Sentiment')
ax3.set_ylabel('Rolling Average')
st.pyplot(fig3)
st.info('When the rolling average is high, the market is mostly greedy. When low, mostly fearful. Sustained trends in the rolling average can help you spot momentum or mean-reversion opportunities.')

# 4. Sentiment Volatility (rolling std)
st.markdown('<a id="sentiment-volatility"></a><h2>4. Volatility: How Much Does Sentiment Change?</h2>', unsafe_allow_html=True)
st.markdown('This plot shows how much sentiment is changing (volatility). High volatility means the market is switching between fear and greed more often.')
df['sentiment_volatility'] = df['sentiment_num'].rolling(window=30).std()
fig4, ax4 = plt.subplots(figsize=(10, 3))
ax4.plot(df['date'], df['sentiment_volatility'], color='purple')
ax4.set_title('30-Day Rolling Volatility of Sentiment')
ax4.set_ylabel('Volatility (Std Dev)')
st.pyplot(fig4)
st.info('Spikes in volatility can signal big changes in market mood. These may be good times to watch for reversals or new trends.')

# Sentiment Transition Matrix
import numpy as np
sentiments = ['Fear', 'Neutral', 'Greed']
transition_counts = pd.DataFrame(0, index=sentiments, columns=sentiments)
for prev, curr in zip(df['sentiment'][:-1], df['sentiment'][1:]):
    if prev in sentiments and curr in sentiments:
        transition_counts.loc[prev, curr] += 1
transition_probs = transition_counts.div(transition_counts.sum(axis=1), axis=0)

# 5. Sentiment Transition Matrix
st.markdown('<a id="sentiment-transition-matrix"></a><h2>5. How Does Sentiment Switch from One State to Another?</h2>', unsafe_allow_html=True)
st.markdown('This table shows the probability that the market will stay in the same sentiment or switch to another the next day.')
st.dataframe(transition_probs.style.format('{:.2f}'))
st.info('The market usually stays in the same state, but sometimes switches. For example, after Fear, it often stays in Fear, but can move to Neutral or Greed. Understanding these probabilities can help you anticipate likely sentiment shifts and plan your trades.')

# 6. Longest Streaks and Extreme Sentiment Periods
st.markdown('<a id="longest-streaks-and-extreme-sentiment-periods"></a><h2>6. Longest Streaks and Extreme Sentiment Periods</h2>', unsafe_allow_html=True)
st.markdown('Here we show the longest periods where the market stayed in Fear or Greed, and all streaks of 5 days or more.')
st.write(f'**Longest Fear Streak:** {longest_fear} days')
st.write(f'**Longest Greed Streak:** {longest_greed} days')

st.dataframe(extreme_periods.sort_values(by='length', ascending=False))
st.info('Long streaks of Fear or Greed are rare. After such streaks, the market often reverses. Monitoring for the end of a long streak can help you catch turning points in the market.')

# --- Summary Section ---
st.markdown('<a id="summary--key-takeaways"></a><h2>Summary & Key Takeaways</h2>', unsafe_allow_html=True)
st.markdown('''
**Key Insights:**
- The market spends most of its time in one sentiment regime, but transitions do occur and can be anticipated.
- High volatility in sentiment often precedes major market moves—watch for these periods.
- Long streaks of Fear or Greed are uncommon and often followed by reversals.
- Use rolling averages and transition probabilities to inform momentum or contrarian strategies.

**Actionable Tips:**
- Adjust your trading style based on the prevailing sentiment and its recent history.
- Be alert for regime changes after long streaks or volatility spikes.
- Combine sentiment analytics with price/volume data for deeper insights.
''')

st.markdown('---')
st.markdown('**Tip:** Combine these sentiment analytics with price or volume data for even deeper trading insights.')