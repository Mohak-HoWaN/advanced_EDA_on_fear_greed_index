import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Trade History & Market Sentiment Dashboard",
    page_icon="📊",
    layout="wide"
)

# Add custom CSS for smooth scrolling and styled navbar
st.markdown("""
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
""", unsafe_allow_html=True)

# Correct sidebar points to avoid duplication and ensure proper structure

# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the sections below to explore the dashboard:")
st.sidebar.markdown("""
<ul>
    <li><a href="#data-overview">Data Overview</a></li>
    <li><a href="#trade-side-distribution">Trade Side Distribution</a></li>
    <li><a href="#most-traded-coins">Most Traded Coins</a></li>
    <li><a href="#trade-size-distribution">Trade Size Distribution</a></li>
    <li><a href="#closed-pnl-overview">Closed PnL Overview</a></li>
    <li><a href="#trade-frequency-over-time">Trade Frequency Over Time</a></li>
    <li><a href="#average-execution-price-over-time-top-3-coins">Execution Price Trends</a></li>
    <li><a href="#fees-paid-by-top-coins">Fees Analysis</a></li>
    <li><a href="#correlation-matrix-of-numeric-features">Correlation Matrix</a></li>
    <li><a href="#trader-performance-by-market-sentiment">Performance by Sentiment</a></li>
    <li><a href="#streaks-of-winning-and-losing-trades">Streak Analysis</a></li>
    <li><a href="#summary--key-takeaways">Summary</a></li>
</ul>
""", unsafe_allow_html=True)

# Load trade data
df = pd.read_csv('historical_data.csv')
df.columns = df.columns.str.strip()
df['Timestamp IST'] = pd.to_datetime(df['Timestamp IST'], errors='coerce')
df['Timestamp'] = pd.to_datetime(df['Timestamp'], errors='coerce')
num_cols = ['Execution Price', 'Size Tokens', 'Size USD', 'Closed PnL', 'Fee']
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df['Side'] = df['Side'].str.upper()

# Load sentiment data and merge
sent = pd.read_csv('fear_greed_index.csv')
sent.columns = sent.columns.str.lower()
sent['date'] = pd.to_datetime(sent['date'])
sent['classification'] = sent['classification'].str.lower()
sentiment_map = {'extreme fear': 'Fear', 'fear': 'Fear', 'extreme greed': 'Greed', 'greed': 'Greed', 'neutral': 'Neutral'}
sent['sentiment'] = sent['classification'].map(sentiment_map)

# Merge on date (align trade to sentiment date)
df['date'] = df['Timestamp IST'].dt.date
sent['date'] = sent['date'].dt.date
df = df.merge(sent[['date', 'sentiment']], on='date', how='left')

# Basic overview in Streamlit dashboard

st.title("📊 Advanced EDA Dashboard: Trade History & Market Sentiment")
st.markdown("This dashboard explores the relationship between trader performance and market sentiment, uncovering patterns and insights for smarter trading.")

st.markdown('<a id="data-overview"></a><h2>📋 Data Overview</h2>', unsafe_allow_html=True)
st.write(f"Total Trades: {len(df)}")
st.write(f"Unique Accounts: {df['Account'].nunique()}")
st.write(f"Unique Coins: {df['Coin'].nunique()}")
st.write(f"Sentiment data coverage: {df['sentiment'].notna().mean():.0%} of trades")

st.subheader("Sample Data")
st.write(df.head())


# 1. Trade Side Distribution
st.markdown('<a id="trade-side-distribution"></a><h2>📈 1. Trade Side Distribution</h2>', unsafe_allow_html=True)
st.markdown("This chart shows the number of buy and sell trades.")
fig1, ax1 = plt.subplots()
sns.countplot(data=df, x='Side', ax=ax1, palette=['green', 'red'])
ax1.set_title("Buy vs Sell Trade Counts")
st.pyplot(fig1)
side = df['Side'].mode()[0]
st.info(f"Most trades are {side}s. This may reflect a market bias or trader preference. If you notice a persistent bias, consider if it aligns with prevailing sentiment or if it exposes you to one-sided risk.")


# 2. Most Traded Coins
st.markdown('<a id="most-traded-coins"></a><h2>💰 2. Most Traded Coins</h2>', unsafe_allow_html=True)
st.markdown("Top 10 coins by number of trades.")
top_coins = df['Coin'].value_counts().head(10)
fig2, ax2 = plt.subplots(figsize=(10,4))
sns.barplot(x=top_coins.index, y=top_coins.values, ax=ax2, palette='viridis')
ax2.set_title("Top 10 Traded Coins by Number of Trades")
ax2.set_xlabel("Coin")
ax2.set_ylabel("Number of Trades")
st.pyplot(fig2)
st.info("The most traded coins may reflect current market trends or trader preferences. Focus your analysis on these coins for the most actionable insights.")


# 3. Trade Size (USD) Distribution by Coin
st.markdown('<a id="trade-size-distribution"></a><h2>📦 3. Trade Size (USD) Distribution by Coin</h2>', unsafe_allow_html=True)
st.markdown("Boxplot of trade sizes (USD) for the top traded coins.")
fig3, ax3 = plt.subplots(figsize=(12,6))
sns.boxplot(data=df[df['Coin'].isin(top_coins.index)], x='Coin', y='Size USD', ax=ax3)
ax3.set_yscale('log')
ax3.set_title("Log Scale Distribution of Trade Size (USD) by Top Coins")
st.pyplot(fig3)
st.info("Wide variation in trade size may indicate different trader types (retail vs. institutional) or changing conviction. Outliers can signal large players or unusual activity.")


# 4. Closed PnL Overview
st.markdown('<a id="closed-pnl-overview"></a><h2>📉 4. Closed PnL Overview</h2>', unsafe_allow_html=True)
st.markdown("Distribution of profit and loss (PnL) for all trades.")
fig4, ax4 = plt.subplots()
sns.histplot(df['Closed PnL'].dropna(), bins=50, kde=True, ax=ax4)
ax4.set_title("Distribution of Closed PnL")
st.pyplot(fig4)
mean_pnl = df['Closed PnL'].mean()
median_pnl = df['Closed PnL'].median()
st.info(f"Average PnL: {mean_pnl:.2f}, Median: {median_pnl:.2f}. If your average PnL is positive, your strategy is profitable overall. Compare the mean and median: if the mean is much higher, a few big wins may be driving results; if the median is higher, your wins are more consistent.")


# 5. Trade Frequency Over Time
st.markdown('<a id="trade-frequency-over-time"></a><h2>📅 5. Trade Frequency Over Time</h2>', unsafe_allow_html=True)
st.markdown("Number of trades per day.")
trade_counts = df.groupby(df['Timestamp IST'].dt.date).size()
fig5, ax5 = plt.subplots(figsize=(10,4))
trade_counts.plot(ax=ax5)
ax5.set_title("Number of Trades Per Day")
ax5.set_xlabel("Date")
ax5.set_ylabel("Number of Trades")
st.pyplot(fig5)
st.info("Spikes or drops in daily trade frequency may be linked to news, sentiment shifts, or market events. Use these patterns to time entries or exits.")


# 6. Average Execution Price Over Time (Top 3 Coins)
st.markdown('<a id="average-execution-price-over-time-top-3-coins"></a><h2>📊 6. Average Execution Price Over Time (Top 3 Coins)</h2>', unsafe_allow_html=True)
st.markdown("Shows how the average execution price changes over time for the most traded coins.")
top_3_coins = top_coins.head(3).index
df_top3 = df[df['Coin'].isin(top_3_coins)]
avg_price = df_top3.groupby(['Timestamp IST', 'Coin'])['Execution Price'].mean().unstack()
fig6, ax6 = plt.subplots(figsize=(12,6))
avg_price.plot(ax=ax6)
ax6.set_title("Average Execution Price Over Time for Top 3 Coins")
ax6.set_xlabel("Timestamp")
ax6.set_ylabel("Average Execution Price")
st.pyplot(fig6)
st.info("Tracking execution price trends helps you spot momentum, mean reversion, or regime changes in the most active coins.")


# 7. Fees Paid by Top Coins
st.markdown('<a id="fees-paid-by-top-coins"></a><h2>💸 7. Fees Paid by Top Coins</h2>', unsafe_allow_html=True)
st.markdown("Total fees paid per coin for the top traded coins.")
fee_sum = df.groupby('Coin')['Fee'].sum().loc[top_coins.index]
fig7, ax7 = plt.subplots(figsize=(10,4))
sns.barplot(x=fee_sum.index, y=fee_sum.values, ax=ax7, palette='magma')
ax7.set_title("Total Fees Paid per Coin")
ax7.set_ylabel("Total Fees")
ax7.set_xlabel("Coin")
st.pyplot(fig7)
st.info("High fees on certain coins may indicate high activity or less efficient trading. Consider fee impact when choosing which coins to trade.")


# 8. Correlation Matrix of Numeric Features
st.markdown('<a id="correlation-matrix-of-numeric-features"></a><h2>🔗 8. Correlation Matrix of Numeric Features</h2>', unsafe_allow_html=True)
st.markdown("Correlation between trade size, price, PnL, and fees.")
num_df = df[num_cols + ['Execution Price']]
corr = num_df.corr()
fig8, ax8 = plt.subplots(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax8)
ax8.set_title("Correlation Heatmap")
st.pyplot(fig8)
st.info("Strong correlations between features can help you build predictive models or spot risk factors. For example, if PnL and trade size are highly correlated, larger trades may be riskier or more profitable.")

# --- Advanced EDA: Relating Performance to Sentiment ---
st.markdown('<a id="trader-performance-by-market-sentiment"></a><h2>📈 9. Trader Performance by Market Sentiment</h2>', unsafe_allow_html=True)
st.markdown("How does trader performance (PnL, win rate, trade size) change with market sentiment?")
sentiment_order = ['Fear', 'Neutral', 'Greed']
sentiment_pnl = df.groupby('sentiment')['Closed PnL'].mean().reindex(sentiment_order)
fig9, ax9 = plt.subplots()
sentiment_pnl.plot(kind='bar', color=['red','gray','green'], ax=ax9)
ax9.set_title('Average Closed PnL by Sentiment')
ax9.set_ylabel('Average Closed PnL')
st.pyplot(fig9)
best_sent = sentiment_pnl.idxmax()
worst_sent = sentiment_pnl.idxmin()
st.info(f"Traders tend to have the highest average PnL during {best_sent} periods and the lowest during {worst_sent} periods. This suggests you may want to be more aggressive or take more risk during {best_sent} and be more cautious during {worst_sent}.")

# Win rate by sentiment
df['win'] = df['Closed PnL'] > 0
sentiment_win = df.groupby('sentiment')['win'].mean().reindex(sentiment_order)
fig10, ax10 = plt.subplots()
sentiment_win.plot(kind='bar', color=['red','gray','green'], ax=ax10)
ax10.set_title('Win Rate by Sentiment')
ax10.set_ylabel('Win Rate')
st.pyplot(fig10)
best_win = sentiment_win.idxmax()
worst_win = sentiment_win.idxmin()
st.info(f"Win rate is highest during {best_win} and lowest during {worst_win}. This means your strategy may be more reliable in {best_win} markets. Consider adjusting your position size or risk tolerance based on the current sentiment.")

# Trade volume by sentiment
sentiment_vol = df.groupby('sentiment')['Size USD'].sum().reindex(sentiment_order)
fig11, ax11 = plt.subplots()
sentiment_vol.plot(kind='bar', color=['red','gray','green'], ax=ax11)
ax11.set_title('Total Trade Volume (USD) by Sentiment')
ax11.set_ylabel('Total Volume (USD)')
st.pyplot(fig11)
max_vol_sent = sentiment_vol.idxmax()
st.info(f"Trade volume is highest during {max_vol_sent} periods. High volume in a sentiment regime may indicate crowd behavior or increased opportunity, but also higher risk of reversals.")

# PnL volatility by sentiment
sentiment_pnl_vol = df.groupby('sentiment')['Closed PnL'].std().reindex(sentiment_order)
fig12, ax12 = plt.subplots()
sentiment_pnl_vol.plot(kind='bar', color=['red','gray','green'], ax=ax12)
ax12.set_title('PnL Volatility by Sentiment')
ax12.set_ylabel('PnL Std Dev')
st.pyplot(fig12)
max_vol = sentiment_pnl_vol.idxmax()
st.info(f"PnL volatility is highest during {max_vol} periods. High volatility means larger swings in profit and loss, so you may want to reduce position size or use tighter stops during these times.")

# Streaks of winning/losing trades
st.markdown('<a id="streaks-of-winning-and-losing-trades"></a><h2>🔄 10. Streaks of Winning and Losing Trades</h2>', unsafe_allow_html=True)
st.markdown("How long do traders stay on a winning or losing streak?")
streaks = []
current = None
for val in df['win']:
    if current is None or val != current['type']:
        if current is not None:
            streaks.append(current)
        current = {'type': val, 'length': 1}
    else:
        current['length'] += 1
if current is not None:
    streaks.append(current)
streaks_df = pd.DataFrame(streaks)
fig13, ax13 = plt.subplots()
streaks_df['length'].hist(bins=30, ax=ax13)
ax13.set_title('Distribution of Win/Loss Streak Lengths')
ax13.set_xlabel('Streak Length')
ax13.set_ylabel('Count')
st.pyplot(fig13)
longest_win = streaks_df[streaks_df['type']==True]['length'].max()
longest_loss = streaks_df[streaks_df['type']==False]['length'].max()
st.info(f"Long winning streaks (max: {longest_win}) and losing streaks (max: {longest_loss}) are rare. After a long streak, traders often experience a reversal. Consider reducing risk or taking profits after a long win streak, and reviewing your strategy after a losing streak.")


# --- Summary Section ---
st.markdown('<a id="summary--key-takeaways"></a><h2>📌 Summary & Key Takeaways</h2>', unsafe_allow_html=True)
st.markdown('''
**Key Insights:**
- Trader performance and behavior shift with market sentiment. Highest PnL and win rates often occur in specific sentiment regimes.
- Trade size, frequency, and volume patterns reveal crowd behavior and market cycles.
- Outliers and streaks can signal regime changes or the need for risk management.
- Correlations between features can inform predictive models and risk controls.

**Actionable Tips:**
- Adjust your trading strategy based on sentiment, trade size, and recent streaks.
- Monitor top coins and fee impact for best opportunities.
- Use EDA findings to refine your risk management and strategy timing.
''')

# Show raw data if needed
if st.checkbox("Show raw data"):
    st.write(df)

