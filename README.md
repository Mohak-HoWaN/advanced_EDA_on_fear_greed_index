# Advanced EDA on Fear & Greed Index

This project provides two interactive Streamlit dashboards designed to analyze market sentiment and trade history data. These dashboards aim to help traders and analysts uncover actionable insights and patterns for smarter decision-making.

---

## Dashboards

### 1. Trade History & Market Sentiment Dashboard
- **File**: `streamlit_hist.py`
- **Purpose**: Analyze trade history and its relationship to market sentiment.
- **Deployed Link**: [Trade History Dashboard](https://tradehistory.streamlit.app)
- **Key Features**:
  - Comprehensive overview of trade data (e.g., total trades, unique accounts, unique coins).
  - Visualizations for:
    - Trade side distribution.
    - Most traded coins.
    - Trade size distribution.
    - Closed PnL analysis.
  - Time-series analysis of trade frequency and execution price trends.
  - Correlation matrix of numeric features.
  - Sentiment-based performance analysis (PnL, win rate, trade volume).
  - Streak analysis for winning and losing trades.
  - Summary section with key insights and actionable tips.

### 2. Fear & Greed Index Dashboard
- **File**: `streamlit_f_g.py`
- **Purpose**: Explore the Fear & Greed Index to understand market sentiment trends.
- **Deployed Link**: [Fear & Greed Index Dashboard](https://fearandgreedindex.streamlit.app)
- **Key Features**:
  - Sentiment counts and time-series analysis.
  - Rolling average and volatility of sentiment.
  - Sentiment transition matrix.
  - Analysis of longest streaks and extreme sentiment periods.
  - Summary section with key insights and actionable tips.

---

## Data Files

- **`historical_data.csv`**: Contains trade history data.
- **`fear_greed_index.csv`**: Contains Fear & Greed Index data.

---

## How to Run

1. **Install Dependencies**:
   Ensure you have Python installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboards**:
   Use the following commands to launch the dashboards:
   ```bash
   streamlit run streamlit_hist.py
   streamlit run streamlit_f_g.py
   ```

---

## Project Structure

```
.
├── streamlit_hist.py         # Trade History & Market Sentiment Dashboard
├── streamlit_f_g.py          # Fear & Greed Index Dashboard
├── historical_data.csv       # Trade history data
├── fear_greed_index.csv      # Fear & Greed Index data
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## Features

- **Interactive Visualizations**: Built using Streamlit for an intuitive user experience.
- **Smooth Navigation**: Clickable sidebar links for seamless exploration.
- **Actionable Insights**: Each section provides key takeaways and tips for traders and analysts.

---

## Tags

- **Languages**: Python
- **Frameworks**: Streamlit
- **Libraries**: Pandas, Seaborn, Matplotlib
- **Use Cases**: Market Sentiment Analysis, Trade Data Analysis, Interactive Dashboards

---

## Customization

This project is designed to be flexible and extensible. Feel free to modify the dashboards to suit your specific needs. The code is well-documented for easy customization.

---

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---

## Acknowledgments

- [Streamlit](https://streamlit.io/) for the interactive dashboard framework.
- [Seaborn](https://seaborn.pydata.org/) and [Matplotlib](https://matplotlib.org/) for data visualizations.
- [Pandas](https://pandas.pydata.org/) for data manipulation.

---

Thank you for exploring this project! If you have any questions or suggestions, feel free to reach out.
