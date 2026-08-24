# lob-ml-prediction
# Short-Term Price Movement Prediction Using LOB Microstructure

An end-to-end quantitative finance and machine learning pipeline that streams live Level-2 Limit Order Book (LOB) data to predict short-term directional price movements.

## 📌 Project Overview
This project applies non-linear machine learning models (XGBoost, Random Forest) to high-frequency cryptocurrency data (Kraken XBT/USD). By engineering micro-structural features such as Order Book Imbalance (OBI) and volume-weighted micro-prices, the model predicts whether the mid-price will exceed a rolling-volatility threshold in the near future.

## 🚀 Key Features
* **Live Data Engineering:** Vectorized pipeline connecting to the Kraken REST API.
* **Microstructure Feature Engineering:** Calculates Spread, Level-0 OBI, Weighted Multi-Level OBI.
* **Dynamic Target Labeling:** Uses a rolling 30-period standard deviation for UP/DOWN targets.
* **Institutional Modeling:** Class-weighted XGBoost classifier optimized with TimeSeriesSplit.
* **Vectorized Backtesting:** Accounts for 0.5 bps transaction costs. Evaluates Sharpe, Sortino, Profit Factor.
