from src.data_collector import LOBCollector
from src.features import engineer_advanced_features
from src.model import train_and_predict
from src.backtest import advanced_backtest

if __name__ == "__main__":
    # 1. Collect Data
    collector = LOBCollector(symbol="XBTUSD", depth=5)
    df_raw = collector.collect_data(samples=500, sleep_time=1)
    
    # 2. Engineer Features
    df_features = engineer_advanced_features(df_raw, forecast_horizon=10)
    
    # 3. Train Model & Predict
    feature_cols = ['spread', 'spread_roc', 'obi_0', 'weighted_obi', 'micro_mid_diff', 'ret_1', 'rolling_volatility_30']
    predictions, price_test = train_and_predict(df_features, feature_cols)
    
    # 4. Run Backtest
    equity_curve, drawdown = advanced_backtest(predictions, price_test, transaction_cost=0.00005)
