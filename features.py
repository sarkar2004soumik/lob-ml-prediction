import pandas as pd
import numpy as np

def engineer_advanced_features(df, depth=5, forecast_horizon=10):
    df = df.copy()
    
    df['mid_price'] = (df['bid_price_0'] + df['ask_price_0']) / 2
    df['spread'] = df['ask_price_0'] - df['bid_price_0']
    df['spread_roc'] = df['spread'].pct_change()
    
    weighted_bid_vol = sum(df[f'bid_vol_{i}'] / (i + 1) for i in range(depth))
    weighted_ask_vol = sum(df[f'ask_vol_{i}'] / (i + 1) for i in range(depth))
    df['weighted_obi'] = (weighted_bid_vol - weighted_ask_vol) / (weighted_bid_vol + weighted_ask_vol)
    
    df['obi_0'] = (df['bid_vol_0'] - df['ask_vol_0']) / (df['bid_vol_0'] + df['ask_vol_0'])
    df['micro_price'] = (df['bid_price_0'] * df['ask_vol_0'] + df['ask_price_0'] * df['bid_vol_0']) / (df['bid_vol_0'] + df['ask_vol_0'])
    df['micro_mid_diff'] = df['micro_price'] - df['mid_price']
    
    df['ret_1'] = df['mid_price'].pct_change(1)
    df['rolling_volatility_30'] = df['ret_1'].rolling(30).std()
    
    df['future_mid'] = df['mid_price'].shift(-forecast_horizon)
    df['future_return'] = (df['future_mid'] - df['mid_price']) / df['mid_price']
    
    dynamic_threshold = df['rolling_volatility_30'] * 0.5 
    dynamic_threshold.fillna(0.00005, inplace=True)
    
    conditions = [
        (df['future_return'] > dynamic_threshold),
        (df['future_return'] < -dynamic_threshold)
    ]
    df['target'] = np.select(conditions, [1, -1], default=0)
    
    df.dropna(inplace=True)
    return df
