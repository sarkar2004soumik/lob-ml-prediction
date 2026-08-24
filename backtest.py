import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def advanced_backtest(predictions, prices, transaction_cost=0.00005):
    signals = pd.Series(predictions, index=prices.index)
    asset_returns = prices.pct_change().shift(-1).fillna(0)
    
    strategy_returns = signals * asset_returns
    position_changes = signals.diff().abs().fillna(0)
    net_strategy_returns = strategy_returns - (position_changes * transaction_cost)
    
    cumulative_returns = (1 + net_strategy_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    
    winning_trades = net_strategy_returns[net_strategy_returns > 0]
    losing_trades = net_strategy_returns[net_strategy_returns < 0]
    
    win_rate = len(winning_trades) / len(net_strategy_returns[net_strategy_returns != 0]) if len(net_strategy_returns[net_strategy_returns != 0]) > 0 else 0
    profit_factor = abs(winning_trades.sum() / losing_trades.sum()) if losing_trades.sum() != 0 else np.inf
    
    time_scalar = np.sqrt(3600) 
    sharpe_ratio = (net_strategy_returns.mean() / net_strategy_returns.std()) * time_scalar if net_strategy_returns.std() > 0 else 0
    downside_std = losing_trades.std()
    sortino_ratio = (net_strategy_returns.mean() / downside_std) * time_scalar if downside_std > 0 else 0
    
    print("=== Strategy Performance Report ===")
    print(f"Total Return:      {(cumulative_returns.iloc[-1] - 1):.4%}")
    print(f"Win Rate:          {win_rate:.2%}")
    print(f"Profit Factor:     {profit_factor:.2f}")
    print(f"Hourly Sharpe:     {sharpe_ratio:.3f}")
    print(f"Hourly Sortino:    {sortino_ratio:.3f}")
    print(f"Max Drawdown:      {drawdown.min():.2%}")
    
    return cumulative_returns, drawdown
