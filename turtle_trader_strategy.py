import numpy as np
import pandas as pd
import yfinance as yf
from backtesting import Backtest, Strategy
import unittest

def calc_TR(high, low, close):
    """Calculate True Range"""
    return np.max(np.abs([high-low, close-low, low-close]))

class TurtleStrategy(Strategy):
    sys1_entry = 20
    sys1_exit = 10
    sys2_entry = 55
    sys2_exit = 20
    atr_periods = 20
    risk_level = 2
    r_max = 0.02
    unit_limit = 5
    
    def init(self):
        self.sys1_long = self.I(lambda: self.data.Close.rolling(self.sys1_entry).max())
        self.sys1_short = self.I(lambda: self.data.Close.rolling(self.sys1_entry).min())
        self.sys1_exit_long = self.I(lambda: self.data.Close.rolling(self.sys1_exit).min())
        self.sys1_exit_short = self.I(lambda: self.data.Close.rolling(self.sys1_exit).max())
        
        self.sys2_long = self.I(lambda: self.data.Close.rolling(self.sys2_entry).max())
        self.sys2_short = self.I(lambda: self.data.Close.rolling(self.sys2_entry).min())
        self.sys2_exit_long = self.I(lambda: self.data.Close.rolling(self.sys2_exit).min())
        self.sys2_exit_short = self.I(lambda: self.data.Close.rolling(self.sys2_exit).max())
        
        self.atr = self.I(lambda: self.data.Close.rolling(self.atr_periods).apply(
            lambda x: calc_TR(self.data.High[-self.atr_periods:], 
                              self.data.Low[-self.atr_periods:], 
                              self.data.Close[-self.atr_periods:])).mean())
        
        self.last_s1_win = False

    def next(self):
        price = self.data.Close[-1]
        
        for trade in self.trades:
            if trade.is_long:
                if price <= trade.sl or price == self.sys1_exit_long[-1] or price == self.sys2_exit_long[-1]:
                    trade.close()
            else:
                if price >= trade.sl or price == self.sys1_exit_short[-1] or price == self.sys2_exit_short[-1]:
                    trade.close()
        
        if not self.position:
            # System 1 entry
            if price == self.sys1_long[-1] and not self.last_s1_win:
                self.buy(size=self._size_position(self.atr[-1]), sl=price - self.risk_level * self.atr[-1])
                self.last_s1_win = False
            elif price == self.sys1_short[-1] and not self.last_s1_win:
                self.sell(size=self._size_position(self.atr[-1]), sl=price + self.risk_level * self.atr[-1])
                self.last_s1_win = False
            
            # System 2 entry
            elif price == self.sys2_long[-1]:
                self.buy(size=self._size_position(self.atr[-1]), sl=price - self.risk_level * self.atr[-1])
            elif price == self.sys2_short[-1]:
                self.sell(size=self._size_position(self.atr[-1]), sl=price + self.risk_level * self.atr[-1])
        
        else:
            # Pyramiding
            for trade in self.trades:
                if len(self.trades) < self.unit_limit:
                    if trade.is_long and price >= trade.entry_price + self.atr[-1]:
                        self.buy(size=self._size_position(self.atr[-1]), sl=price - self.risk_level * self.atr[-1])
                    elif not trade.is_long and price <= trade.entry_price - self.atr[-1]:
                        self.sell(size=self._size_position(self.atr[-1]), sl=price + self.risk_level * self.atr[-1])
    
    def _size_position(self, atr):
        return np.floor(self.r_max * self.equity / (self.risk_level * atr))

def run_backtest(data):
    bt = Backtest(data, TurtleStrategy, cash=10000, commission=.002)
    stats = bt.run()
    return stats

def get_data(tickers, start='2000-01-01', end='2020-12-31'):
    yf_obj = yf.Tickers(tickers)
    data = yf_obj.history(start=start, end=end)
    data.drop(['Open', 'Dividends', 'Stock Splits', 'Volume'], inplace=True, axis=1)
    data.ffill(inplace=True)
    return data

class TestTurtleTrader(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame({
            'Open': [100]*100,
            'High': [110]*100,
            'Low': [90]*100,
            'Close': [105]*100
        })
    
    def test_calc_TR(self):
        self.assertEqual(calc_TR(110, 90, 105), 20)
    
    def test_size_position(self):
        strategy = TurtleStrategy
        strategy.equity = 10000
        self.assertEqual(strategy._size_position(strategy, 10), 10)

    def test_backtest(self):
        stats = run_backtest(self.data)
        self.assertIsNotNone(stats)

if __name__ == "__main__":
    # Run unit tests
    unittest.main(argv=[''], exit=False)
    
    # Run backtest
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    data = get_data(tickers)
    stats = run_backtest(data)
    print(stats)
