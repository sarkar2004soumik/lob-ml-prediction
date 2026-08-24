import requests
import time
import pandas as pd

class LOBCollector:
    def __init__(self, symbol="XBTUSD", depth=5):
        self.symbol = symbol
        self.depth = depth
        self.url = f"https://api.kraken.com/0/public/Depth?pair={self.symbol}&count={self.depth}"
        self.headers = {'User-Agent': 'Mozilla/5.0'}
        
    def collect_data(self, samples=3000, sleep_time=1):
        print(f"Starting data collection for {self.symbol} via Kraken...")
        data = []
        for i in range(samples):
            try:
                response = requests.get(self.url, headers=self.headers).json()
                if response.get('error'):
                    time.sleep(sleep_time)
                    continue
                
                pair_key = list(response['result'].keys())[0]
                book = response['result'][pair_key]
                row = {'timestamp': pd.Timestamp.now()}
                
                for level in range(self.depth):
                    row[f'bid_price_{level}'] = float(book['bids'][level][0])
                    row[f'bid_vol_{level}'] = float(book['bids'][level][1])
                    row[f'ask_price_{level}'] = float(book['asks'][level][0])
                    row[f'ask_vol_{level}'] = float(book['asks'][level][1])
                    
                data.append(row)
                time.sleep(sleep_time)
            except Exception:
                time.sleep(sleep_time)
                
        df = pd.DataFrame(data)
        if not df.empty:
            df.set_index('timestamp', inplace=True)
        return df
