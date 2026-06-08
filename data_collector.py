import pandas as pd
import numpy as np
import requests
from datetime import datetime
from pycoingecko import CoinGeckoAPI

class MultiModalDataCollector:
    def __init__(self):
        self.cg = CoinGeckoAPI()

    def collect_price_data(self, crypto_id, days=365):
        print(f"Collecting price data for {crypto_id}...")
        try:
            data = self.cg.get_coin_market_chart_by_id(
                id=crypto_id.lower(),
                vs_currency='usd',
                days=days
            )
            df = pd.DataFrame({
                'timestamp': [x[0] for x in data['prices']],
                'close': [x[1] for x in data['prices']],
                'volume': [x[1] for x in data['total_volumes']],
                'market_cap': [x[1] for x in data['market_caps']]
            })
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            # Resample to daily to reduce data size
            df = df.resample('D').agg({
                'close': 'last',
                'volume': 'sum',
                'market_cap': 'last'
            })
            
            # Calculate OHLC (approximate)
            df['high'] = df['close'] * 1.01
            df['low'] = df['close'] * 0.99
            df['open'] = df['close'].shift(1)
            df = self._add_technical_indicators(df)
            print(f"✓ Collected {len(df)} price records for {crypto_id}")
            return df
        except Exception as e:
            print(f"Error collecting price data for {crypto_id}: {e}")
            return pd.DataFrame()

    def _add_technical_indicators(self, df):
        df['ma_7'] = df['close'].rolling(window=7).mean()
        df['ma_21'] = df['close'].rolling(window=21).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['std_21'] = df['close'].rolling(window=21).std()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['return_1d'] = df['close'].pct_change(1)
        df['return_7d'] = df['close'].pct_change(7)
        df['vol_ma_7'] = df['volume'].rolling(window=7).mean()
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        return df

    def collect_sentiment_data(self, crypto_symbol, days=7):
        print(f"Generating sentiment data for {crypto_symbol}...")
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        sentiment_scores = np.random.randn(len(dates)) * 0.3
        tweet_volumes = np.random.poisson(lam=100, size=len(dates))
        df = pd.DataFrame({
            'timestamp': dates,
            'sentiment_score': sentiment_scores,
            'tweet_volume': tweet_volumes,
            'positive_ratio': np.clip(0.5 + sentiment_scores * 0.2, 0, 1),
            'negative_ratio': np.clip(0.5 - sentiment_scores * 0.2, 0, 1)
        })
        df = df.set_index('timestamp')
        print(f"✓ Generated {len(df)} sentiment records for {crypto_symbol}")
        return df

    def collect_onchain_data(self, crypto_id, days=365):
        print(f"Generating on-chain data for {crypto_id}...")
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': dates,
            'active_addresses': np.random.poisson(lam=50000, size=len(dates)),
            'transaction_count': np.random.poisson(lam=300000, size=len(dates)),
            'hash_rate': np.random.normal(loc=150, scale=20, size=len(dates)),
            'nvt_ratio': np.random.normal(loc=50, scale=10, size=len(dates)),
            'mvrv_ratio': np.random.normal(loc=1.5, scale=0.3, size=len(dates))
        })
        df = df.set_index('timestamp')
        print(f"✓ Generated {len(df)} on-chain records for {crypto_id}")
        return df

    def collect_macro_data(self, days=365):
        print(f"Generating macroeconomic data...")
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        np.random.seed(42)
        df = pd.DataFrame({
            'timestamp': dates,
            'sp500': 4000 + np.random.randn(len(dates)).cumsum() * 10,
            'gold_price': 1800 + np.random.randn(len(dates)).cumsum() * 5,
            'oil_price': 80 + np.random.randn(len(dates)).cumsum() * 2,
            'dxy': 100 + np.random.randn(len(dates)).cumsum() * 0.5,
            'vix': 20 + np.abs(np.random.randn(len(dates))) * 2
        })
        df = df.set_index('timestamp')
        df['sp500_return'] = df['sp500'].pct_change()
        df['gold_return'] = df['gold_price'].pct_change()
        print(f"✓ Generated {len(df)} macroeconomic records")
        return df

    def merge_all_data(self, price_df, sentiment_df, onchain_df, macro_df):
        print("Merging all data sources...")
        
        # Start with price_df as base - THIS IS CRITICAL!
        merged = price_df.copy()
        
        print(f"Base price_df shape: {merged.shape}")
        print(f"Price range BEFORE merge: ${merged['close'].min():.2f} - ${merged['close'].max():.2f}")
        
        # Join other data with suffixes to avoid column conflicts
        merged = merged.join(sentiment_df, how='left', rsuffix='_sent')
        merged = merged.join(onchain_df, how='left', rsuffix='_onchain')
        merged = merged.join(macro_df, how='left', rsuffix='_macro')
        
        print(f"After joins - shape: {merged.shape}")
        print(f"Price range AFTER joins: ${merged['close'].min():.2f} - ${merged['close'].max():.2f}")
        
        # Fill NaN but preserve price data
        # Forward fill first, then backward fill, then zero
        merged = merged.ffill().bfill().fillna(0)
        
        print(f"After fillna - shape: {merged.shape}")
        print(f"Price range AFTER fillna: ${merged['close'].min():.2f} - ${merged['close'].max():.2f}")
        print(f"✓ Merged data: {len(merged)} records with {len(merged.columns)} features")
        
        return merged

if __name__ == "__main__":
    import os
    collector = MultiModalDataCollector()
    price_df = collector.collect_price_data('bitcoin', days=365)
    sentiment_df = collector.collect_sentiment_data('BTC', days=365)
    onchain_df = collector.collect_onchain_data('bitcoin', days=365)
    macro_df = collector.collect_macro_data(days=365)
    merged_df = collector.merge_all_data(price_df, sentiment_df, onchain_df, macro_df)
    os.makedirs('data', exist_ok=True)
    merged_df.to_csv('data/bitcoin_multimodal_data.csv')
    print(f"\n✓ Data saved to data/bitcoin_multimodal_data.csv")
    print(f"Shape: {merged_df.shape}")
    print(f"\nColumns: {list(merged_df.columns)}")
