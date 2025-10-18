#!/usr/bin/env python3
"""
Data fetching script for Docker deployment
"""
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

try:
    from data_pipeline.global_price_fetcher import GlobalCommodityFetcher
    from data_pipeline.real_trade_fetcher import RealTradeFetcher
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_all_data():
    """Fetch all data sources"""
    logger.info("Starting data update...")

    try:
        # Simple config for the fetchers
        config = {
            'materials': {
                'lithium': {'comtrade_codes': ['283691']},
                'cobalt': {'comtrade_codes': ['260500', '810520']},
                'nickel': {'comtrade_codes': ['750100', '750210']},
                'copper': {'comtrade_codes': ['740311', '740319']},
                'rare_earths': {'comtrade_codes': ['280530']}
            },
            'paths': {
                'raw_data': './data/raw'
            }
        }

        # Fetch price data
        logger.info("Fetching price data...")
        price_fetcher = GlobalCommodityFetcher(config)
        prices_df = price_fetcher.fetch_all_prices()

        if not prices_df.empty:
            Path('./data/raw').mkdir(parents=True, exist_ok=True)
            prices_df.to_csv('./data/raw/prices.csv', index=False)
            logger.info(f"✅ Saved {len(prices_df)} price records")

        # Fetch trade flows
        logger.info("Fetching trade flow data...")
        trade_fetcher = RealTradeFetcher(config)
        trade_df = trade_fetcher.fetch_all_trade_flows()

        if not trade_df.empty:
            Path('./data/raw').mkdir(parents=True, exist_ok=True)
            trade_df.to_csv('./data/raw/trade_flows.csv', index=False)
            logger.info(f"✅ Saved {len(trade_df)} trade records")

        logger.info("Data update complete!")

    except Exception as e:
        logger.error(f"Data update failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    fetch_all_data()