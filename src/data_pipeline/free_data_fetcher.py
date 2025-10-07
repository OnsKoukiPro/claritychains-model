import pandas as pd
import numpy as np
import requests
from src.data_pipeline.real_price_fetcher import RealPriceFetcher
from src.data_pipeline.real_trade_fetcher import RealTradeFetcher
import wbdata
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class FreeDataFetcher:
    """Fetch all data using free APIs only"""

    def __init__(self, config):
        self.config = config
        self.price_fetcher = RealPriceFetcher(config)
        self.trade_fetcher = RealTradeFetcher(config)

    def fetch_all_free_data(self):
        """Fetch all data using free APIs"""
        logger.info("Starting free data collection from public APIs")

        # Fetch price data
        prices_df = self.price_fetcher.fetch_all_prices()

        # Fetch trade data (simplified for free API)
        trade_df = self.trade_fetcher.fetch_simplified_trade_flows()

        return prices_df, trade_df

    def get_market_overview(self):
        """Get comprehensive market overview using free data"""
        materials = list(self.config['materials'].keys())

        overview_data = []

        for material in materials:
            logger.info(f"Building market overview for {material}")

            # Get recent prices
            price_data = self.price_fetcher.fetch_all_prices([material])
            current_price = price_data['price'].iloc[-1] if not price_data.empty else None

            # Get major exporters
            exporters = self.trade_fetcher.get_major_exporters(material, year=2023, top_n=5)

            overview = {
                'material': material,
                'current_price': current_price,
                'major_exporters': exporters['reporter'].tolist() if not exporters.empty else [],
                'market_concentration': exporters['market_share'].iloc[0] if not exporters.empty else 0,
                'total_trade_value': exporters['value_usd'].sum() if not exporters.empty else 0
            }

            overview_data.append(overview)
            time.sleep(1)  # Rate limiting

        return pd.DataFrame(overview_data)