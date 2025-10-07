import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RealPriceFetcher:
    """Fetch price data using only Yahoo Finance - no wbdata dependency"""

    def __init__(self, config):
        self.config = config

    def fetch_yahoo_finance_prices(self, material):
        """Fetch prices from Yahoo Finance - most reliable free source"""
        # Map materials to relevant ETFs and commodities
        ticker_map = {
            'lithium': 'LIT',    # Global X Lithium & Battery Tech ETF
            'copper': 'CPER',    # United States Copper Index Fund
            'nickel': 'JJN',     # iPath Bloomberg Nickel Subindex ETN
            'cobalt': 'JJN',     # Using nickel as proxy for now
        }

        if material not in ticker_map:
            return pd.DataFrame()

        try:
            ticker = ticker_map[material]
            stock = yf.Ticker(ticker)

            # Get 2 years of historical data
            hist = stock.history(period="2y")

            if hist.empty:
                logger.warning(f"No Yahoo Finance data for {material} ({ticker})")
                return pd.DataFrame()

            df = hist.reset_index()
            df['material'] = material
            df['price'] = df['Close']
            df['source'] = 'yahoo_finance'

            return df[['Date', 'material', 'price', 'source']].rename(columns={'Date': 'date'})

        except Exception as e:
            logger.error(f"Yahoo Finance error for {material}: {e}")
            return pd.DataFrame()

    def fetch_all_prices(self, materials=None):
        """Fetch prices with fallback to sample data"""
        if materials is None:
            materials = ['lithium', 'copper', 'nickel']

        all_data = []

        for material in materials:
            logger.info(f"Fetching price data for {material}")

            # Try Yahoo Finance
            df = self.fetch_yahoo_finance_prices(material)
            if not df.empty:
                all_data.append(df)
                logger.info(f"✅ Yahoo Finance data for {material}: {len(df)} records")
            else:
                # Use sample data as fallback
                sample_df = self._generate_sample_price_data(material)
                all_data.append(sample_df)
                logger.info(f"📊 Using sample data for {material}: {len(sample_df)} records")

            # Rate limiting
            time.sleep(1)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('date')
            return combined_df
        else:
            logger.warning("No price data found from any source")
            return self._generate_all_sample_price_data()

    def _generate_sample_price_data(self, material):
        """Generate sample price data for one material"""
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='M')

        base_prices = {
            'lithium': 15000, 'cobalt': 35000, 'nickel': 18000,
            'copper': 8000, 'rare_earths': 50000
        }

        base_price = base_prices.get(material, 10000)

        price_data = []
        for i, date in enumerate(dates):
            trend = 1 + (i / len(dates)) * 1.5
            noise = np.random.normal(0, 0.1)
            price = base_price * trend * (1 + noise)

            price_data.append({
                'date': date,
                'material': material,
                'price': max(round(price, 2), 100),
                'source': 'sample_data'
            })

        return pd.DataFrame(price_data)

    def _generate_all_sample_price_data(self):
        """Generate sample data for all materials"""
        materials = ['lithium', 'cobalt', 'nickel', 'copper', 'rare_earths']
        all_data = []

        for material in materials:
            all_data.append(self._generate_sample_price_data(material))

        return pd.concat(all_data, ignore_index=True)