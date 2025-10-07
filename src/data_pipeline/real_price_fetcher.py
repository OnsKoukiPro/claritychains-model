import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader as pdr
from datetime import datetime, timedelta
import time
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class RealPriceFetcher:
    """Fetch price data using reliable Python data libraries"""

    def __init__(self, config):
        self.config = config

    def fetch_yahoo_finance_robust(self, material):
        """Fetch prices from Yahoo Finance with better error handling"""
        # Use more reliable tickers and producers
        ticker_map = {
            'lithium': ['ALB', 'SQM'],      # Major lithium producers
            'copper': ['FCX', 'SCCO'],      # Freeport-McMoRan, Southern Copper
            'nickel': ['VALE', 'BHP'],      # Vale, BHP (major nickel producers)
            'cobalt': ['GLNCY', 'VALE'],    # Glencore, Vale
            'rare_earths': ['MP', 'LYSCF']  # MP Materials, Lynas Rare Earths
        }

        if material not in ticker_map:
            return pd.DataFrame()

        for ticker in ticker_map[material]:
            try:
                stock = yf.Ticker(ticker)
                # Try different periods
                for period in ["1y", "6mo", "3mo"]:
                    try:
                        hist = stock.history(period=period)
                        if not hist.empty and len(hist) > 10:
                            df = hist.reset_index()
                            df['material'] = material
                            df['price'] = df['Close']
                            df['source'] = f'yahoo_{ticker}'
                            return df[['Date', 'material', 'price', 'source']].rename(columns={'Date': 'date'})
                    except:
                        continue
            except Exception as e:
                logger.debug(f"Yahoo Finance failed for {ticker}: {e}")
                continue

        return pd.DataFrame()

    def fetch_pandas_datareader_data(self, material):
        """Fetch data using pandas-datareader from multiple sources"""
        try:
            # FRED (Federal Reserve) economic data - very reliable
            fred_codes = {
                'copper': 'PCOPPUSDM',  # Copper price
                'nickel': 'PNICKUSDM',  # Nickel price
            }

            if material in fred_codes:
                code = fred_codes[material]
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365*3)  # 3 years

                df = pdr.DataReader(code, 'fred', start_date, end_date)
                if not df.empty:
                    df = df.reset_index()
                    df['material'] = material
                    df['price'] = df[code]
                    df['source'] = 'fred'
                    return df[['DATE', 'material', 'price', 'source']].rename(columns={'DATE': 'date'})

        except Exception as e:
            logger.debug(f"pandas-datareader failed for {material}: {e}")

        return pd.DataFrame()

    def fetch_commodity_futures(self, material):
        """Fetch commodity futures data"""
        futures_map = {
            'copper': 'HG=F',    # Copper Futures
            'nickel': 'NIK=F',   # Nickel Futures (if available)
        }

        if material in futures_map:
            try:
                ticker = futures_map[material]
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")

                if not hist.empty:
                    df = hist.reset_index()
                    df['material'] = material
                    df['price'] = df['Close']
                    df['source'] = f'futures_{ticker}'
                    return df[['Date', 'material', 'price', 'source']].rename(columns={'Date': 'date'})
            except Exception as e:
                logger.debug(f"Futures data failed for {material}: {e}")

        return pd.DataFrame()

    def fetch_etf_data(self, material):
        """Fetch ETF data for materials"""
        etf_map = {
            'lithium': 'LIT',    # Global X Lithium & Battery Tech ETF
            'copper': 'CPER',    # United States Copper Index Fund
            'rare_earths': 'REMX',  # VanEck Rare Earth/Strategic Metals ETF
        }

        if material in etf_map:
            try:
                ticker = etf_map[material]
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1y")

                if not hist.empty:
                    df = hist.reset_index()
                    df['material'] = material
                    df['price'] = df['Close']
                    df['source'] = f'etf_{ticker}'
                    return df[['Date', 'material', 'price', 'source']].rename(columns={'Date': 'date'})
            except Exception as e:
                logger.debug(f"ETF data failed for {material}: {e}")

        return pd.DataFrame()

    def fetch_all_prices(self, materials=None):
        """Fetch prices using multiple Python libraries"""
        if materials is None:
            materials = ['lithium', 'copper', 'nickel', 'cobalt', 'rare_earths']

        all_data = []

        for material in materials:
            logger.info(f"Fetching REAL price data for {material} using Python libraries")

            df = pd.DataFrame()

            # Try multiple data sources in order of reliability
            sources_to_try = [
                self.fetch_yahoo_finance_robust,
                self.fetch_pandas_datareader_data,
                self.fetch_commodity_futures,
                self.fetch_etf_data,
            ]

            for source_func in sources_to_try:
                if df.empty:
                    df = source_func(material)
                    if not df.empty:
                        logger.info(f"✅ Got {material} data from {source_func.__name__}: {len(df)} records")
                        break

            if not df.empty:
                all_data.append(df)
            else:
                logger.warning(f"❌ No real data found for {material} from any Python library")

            # Rate limiting
            time.sleep(2)

        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.sort_values('date')
            logger.info(f"✅ Total real price records: {len(combined_df)}")
            return combined_df
        else:
            logger.error("❌ No price data found from any Python library")
            return pd.DataFrame()