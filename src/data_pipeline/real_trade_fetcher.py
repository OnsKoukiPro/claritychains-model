import pandas as pd
import requests
from datetime import datetime
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class RealTradeFetcher:
    """Fetch trade flow data from free UN Comtrade API"""

    def __init__(self, config):
        self.config = config
        self.base_url = "https://comtrade.un.org/api/get"

    def fetch_comtrade_data(self, material, year=2023, reporter="all", partner="all", max_records=10000):
        """Fetch trade data from free UN Comtrade API"""
        try:
            commodity_codes = self.config['materials'][material]['comtrade_codes']

            all_trade_data = []

            for code in commodity_codes:
                logger.info(f"Fetching Comtrade data for {material} (HS{code}) - {year}")

                params = {
                    'type': 'C',
                    'freq': 'A',
                    'px': 'HS',
                    'ps': year,  # Period (year)
                    'r': reporter,  # Reporter country (all for all countries)
                    'p': partner,   # Partner country
                    'rg': 'all',    # Trade regime (all = both import and export)
                    'cc': code,     # Commodity code
                    'fmt': 'json',  # Format
                    'max': max_records  # Maximum records to return
                }

                response = requests.get(self.base_url, params=params, timeout=30)

                if response.status_code == 200:
                    data = response.json()

                    if 'dataset' in data and data['dataset']:
                        df = pd.DataFrame(data['dataset'])

                        # Extract relevant information
                        processed_data = self._process_comtrade_data(df, material, code)
                        all_trade_data.append(processed_data)
                        logger.info(f"✅ Found {len(processed_data)} records for HS{code}")
                    else:
                        logger.warning(f"No data found for {material} (HS{code}) in {year}")
                else:
                    logger.error(f"API error {response.status_code} for {material} (HS{code})")

                # Be respectful to the free API - add delay between requests
                time.sleep(2)

            if all_trade_data:
                combined_df = pd.concat(all_trade_data, ignore_index=True)
                logger.info(f"✅ Total records for {material}: {len(combined_df)}")
                return combined_df
            else:
                logger.warning(f"No trade data found for {material}")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"Comtrade API error for {material}: {e}")
            return pd.DataFrame()

    def _process_comtrade_data(self, df, material, hs_code):
        """Process and clean Comtrade data"""
        try:
            # Select and rename columns - use 'exporter' for consistency
            column_mapping = {
                'yr': 'year',
                'rtTitle': 'exporter',  # Changed from 'reporter' to 'exporter'
                'ptTitle': 'partner',
                'cmdDescE': 'commodity_description',
                'rgDesc': 'trade_flow',
                'TradeValue': 'value_usd',
                'qtDesc': 'quantity_unit',
                'NetWeight': 'weight_kg'
            }

            # Only keep columns that exist in the data
            available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            processed_df = df[list(available_columns.keys())].copy()
            processed_df.rename(columns=available_columns, inplace=True)

            # Add material and HS code information
            processed_df['material'] = material
            processed_df['hs_code'] = hs_code
            processed_df['source'] = 'comtrade'

            # Convert numeric columns
            if 'value_usd' in processed_df.columns:
                processed_df['value_usd'] = pd.to_numeric(processed_df['value_usd'], errors='coerce')

            if 'weight_kg' in processed_df.columns:
                processed_df['weight_kg'] = pd.to_numeric(processed_df['weight_kg'], errors='coerce')

            # Filter out invalid records
            processed_df = processed_df.dropna(subset=['value_usd'])
            processed_df = processed_df[processed_df['value_usd'] > 0]

            return processed_df

        except Exception as e:
            logger.error(f"Error processing Comtrade data: {e}")
            return pd.DataFrame()

    def fetch_simplified_trade_flows(self, materials=None, years=None):
        """Fetch simplified trade flows focusing on major exporters"""
        if materials is None:
            materials = list(self.hs_codes.keys())

        if years is None:
            years = [2023]

        all_trade_data = []

        for material in materials:
            for year in years:
                logger.info(f"Fetching {material} trade data for {year}")

                trade_data = self.fetch_comtrade_data(material, year=year, max_records=5000)

                if not trade_data.empty:
                    # Focus on export data
                    export_data = trade_data[trade_data['trade_flow'].str.contains('Export', na=False)]

                    if not export_data.empty:
                        # Aggregate by exporter (was reporter)
                        exporter_summary = (export_data.groupby(['exporter', 'material', 'year'])
                                        .agg({'value_usd': 'sum'})
                                        .reset_index()
                                        .sort_values('value_usd', ascending=False))

                        all_trade_data.append(exporter_summary)
                        logger.info(f"✅ Found {len(export_data)} export records for {material}")
                    else:
                        logger.warning(f"No export data found for {material} in {year}")
                else:
                    logger.warning(f"No trade data found for {material} in {year}")

                # Rate limiting
                time.sleep(4)

        if all_trade_data:
            combined_df = pd.concat(all_trade_data, ignore_index=True)
            logger.info(f"✅ Fetched {len(combined_df)} total trade records")
            return combined_df
        else:
            logger.warning("No trade data found for any materials")
            return pd.DataFrame()

    def get_major_exporters(self, material, year=2023, top_n=10):
        """Get top exporters for a specific material"""
        trade_data = self.fetch_comtrade_data(material, year=year)

        if trade_data.empty:
            return pd.DataFrame()

        # Filter export flows and aggregate
        exports = trade_data[trade_data['trade_flow'].str.contains('Export', na=False)]

        if exports.empty:
            return pd.DataFrame()

        exporter_summary = (exports.groupby('reporter')
                          .agg({'value_usd': 'sum', 'weight_kg': 'sum'})
                          .reset_index()
                          .sort_values('value_usd', ascending=False)
                          .head(top_n))

        exporter_summary['market_share'] = exporter_summary['value_usd'] / exporter_summary['value_usd'].sum()
        exporter_summary['material'] = material
        exporter_summary['year'] = year

        return exporter_summary

    def fetch_all_trade_flows(self, years=None):
        """Wrapper method for compatibility with Streamlit app"""
        return self.fetch_simplified_trade_flows(years=years)