# app/utils/data_loader.py
import pandas as pd
from pathlib import Path
import streamlit as st
import sys
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Import handling with fallbacks
def safe_import(module_name, class_name):
    """Safely import classes with detailed error reporting"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(f"Import error for {class_name} from {module_name}: {e}")
        return None

# Try imports with fallbacks
try:
    from src.data_pipeline.global_price_fetcher import GlobalCommodityFetcher
    logger.info("✅ Successfully imported GlobalCommodityFetcher")
except ImportError as e:
    logger.warning(f"GlobalCommodityFetcher import failed: {e}")
    GlobalCommodityFetcher = None

try:
    from src.data_pipeline.real_trade_fetcher import RealTradeFetcher
except ImportError as e:
    logger.warning(f"RealTradeFetcher import failed: {e}")
    RealTradeFetcher = None

# Fallback classes
class FallbackBaselineForecaster:
    def __init__(self, config):
        self.config = config
        logger.warning("Using fallback BaselineForecaster - limited functionality")

    def fit_predict(self, prices_df, material, use_fundamentals=False):
        logger.error("BaselineForecaster not properly imported - using dummy data")
        return {
            'historical': prices_df.tail(10),
            'forecast': pd.DataFrame({
                'date': pd.date_range(start=datetime.now(), periods=6, freq='M'),
                'forecast_mean': [prices_df['price'].iloc[-1]] * 6,
                'forecast_p10': [prices_df['price'].iloc[-1] * 0.9] * 6,
                'forecast_p90': [prices_df['price'].iloc[-1] * 1.1] * 6
            }),
            'metrics': {'current_price': prices_df['price'].iloc[-1] if not prices_df.empty else 0},
            'model_info': {'model_type': 'fallback'},
            'fundamentals': None
        }

def fetch_real_data():
    """Fetch real data using reliable Python libraries"""
    from app.utils.config import load_config
    config = load_config()

    # Check if fetchers are available
    if GlobalCommodityFetcher is None or RealTradeFetcher is None:
        st.error("❌ Data fetchers not available. Please check the installation.")
        return pd.DataFrame(), pd.DataFrame(), "error"

    # Initialize data fetchers
    price_fetcher = GlobalCommodityFetcher(config)
    trade_fetcher = RealTradeFetcher(config)

    # Create data directory
    data_dir = Path(config['paths']['raw_data'])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Fetch price data
    st.info("🌍 Fetching price data from global sources (FRED, ECB, World Bank, LME)...")
    try:
        prices_df = price_fetcher.fetch_all_prices()

        if not prices_df.empty:
            prices_df.to_csv(data_dir / "real_prices.csv", index=False)
            if 'source' in prices_df.columns:
                source_counts = prices_df['source'].value_counts()
                source_info = ", ".join([f"{k} ({v} recs)" for k, v in source_counts.items()])
                st.success(f"✅ Loaded {len(prices_df)} price records from: {source_info}")
            else:
                st.success(f"✅ Loaded {len(prices_df)} price records")
        else:
            st.error("❌ Could not load any price data")
            return pd.DataFrame(), pd.DataFrame(), "error"
    except Exception as e:
        st.error(f"❌ Price data fetch failed: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame(), "error"

    # Fetch trade data
    st.info("🌍 Fetching trade data from USGS and World Bank...")
    try:
        trade_df = trade_fetcher.fetch_simplified_trade_flows(years=[2025])

        if not trade_df.empty:
            trade_df.to_csv(data_dir / "real_trade_flows.csv", index=False)
            source_info = ", ".join(trade_df['source'].unique()) if 'source' in trade_df.columns else "Unknown"
            st.success(f"✅ Loaded {len(trade_df)} trade records from: {source_info}")
        else:
            st.error("❌ Could not load any trade data")
            return pd.DataFrame(), pd.DataFrame(), "error"
    except Exception as e:
        st.error(f"❌ Trade data fetch failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), "error"

    return prices_df, trade_df, "real"

def load_data():
    """Load data from files or fetch real data"""
    from app.utils.config import load_config
    config = load_config()
    data_dir = Path(config['paths']['raw_data'])

    # Check if real data exists
    real_prices_path = data_dir / "real_prices.csv"
    real_trade_path = data_dir / "real_trade_flows.csv"

    if real_prices_path.exists() and real_trade_path.exists():
        try:
            prices_df = pd.read_csv(real_prices_path)
            if 'date' in prices_df.columns:
                prices_df['date'] = pd.to_datetime(prices_df['date'])
            trade_df = pd.read_csv(real_trade_path)
            return prices_df, trade_df, "real"
        except Exception as e:
            st.warning(f"Error loading saved data: {e}")

    # Only fetch real data - no sample fallback
    return fetch_real_data()

# Add these fallback classes to app/utils/data_loader.py

class EVAdoptionFetcher:
    """Fallback EV Adoption Fetcher"""
    def __init__(self, config):
        self.config = config
        self.material_intensity = {
            'lithium': {'per_ev_kg': 8.0, 'growth_factor': 1.15},
            'cobalt': {'per_ev_kg': 12.0, 'growth_factor': 1.10},
            'nickel': {'per_ev_kg': 40.0, 'growth_factor': 1.12},
            'copper': {'per_ev_kg': 80.0, 'growth_factor': 1.08},
            'rare_earths': {'per_ev_kg': 1.0, 'growth_factor': 1.18}
        }

    def calculate_material_demand(self, material: str, scenario: str = 'stated_policies') -> pd.DataFrame:
        """Fixed version that handles key errors"""
        try:
            if material not in self.material_intensity:
                return pd.DataFrame()

            # Use safe scenario data access
            scenario_data = getattr(self, 'scenarios', {}).get(scenario, {})

            # Use safe key access with defaults
            start_sales = scenario_data.get('sales_2024', scenario_data.get('ev_sales_2024', 14.0)) * 1e6
            end_sales = scenario_data.get('sales_2030', scenario_data.get('ev_sales_2030', 60.9)) * 1e6
            annual_growth = scenario_data.get('annual_growth', 0.28)

            years = list(range(2024, 2031))
            demand_data = []
            intensity = self.material_intensity[material]

            for i, year in enumerate(years):
                if year == 2024:
                    ev_sales = start_sales
                else:
                    growth_years = year - 2024
                    ev_sales = start_sales * ((1 + annual_growth) ** growth_years)

                base_demand = ev_sales * intensity['per_ev_kg'] / 1000
                years_from_2024 = max(0, year - 2024)
                growth_multiplier = intensity['growth_factor'] ** years_from_2024
                total_demand = base_demand * growth_multiplier

                demand_data.append({
                    'year': year,
                    'material': material,
                    'ev_sales_millions': ev_sales / 1e6,
                    'material_demand_tons': total_demand,
                    'scenario': scenario
                })

            return pd.DataFrame(demand_data)

        except Exception as e:
            logger.error(f"Error in calculate_material_demand: {e}")
            return pd.DataFrame()

    def get_demand_forecast_adjustment(self, material):
        return {'adjustment_factor': 1.0, 'demand_growth': 0.0}

class GDELTFetcher:
    """Fallback GDELT Fetcher"""
    def __init__(self, config):
        self.config = config

    def fetch_events_for_material(self, material, days_back=30):
        return pd.DataFrame()

    def generate_risk_score(self, events, material, country=None):
        return {
            'risk_score': 0.0,
            'risk_level': 'LOW',
            'recent_events': 0,
            'avg_sentiment': 0.0,
            'risk_description': 'No risk data available'
        }