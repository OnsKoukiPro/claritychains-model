import json
from logging import config
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import logging

# Add src to Python path - FIXED PATH ISSUE
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))
else:
    # Try alternative path structure
    src_path = current_dir.parent / 'src'
    if src_path.exists():
        sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Health check
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Import handling with better error reporting
def safe_import(module_name, class_name):
    """Safely import classes with detailed error reporting"""
    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        logger.warning(f"Import error for {class_name} from {module_name}: {e}")
        return None

# Try imports with fallbacks - UPDATED TO USE GlobalCommodityFetcher
try:
    from data_pipeline.global_price_fetcher import GlobalCommodityFetcher
    logger.info("✅ Successfully imported GlobalCommodityFetcher")
except ImportError as e:
    logger.warning(f"GlobalCommodityFetcher import failed: {e}")
    try:
        from data_pipeline.real_price_fetcher import RealPriceFetcher
        GlobalCommodityFetcher = RealPriceFetcher
        logger.info("✅ Using RealPriceFetcher as fallback")
    except ImportError:
        logger.error("Both price fetchers failed to import")
        GlobalCommodityFetcher = None

try:
    from data_pipeline.real_trade_fetcher import RealTradeFetcher
except ImportError as e:
    logger.warning(f"RealTradeFetcher import failed: {e}")
    RealTradeFetcher = None

try:
    from models.baseline_forecaster import BaselineForecaster
except ImportError as e:
    logger.error(f"BaselineForecaster import failed: {e}")
    BaselineForecaster = None
    class BaselineForecaster:
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
        def compare_forecast_methods(self, prices_df, material):
            return {
                'material': material,
                'baseline': {'mean_forecast': 0},
                'enhanced': {'mean_forecast': 0},
                'difference': {'mean_change_pct': 0}
            }

try:
    from data_pipeline.ev_adoption_fetcher import EVAdoptionFetcher
except ImportError as e:
    logger.warning(f"EVAdoptionFetcher import failed: {e}")
    class EVAdoptionFetcher:
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
            try:
                if material not in self.material_intensity:
                    return pd.DataFrame()
                scenario_data = getattr(self, 'scenarios', {}).get(scenario, {})
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

try:
    from data_pipeline.gdelt_fetcher import GDELTFetcher
except ImportError as e:
    logger.warning(f"GDELTFetcher import failed: {e}")
    class GDELTFetcher:
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

def load_config():
    """Load configuration with enhanced error handling"""
    try:
        config_path = Path(__file__).parent / 'config' / 'config.yaml'
        if not config_path.exists():
            config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                logger.info("Configuration loaded successfully")
                return config
        else:
            logger.warning("Config file not found, using defaults")
    except Exception as e:
        logger.error(f"Config load error: {e}")
    return {
        'paths': {
            'data_dir': './data',
            'raw_data': './data/raw',
            'processed_data': './data/processed'
        },
        'materials': {
            'lithium': {}, 'cobalt': {}, 'nickel': {}, 'copper': {}, 'rare_earths': {},
            'aluminum': {}, 'zinc': {}, 'lead': {}, 'tin': {}
        },
        'forecasting': {
            'use_fundamentals': True,
            'ev_adjustment_weight': 0.3,
            'risk_adjustment_weight': 0.2,
            'rolling_window': 12,
            'forecast_horizon': 6,
            'confidence_levels': [0.1, 0.5, 0.9]
        },
        'gdelt': {
            'base_url': "https://api.gdeltproject.org/api/v2/doc/doc",
            'max_records': 250,
            'rate_limit_delay': 0.5
        },
        'ev_adoption': {
            'material_intensity': {
                'lithium': {'per_ev_kg': 8.0, 'growth_factor': 1.15},
                'cobalt': {'per_ev_kg': 12.0, 'growth_factor': 1.10},
                'nickel': {'per_ev_kg': 40.0, 'growth_factor': 1.12},
                'copper': {'per_ev_kg': 80.0, 'growth_factor': 1.08},
                'rare_earths': {'per_ev_kg': 1.0, 'growth_factor': 1.18},
                'aluminum': {'per_ev_kg': 180.0, 'growth_factor': 1.05}
            },
            'price_elasticity': 0.3
        },
        'global_sources': {
            'ecb_enabled': True,
            'worldbank_enabled': True,
            'lme_enabled': True,
            'global_futures_enabled': True
        }
    }

def fetch_real_data():
    """Fetch real data using reliable Python libraries"""
    config = load_config()
    if GlobalCommodityFetcher is None or RealTradeFetcher is None:
        st.error("❌ Data fetchers not available. Please check the installation.")
        return pd.DataFrame(), pd.DataFrame(), "error"
    price_fetcher = GlobalCommodityFetcher(config)
    trade_fetcher = RealTradeFetcher(config)
    data_dir = Path(config['paths']['raw_data'])
    data_dir.mkdir(parents=True, exist_ok=True)
    try:
        lib_success, lib_message = trade_fetcher.test_data_availability()
        if not lib_success:
            logger.warning(f"Data availability test failed: {lib_message}")
    except Exception as e:
        logger.warning(f"Data availability test failed: {e}")
    try:
        prices_df = price_fetcher.fetch_all_prices()
        if not prices_df.empty:
            prices_df.to_csv(data_dir / "real_prices.csv", index=False)
            logger.info(f"Loaded {len(prices_df)} price records")
        else:
            logger.error("Could not load any price data")
            return pd.DataFrame(), pd.DataFrame(), "error"
    except Exception as e:
        logger.error(f"Price data fetch failed: {e}")
        import traceback
        logger.error(f"Detailed error: {traceback.format_exc()}")
        return pd.DataFrame(), pd.DataFrame(), "error"
    try:
        trade_df = trade_fetcher.fetch_simplified_trade_flows(years=[2025])
        if not trade_df.empty:
            trade_df.to_csv(data_dir / "real_trade_flows.csv", index=False)
            logger.info(f"Loaded {len(trade_df)} trade records")
        else:
            logger.error("Could not load any trade data")
            return pd.DataFrame(), pd.DataFrame(), "error"
    except Exception as e:
        logger.error(f"Trade data fetch failed: {e}")
        return pd.DataFrame(), pd.DataFrame(), "error"
    return prices_df, trade_df, "real"

def load_data():
    """Load data from files or fetch real data"""
    config = load_config()
    data_dir = Path(config['paths']['raw_data'])
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
            logger.warning(f"Error loading saved data: {e}")
    return fetch_real_data()

# ============================================================================
# NOTION-STYLE UI COMPONENTS
# ============================================================================

def render_notion_header():
    """Render Notion-style header"""
    st.markdown("""
    <style>
    /* Notion-inspired Design System */
    :root {
        --notion-bg: #ffffff;
        --notion-sidebar: #f7f6f3;
        --notion-text: #37352f;
        --notion-text-light: #787774;
        --notion-border: #e9e9e7;
        --notion-hover: #f1f1ef;
        --notion-blue: #2383e2;
        --notion-blue-bg: #e9f2fa;
        --notion-green: #0f7b6c;
        --notion-green-bg: #ddedea;
        --notion-yellow: #cb6200;
        --notion-yellow-bg: #fdecc8;
        --notion-red: #eb5757;
        --notion-red-bg: #fde2e4;
        --notion-purple: #9d34da;
        --notion-shadow: rgba(15, 15, 15, 0.1);
    }

    /* Remove default Streamlit padding */
    .main .block-container {
        padding-top: 0rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }

    /* Notion-style header */
    .notion-header {
        background: var(--notion-bg);
        border-bottom: 1px solid var(--notion-border);
        padding: 1.5rem 0;
        margin-bottom: 2rem;
    }

    .notion-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--notion-text);
        margin: 0;
        letter-spacing: -0.03em;
    }

    .notion-subtitle {
        font-size: 1rem;
        color: var(--notion-text-light);
        margin-top: 0.5rem;
    }

    /* Notion-style blocks */
    .notion-block {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        border-radius: 6px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s ease;
    }

    .notion-block:hover {
        box-shadow: 0 2px 8px var(--notion-shadow);
    }

    .notion-block-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .notion-block-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: var(--notion-text);
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .notion-block-actions {
        display: flex;
        gap: 0.5rem;
    }

    /* Inline editable text */
    .notion-editable {
        border: none;
        border-bottom: 1px solid transparent;
        background: transparent;
        padding: 2px 0;
        transition: all 0.2s ease;
    }

    .notion-editable:hover {
        border-bottom-color: var(--notion-border);
        background: var(--notion-hover);
    }

    .notion-editable:focus {
        outline: none;
        border-bottom-color: var(--notion-blue);
        background: var(--notion-blue-bg);
    }

    /* Status badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.25rem;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.875rem;
        font-weight: 500;
    }

    .status-draft { background: var(--notion-yellow-bg); color: var(--notion-yellow); }
    .status-active { background: var(--notion-blue-bg); color: var(--notion-blue); }
    .status-completed { background: var(--notion-green-bg); color: var(--notion-green); }
    .status-archived { background: var(--notion-hover); color: var(--notion-text-light); }

    /* Chat interface */
    .chat-container {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        border-radius: 8px;
        padding: 1rem;
        min-height: 400px;
        max-height: 600px;
        overflow-y: auto;
    }

    .chat-message {
        margin-bottom: 1rem;
        padding: 0.75rem 1rem;
        border-radius: 6px;
    }

    .chat-user {
        background: var(--notion-blue-bg);
        color: var(--notion-text);
        margin-left: 2rem;
    }

    .chat-assistant {
        background: var(--notion-hover);
        color: var(--notion-text);
        margin-right: 2rem;
    }

    /* Table view (Airtable-style) */
    .airtable-view {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        border-radius: 8px;
        overflow: hidden;
    }

    .airtable-header {
        background: var(--notion-sidebar);
        padding: 0.75rem 1rem;
        font-weight: 600;
        border-bottom: 2px solid var(--notion-border);
    }

    .airtable-row {
        padding: 0.75rem 1rem;
        border-bottom: 1px solid var(--notion-border);
        transition: background 0.2s ease;
    }

    .airtable-row:hover {
        background: var(--notion-hover);
        cursor: pointer;
    }

    /* Cards */
    .card-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 1rem;
        margin: 1rem 0;
    }

    .card {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        border-radius: 8px;
        padding: 1.5rem;
        transition: all 0.2s ease;
    }

    .card:hover {
        box-shadow: 0 4px 12px var(--notion-shadow);
        transform: translateY(-2px);
    }

    .card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }

    .card-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--notion-text);
    }

    .card-meta {
        font-size: 0.875rem;
        color: var(--notion-text-light);
        margin-top: 0.5rem;
    }

    /* Metrics */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }

    .metric-card {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        border-radius: 6px;
        padding: 1rem;
    }

    .metric-label {
        font-size: 0.875rem;
        color: var(--notion-text-light);
        margin-bottom: 0.25rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--notion-text);
    }

    .metric-change {
        font-size: 0.875rem;
        margin-top: 0.25rem;
    }

    .metric-positive { color: var(--notion-green); }
    .metric-negative { color: var(--notion-red); }

    /* Buttons */
    .stButton > button {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        color: var(--notion-text);
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--notion-hover);
        border-color: var(--notion-text-light);
    }

    .stButton > button[kind="primary"] {
        background: var(--notion-blue);
        color: white;
        border-color: var(--notion-blue);
    }

    .stButton > button[kind="primary"]:hover {
        background: #1a6ec1;
        border-color: #1a6ec1;
    }

    /* Timeline */
    .timeline {
        position: relative;
        padding-left: 2rem;
    }

    .timeline::before {
        content: '';
        position: absolute;
        left: 0.5rem;
        top: 0;
        bottom: 0;
        width: 2px;
        background: var(--notion-border);
    }

    .timeline-item {
        position: relative;
        padding-bottom: 2rem;
    }

    .timeline-dot {
        position: absolute;
        left: -1.55rem;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: var(--notion-blue);
        border: 2px solid var(--notion-bg);
    }

    .timeline-content {
        background: var(--notion-bg);
        border: 1px solid var(--notion-border);
        border-radius: 6px;
        padding: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def render_dashboard_header():
    """Render main dashboard header"""
    st.markdown("""
    <div class="notion-header">
        <h1 class="notion-title">🌍 ClarityChain</h1>
        <p class="notion-subtitle">AI-Powered Procurement Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)

def render_rfq_card(rfq_data):
    """Render a single RFQ card"""
    status_class = f"status-{rfq_data.get('status', 'draft').lower()}"
    status_emoji = {
        'draft': '📝',
        'active': '🔄',
        'completed': '✅',
        'archived': '📦'
    }

    st.markdown(f"""
    <div class="card">
        <div class="card-header">
            <div class="card-title">{rfq_data.get('name', 'Untitled RFQ')}</div>
            <span class="{status_class} status-badge">
                {status_emoji.get(rfq_data.get('status', 'draft'), '📝')} {rfq_data.get('status', 'Draft').title()}
            </span>
        </div>
        <div class="card-meta">
            <div>📅 Created: {rfq_data.get('created_date', 'N/A')}</div>
            <div>📊 Offers: {rfq_data.get('offer_count', 0)}</div>
            <div>💰 Est. Value: ${rfq_data.get('estimated_value', 'N/A')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP NAVIGATION
# ============================================================================

def main():
    st.set_page_config(
        page_title="ClarityChain - Procurement Intelligence",
        layout="wide",
        page_icon="🌍",
        initial_sidebar_state="expanded"
    )

    # Apply Notion-style CSS
    render_notion_header()

    # Initialize session state
    if 'current_view' not in st.session_state:
        st.session_state.current_view = 'dashboard'
    if 'rfq_list' not in st.session_state:
        st.session_state.rfq_list = []
    if 'current_rfq' not in st.session_state:
        st.session_state.current_rfq = None

    # Load data in background
    with st.spinner("Loading data..."):
        prices_df, trade_df, data_source = load_data()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("### 🧭 Navigation")

        if st.button("📊 Dashboard", use_container_width=True, key="nav_dashboard"):
            st.session_state.current_view = 'dashboard'
            st.rerun()

        if st.button("✨ New RFQ", use_container_width=True, type="primary", key="nav_new_rfq"):
            st.session_state.current_view = 'new_rfq'
            st.rerun()

        st.markdown("---")
        st.markdown("### 📁 My RFQs")

        if st.session_state.rfq_list:
            for idx, rfq in enumerate(st.session_state.rfq_list):
                rfq_id = rfq.get('id', idx)
                if st.button(f"📄 {rfq.get('name', 'Untitled')}", use_container_width=True, key=f"nav_rfq_{rfq_id}"):
                    st.session_state.current_rfq = rfq
                    st.session_state.current_view = 'rfq_detail'
                    st.rerun()
        else:
            st.info("No RFQs yet")

        st.markdown("---")
        st.markdown("### 📈 Market Intelligence")

        if st.button("🌐 Price Dashboard", use_container_width=True, key="nav_price_dash"):
            st.session_state.current_view = 'price_dashboard'
            st.rerun()

        if st.button("🔮 Forecasting", use_container_width=True, key="nav_forecasting"):
            st.session_state.current_view = 'forecasting'
            st.rerun()

        if st.button("🚗 EV Demand", use_container_width=True, key="nav_ev_demand"):
            st.session_state.current_view = 'ev_demand'
            st.rerun()

        if st.button("🌍 Geopolitical Risk", use_container_width=True, key="nav_geo_risk"):
            st.session_state.current_view = 'geo_risk'
            st.rerun()

        if st.button("🔗 Supply Chain", use_container_width=True, key="nav_supply_chain"):
            st.session_state.current_view = 'supply_chain'
            st.rerun()

    # Main content area
    if st.session_state.current_view == 'dashboard':
        render_dashboard_view()
    elif st.session_state.current_view == 'new_rfq':
        render_new_rfq_view()
    elif st.session_state.current_view == 'rfq_detail':
        render_rfq_detail_view()
    elif st.session_state.current_view == 'price_dashboard':
        render_price_dashboard_view(prices_df, trade_df, data_source)
    elif st.session_state.current_view == 'forecasting':
        render_forecasting_view(prices_df, data_source)
    elif st.session_state.current_view == 'ev_demand':
        render_ev_demand_view()
    elif st.session_state.current_view == 'geo_risk':
        render_geo_risk_view()
    elif st.session_state.current_view == 'supply_chain':
        render_supply_chain_view(trade_df)

# ============================================================================
# VIEW RENDERERS
# ============================================================================

def render_dashboard_view():
    """Main dashboard view with RFQ overview"""
    render_dashboard_header()

    # Metrics
    total_rfqs = len(st.session_state.rfq_list)
    active_rfqs = sum(1 for rfq in st.session_state.rfq_list if rfq.get('status') == 'active')
    completed_rfqs = sum(1 for rfq in st.session_state.rfq_list if rfq.get('status') == 'completed')

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total RFQs</div>
            <div class="metric-value">{total_rfqs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active</div>
            <div class="metric-value">{active_rfqs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completed</div>
            <div class="metric-value">{completed_rfqs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        total_value = sum(float(rfq.get('estimated_value', 0)) for rfq in st.session_state.rfq_list)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Value</div>
            <div class="metric-value">${total_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # RFQ Cards Grid
    st.markdown("## 📋 Recent RFQs")

    if st.session_state.rfq_list:
        st.markdown('<div class="card-grid">', unsafe_allow_html=True)
        for rfq in st.session_state.rfq_list[:6]:  # Show last 6
            render_rfq_card(rfq)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("🎯 No RFQs yet. Click 'New RFQ' to get started!")

def render_new_rfq_view():
    """Conversational RFQ creation with AI assistant"""
    render_dashboard_header()

    st.markdown("## ✨ Create New RFQ")
    st.markdown("Chat with our AI assistant to build your procurement request")

    # Initialize chat for new RFQ
    if 'rfq_chat_history' not in st.session_state:
        st.session_state.rfq_chat_history = []
        st.session_state.rfq_data = {}
        # Start conversation
        initial_message = {
            'role': 'assistant',
            'content': "👋 Hi! I'll help you create your RFQ. Let's start with the basics.\n\n**What material or product are you looking to procure?**"
        }
        st.session_state.rfq_chat_history.append(initial_message)

    # Two-column layout: Chat + Live Preview
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 💬 Conversation")

        # Chat container
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.rfq_chat_history:
                if msg['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-message chat-user">
                        <strong>You:</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message chat-assistant">
                        <strong>AI Assistant:</strong><br>{msg['content']}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        user_input = st.chat_input("Type your answer...")

        if user_input:
            # Add user message
            st.session_state.rfq_chat_history.append({
                'role': 'user',
                'content': user_input
            })

            # Process and generate AI response
            ai_response = process_rfq_conversation(user_input, st.session_state.rfq_data)

            st.session_state.rfq_chat_history.append({
                'role': 'assistant',
                'content': ai_response
            })

            st.rerun()

    with col2:
        st.markdown("### 📄 Live RFQ Preview")

        # Editable RFQ document
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)

        # Title
        rfq_title = st.text_input(
            "RFQ Title",
            value=st.session_state.rfq_data.get('title', ''),
            placeholder="e.g., Lithium Carbonate Q1 2025",
            key='rfq_title_input'
        )
        if rfq_title:
            st.session_state.rfq_data['title'] = rfq_title

        # Material
        if 'material' in st.session_state.rfq_data:
            st.markdown(f"**Material:** {st.session_state.rfq_data['material']}")

        # Quantity
        if 'quantity' in st.session_state.rfq_data:
            quantity = st.number_input(
                "Quantity (tonnes)",
                value=float(st.session_state.rfq_data.get('quantity', 0)),
                min_value=0.0,
                step=1.0
            )
            st.session_state.rfq_data['quantity'] = quantity

        # Delivery date
        if 'delivery_date' in st.session_state.rfq_data:
            delivery_date = st.date_input(
                "Delivery Date",
                value=datetime.strptime(st.session_state.rfq_data['delivery_date'], '%Y-%m-%d').date()
                if isinstance(st.session_state.rfq_data['delivery_date'], str)
                else st.session_state.rfq_data['delivery_date']
            )
            st.session_state.rfq_data['delivery_date'] = delivery_date.strftime('%Y-%m-%d')

        # Specifications
        if 'specifications' in st.session_state.rfq_data:
            st.markdown("**Technical Specifications:**")
            specs = st.text_area(
                "Specifications",
                value=st.session_state.rfq_data.get('specifications', ''),
                height=150,
                label_visibility="collapsed"
            )
            st.session_state.rfq_data['specifications'] = specs

        st.markdown('</div>', unsafe_allow_html=True)

        # Action buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("💾 Save Draft", use_container_width=True, key="save_draft_btn"):
                save_rfq_draft()
                st.success("Draft saved!")

        with col_b:
            if st.button("📤 Publish RFQ", use_container_width=True, type="primary", key="publish_rfq_btn"):
                publish_rfq()
                st.success("RFQ published!")
                st.session_state.current_view = 'dashboard'
                st.rerun()

def render_rfq_detail_view():
    """Detailed view of a specific RFQ with offers"""
    if not st.session_state.current_rfq:
        st.warning("No RFQ selected")
        return

    rfq = st.session_state.current_rfq

    render_dashboard_header()

    # RFQ Header with inline edit
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"## 📄 {rfq.get('name', 'Untitled RFQ')}")
    with col2:
        status = st.selectbox(
            "Status",
            ['draft', 'active', 'completed', 'archived'],
            index=['draft', 'active', 'completed', 'archived'].index(rfq.get('status', 'draft'))
        )
        rfq['status'] = status

    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Details", "📊 Offers", "💬 Analysis", "📈 Insights"])

    with tab1:
        render_rfq_details_tab(rfq)

    with tab2:
        render_rfq_offers_tab(rfq)

    with tab3:
        render_rfq_analysis_tab(rfq)

    with tab4:
        render_rfq_insights_tab(rfq)

def render_rfq_details_tab(rfq):
    """Render RFQ details in editable blocks"""
    st.markdown("### 📝 RFQ Information")

    # Editable blocks
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("**Material**")
        material = st.text_input("Material", value=rfq.get('material', ''), label_visibility="collapsed")
        rfq['material'] = material
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("**Quantity**")
        quantity = st.number_input("Quantity", value=float(rfq.get('quantity', 0)), label_visibility="collapsed")
        rfq['quantity'] = quantity
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("**Delivery Date**")
        delivery_date = st.date_input("Delivery", label_visibility="collapsed")
        rfq['delivery_date'] = delivery_date.strftime('%Y-%m-%d')
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("**Budget**")
        budget = st.number_input("Budget", value=float(rfq.get('estimated_value', 0)), label_visibility="collapsed")
        rfq['estimated_value'] = budget
        st.markdown('</div>', unsafe_allow_html=True)

    # Specifications block
    st.markdown('<div class="notion-block">', unsafe_allow_html=True)
    st.markdown("**Technical Specifications**")
    specs = st.text_area("Specifications", value=rfq.get('specifications', ''), height=200, label_visibility="collapsed")
    rfq['specifications'] = specs
    st.markdown('</div>', unsafe_allow_html=True)

def render_rfq_offers_tab(rfq):
    """Airtable-style offers view"""
    st.markdown("### 📊 Submitted Offers")

    # Upload new offers
    with st.expander("➕ Add New Offer"):
        uploaded_files = st.file_uploader(
            "Upload offer documents",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'xlsx']
        )

        if st.button("Add Offer") and uploaded_files:
            # Process offer upload
            st.success("Offer added successfully!")

    # Offers table
    if 'offers' not in rfq:
        rfq['offers'] = []

    if rfq['offers']:
        # Create dataframe for offers
        offers_data = []
        for i, offer in enumerate(rfq['offers']):
            offers_data.append({
                'Supplier': offer.get('supplier_name', f'Supplier {i+1}'),
                'Price': f"${offer.get('price', 0):,.2f}",
                'Lead Time': offer.get('lead_time', 'N/A'),
                'Score': f"{offer.get('total_weighted_score', 0):.1f}",
                'Status': offer.get('recommendation', 'Under Review')
            })

        df = pd.DataFrame(offers_data)

        # Interactive table
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Score': st.column_config.ProgressColumn(
                    'Score',
                    min_value=0,
                    max_value=100
                ),
                'Status': st.column_config.TextColumn(
                    'Status',
                    width='medium'
                )
            }
        )

        # Analyze button
        if st.button("🤖 Analyze All Offers", type="primary"):
            with st.spinner("Running AI analysis..."):
                analyze_offers_for_rfq(rfq)
                st.success("Analysis complete!")
                st.rerun()
    else:
        st.info("No offers submitted yet")

def render_rfq_analysis_tab(rfq):
    """Chat-based analysis interface"""
    st.markdown("### 💬 AI Analysis Assistant")

    # Initialize analysis chat
    if 'analysis_chat' not in st.session_state:
        st.session_state.analysis_chat = []

    # Display chat
    for msg in st.session_state.analysis_chat:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])

    # Quick questions
    st.markdown("**Quick Questions:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💰 Best price?"):
            ask_analysis_question("Which offer has the best price?")

    with col2:
        if st.button("⚡ Fastest delivery?"):
            ask_analysis_question("Which offer has the fastest delivery?")

    with col3:
        if st.button("⚠️ Risk assessment?"):
            ask_analysis_question("What are the main risks for each offer?")

    # Chat input
    user_question = st.chat_input("Ask about the offers...")
    if user_question:
        ask_analysis_question(user_question)
        st.rerun()

def render_rfq_insights_tab(rfq):
    """Market insights and recommendations"""
    st.markdown("### 📈 Market Insights")

    material = rfq.get('material', 'lithium')

    # Load price data
    prices_df, _, _ = load_data()

    if not prices_df.empty and material in prices_df['material'].values:
        material_data = prices_df[prices_df['material'] == material]

        # Price trend chart
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("**Price Trends**")

        fig = px.line(
            material_data.sort_values('date'),
            x='date',
            y='price',
            title=f"{material.title()} Price Trend (Last 12 Months)"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Recommendations
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("**AI Recommendations**")

        current_price = material_data['price'].iloc[-1]
        avg_price = material_data['price'].tail(90).mean()

        if current_price < avg_price * 0.95:
            st.success("✅ **Good time to buy** - Current price is below 90-day average")
        elif current_price > avg_price * 1.05:
            st.warning("⚠️ **Consider waiting** - Current price is above 90-day average")
        else:
            st.info("ℹ️ **Normal market conditions** - Price is near average")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("No market data available for this material")

def render_price_dashboard_view(prices_df, trade_df, data_source):
    """Enhanced price dashboard with filters"""
    render_dashboard_header()

    st.markdown("## 🌐 Global Price Intelligence")

    if prices_df.empty:
        st.warning("No price data available")
        return

    # Enrich data
    prices_df = enrich_price_data(prices_df)

    # Filters in expandable section
    with st.expander("🔍 Filters", expanded=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            materials = st.multiselect(
                "Materials",
                options=sorted(prices_df['material'].unique()),
                default=sorted(prices_df['material'].unique())[:3]
            )

        with col2:
            regions = st.multiselect(
                "Regions",
                options=sorted(prices_df['region'].unique()),
                default=sorted(prices_df['region'].unique())
            )

        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(prices_df['date'].min(), prices_df['date'].max())
            )

        with col4:
            chart_type = st.selectbox("Chart Type", ["Line", "Area", "Candlestick"])

    # Apply filters
    filtered_df = prices_df[
        (prices_df['material'].isin(materials)) &
        (prices_df['region'].isin(regions))
    ]

    if len(date_range) == 2:
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(date_range[0])) &
            (filtered_df['date'] <= pd.Timestamp(date_range[1]))
        ]

    # Main chart
    st.markdown('<div class="notion-block">', unsafe_allow_html=True)

    if chart_type == "Line":
        fig = px.line(
            filtered_df.sort_values('date'),
            x='date',
            y='price',
            color='material',
            title="Price Trends"
        )
    elif chart_type == "Area":
        fig = px.area(
            filtered_df.sort_values('date'),
            x='date',
            y='price',
            color='material',
            title="Price Trends"
        )
    else:
        # Candlestick placeholder
        fig = px.line(
            filtered_df.sort_values('date'),
            x='date',
            y='price',
            color='material',
            title="Price Trends"
        )

    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Statistics cards
    st.markdown("### 📊 Current Market Stats")

    col1, col2, col3, col4 = st.columns(4)

    for i, material in enumerate(materials[:4]):
        mat_data = filtered_df[filtered_df['material'] == material]
        if not mat_data.empty:
            current_price = mat_data['price'].iloc[-1]
            price_change = ((current_price - mat_data['price'].iloc[0]) / mat_data['price'].iloc[0] * 100) if len(mat_data) > 1 else 0

            with [col1, col2, col3, col4][i]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{material.title()}</div>
                    <div class="metric-value">${current_price:,.0f}</div>
                    <div class="metric-change {'metric-positive' if price_change > 0 else 'metric-negative'}">
                        {price_change:+.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_forecasting_view(prices_df, data_source):
    """Forecasting with conversational interface"""
    render_dashboard_header()

    st.markdown("## 🔮 Price Forecasting")

    if prices_df.empty:
        st.warning("No price data available")
        return

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        st.markdown("### ⚙️ Configuration")

        material = st.selectbox(
            "Material",
            options=sorted(prices_df['material'].unique())
        )

        horizon = st.slider("Forecast Horizon (months)", 1, 12, 6)

        use_fundamentals = st.checkbox("Include Fundamental Analysis", value=True)

        if st.button("Generate Forecast", type="primary", use_container_width=True):
            st.session_state.forecast_generated = True
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.session_state.get('forecast_generated', False):
            st.markdown('<div class="notion-block">', unsafe_allow_html=True)
            st.markdown("### 📈 Forecast Results")

            # Generate forecast
            material_data = prices_df[prices_df['material'] == material]

            if BaselineForecaster and len(material_data) >= 10:
                config = load_config()
                forecaster = BaselineForecaster(config)
                result = forecaster.fit_predict(material_data, material, use_fundamentals)

                # Plot
                fig = go.Figure()

                # Historical
                fig.add_trace(go.Scatter(
                    x=material_data['date'],
                    y=material_data['price'],
                    name='Historical',
                    line=dict(color='blue')
                ))

                # Forecast
                forecast_df = result.get('forecast', pd.DataFrame())
                if not forecast_df.empty:
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'],
                        y=forecast_df['forecast_mean'],
                        name='Forecast',
                        line=dict(color='orange', dash='dash')
                    ))

                fig.update_layout(title=f"{material.title()} Price Forecast", height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                metrics = result.get('metrics', {})
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Current Price", f"${metrics.get('current_price', 0):,.0f}")
                with col_b:
                    st.metric("Trend", metrics.get('trend', 'N/A').title())
                with col_c:
                    st.metric("Volatility", metrics.get('volatility_regime', 'N/A').title())
            else:
                st.warning("Insufficient data for forecasting")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("Configure and generate a forecast to see results")

def render_ev_demand_view():
    """EV demand analysis"""
    render_dashboard_header()

    st.markdown("## 🚗 EV Adoption Impact")

    config = load_config()
    ev_fetcher = EVAdoptionFetcher(config)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        material = st.selectbox("Material", ['lithium', 'cobalt', 'nickel', 'copper', 'rare_earths'])
        scenario = st.radio("Scenario", ['conservative', 'stated_policies', 'sustainable'])

        if st.button("Calculate Demand", type="primary", use_container_width=True):
            st.session_state.ev_demand_calculated = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if st.session_state.get('ev_demand_calculated', False):
            demand_data = ev_fetcher.calculate_material_demand(material, scenario)

            if not demand_data.empty:
                fig = px.line(
                    demand_data,
                    x='year',
                    y='material_demand_tons',
                    title=f"{material.title()} Demand Projection"
                )
                st.plotly_chart(fig, use_container_width=True)

                # Metrics
                current = demand_data[demand_data['year'] == 2024]['material_demand_tons'].iloc[0]
                future = demand_data[demand_data['year'] == 2030]['material_demand_tons'].iloc[0]
                growth = ((future - current) / current) * 100

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("2024 Demand", f"{current:,.0f}t")
                with col_b:
                    st.metric("2030 Projected", f"{future:,.0f}t")
                with col_c:
                    st.metric("Growth", f"{growth:+.1f}%")

def render_geo_risk_view():
    """Geopolitical risk monitoring"""
    render_dashboard_header()

    st.markdown("## 🌍 Geopolitical Risk Monitor")

    config = load_config()
    gdelt_fetcher = GDELTFetcher(config)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown('<div class="notion-block">', unsafe_allow_html=True)
        material = st.selectbox("Material", ['lithium', 'cobalt', 'nickel', 'copper', 'rare_earths'], key='geo_mat')
        days_back = st.slider("Analysis Period (days)", 7, 90, 30)

        if st.button("Analyze Risk", type="primary", use_container_width=True):
            with st.spinner("Scanning events..."):
                events = gdelt_fetcher.fetch_events_for_material(material, days_back)
                risk_score = gdelt_fetcher.generate_risk_score(events, material)
                st.session_state.risk_analysis = risk_score
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if 'risk_analysis' in st.session_state:
            risk = st.session_state.risk_analysis

            # Risk level card
            st.markdown('<div class="notion-block">', unsafe_allow_html=True)

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                score = risk['risk_score']
                color = 'red' if score > 0.7 else 'yellow' if score > 0.4 else 'green'
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Risk Score</div>
                    <div class="metric-value" style="color: var(--notion-{color})">{score:.2f}</div>
                </div>
                """, unsafe_allow_html=True)

            with col_b:
                st.metric("Risk Level", risk['risk_level'])

            with col_c:
                st.metric("Recent Events", risk['recent_events'])

            st.markdown("**Risk Description:**")
            st.info(risk.get('risk_description', 'No specific risks detected'))

            st.markdown('</div>', unsafe_allow_html=True)

def render_supply_chain_view(trade_df):
    """Supply chain analysis"""
    render_dashboard_header()

    st.markdown("## 🔗 Supply Chain Intelligence")

    if trade_df.empty:
        st.warning("No trade data available")
        return

    material = st.selectbox("Material", trade_df['material'].unique())
    material_trade = trade_df[trade_df['material'] == material]

    if not material_trade.empty and 'value_usd' in material_trade.columns:
        col1, col2 = st.columns(2)

        with col1:
            # Market concentration
            st.markdown('<div class="notion-block">', unsafe_allow_html=True)
            st.markdown("### Supplier Concentration")

            total_value = material_trade['value_usd'].sum()
            supplier_shares = material_trade.groupby('exporter')['value_usd'].sum() / total_value

            fig = px.pie(
                values=supplier_shares.values,
                names=supplier_shares.index,
                title="Market Share by Supplier"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Risk assessment
            st.markdown('<div class="notion-block">', unsafe_allow_html=True)
            st.markdown("### Concentration Risk")

            hhi = (supplier_shares ** 2).sum()

            st.metric("HHI Score", f"{hhi:.3f}")

            if hhi > 0.25:
                st.error("🔴 **HIGH RISK** - Market highly concentrated")
            elif hhi > 0.15:
                st.warning("🟡 **MEDIUM RISK** - Moderate concentration")
            else:
                st.success("🟢 **LOW RISK** - Well diversified")

            st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def enrich_price_data(prices_df):
    """Add region and market type classifications"""
    if prices_df.empty:
        return prices_df

    prices_df = prices_df.copy()

    def classify_source(source):
        source_lower = str(source).lower()
        if 'fred' in source_lower:
            return 'North America', 'Government Data'
        elif 'ecb' in source_lower:
            return 'Europe', 'Central Bank'
        elif 'worldbank' in source_lower:
            return 'Global', 'International'
        elif 'lme' in source_lower:
            return 'Europe', 'Exchange'
        else:
            return 'Other', 'Market Data'

    if 'source' in prices_df.columns:
        classifications = prices_df['source'].apply(lambda x: pd.Series(classify_source(x)))
        prices_df['region'] = classifications[0]
        prices_df['market_type'] = classifications[1]

    return prices_df

def process_rfq_conversation(user_input, rfq_data):
    """Process user input and generate AI response for RFQ creation"""
    user_lower = user_input.lower()

    # Determine what information is missing
    if 'material' not in rfq_data:
        # Extract material from input
        materials = ['lithium', 'cobalt', 'nickel', 'copper', 'aluminum', 'zinc']
        for mat in materials:
            if mat in user_lower:
                rfq_data['material'] = mat
                rfq_data['title'] = f"{mat.title()} Procurement"
                return f"Great! I've set the material to **{mat.title()}**.\n\n**How much do you need? (Please specify quantity in tonnes)**"

        rfq_data['material'] = user_input
        rfq_data['title'] = f"{user_input.title()} Procurement"
        return f"I've set the material to **{user_input}**.\n\n**How much do you need? (Please specify quantity in tonnes)**"

    elif 'quantity' not in rfq_data:
        # Extract quantity
        import re
        numbers = re.findall(r'\d+\.?\d*', user_input)
        if numbers:
            rfq_data['quantity'] = float(numbers[0])
            return f"Perfect! **{rfq_data['quantity']} tonnes** of {rfq_data['material']}.\n\n**When do you need it delivered? (Please provide a date)**"
        else:
            return "I couldn't understand the quantity. Could you specify it as a number? (e.g., 1000 tonnes)"

    elif 'delivery_date' not in rfq_data:
        # Extract date
        rfq_data['delivery_date'] = datetime.now().strftime('%Y-%m-%d')
        return f"Got it! Delivery date noted.\n\n**Do you have any specific technical specifications or quality requirements?**\n\n(You can type 'none' if no special requirements)"

    elif 'specifications' not in rfq_data:
        if user_lower != 'none':
            rfq_data['specifications'] = user_input
        else:
            rfq_data['specifications'] = "Standard specifications"

        return f"Excellent! Your RFQ is now complete.\n\n**Summary:**\n- Material: {rfq_data['material']}\n- Quantity: {rfq_data['quantity']} tonnes\n- Delivery: {rfq_data['delivery_date']}\n- Specifications: {rfq_data['specifications']}\n\n**You can now save as draft or publish to suppliers!**"

    else:
        return "Your RFQ is complete! You can edit any field on the right, or click 'Publish RFQ' to send it to suppliers."

def save_rfq_draft():
    """Save current RFQ as draft"""
    if st.session_state.rfq_data:
        rfq = st.session_state.rfq_data.copy()
        rfq['status'] = 'draft'
        rfq['created_date'] = datetime.now().strftime('%Y-%m-%d')
        rfq['name'] = rfq.get('title', 'Untitled RFQ')
        rfq['offer_count'] = 0
        rfq['estimated_value'] = 0

        if 'id' not in rfq:
            rfq['id'] = len(st.session_state.rfq_list) + 1
            st.session_state.rfq_list.append(rfq)
        else:
            # Update existing
            for i, existing in enumerate(st.session_state.rfq_list):
                if existing.get('id') == rfq['id']:
                    st.session_state.rfq_list[i] = rfq
                    break

def publish_rfq():
    """Publish RFQ and make it active"""
    if st.session_state.rfq_data:
        rfq = st.session_state.rfq_data.copy()
        rfq['status'] = 'active'
        rfq['created_date'] = datetime.now().strftime('%Y-%m-%d')
        rfq['name'] = rfq.get('title', 'Untitled RFQ')
        rfq['offer_count'] = 0

        # Estimate value
        quantity = float(rfq.get('quantity', 0))
        material = rfq.get('material', 'lithium')

        # Simple price estimation (you can make this more sophisticated)
        price_map = {
            'lithium': 15000,
            'cobalt': 35000,
            'nickel': 18000,
            'copper': 8500,
            'aluminum': 2500
        }
        unit_price = price_map.get(material, 10000)
        rfq['estimated_value'] = quantity * unit_price

        if 'id' not in rfq:
            rfq['id'] = len(st.session_state.rfq_list) + 1
            st.session_state.rfq_list.append(rfq)
        else:
            for i, existing in enumerate(st.session_state.rfq_list):
                if existing.get('id') == rfq['id']:
                    st.session_state.rfq_list[i] = rfq
                    break

        # Reset chat
        st.session_state.rfq_chat_history = []
        st.session_state.rfq_data = {}

def analyze_offers_for_rfq(rfq):
    """Analyze all offers for an RFQ using AI"""
    try:
        from utils.agent_client import get_agent_client
        agent = get_agent_client()

        if not agent.health_check():
            st.error("AI Agent not available")
            return

        # Mock analysis for demo - replace with actual agent call
        if 'offers' in rfq and rfq['offers']:
            for offer in rfq['offers']:
                offer['analyzed'] = True
                offer['total_weighted_score'] = np.random.uniform(60, 95)
                offer['recommendation'] = 'Best Offer' if offer['total_weighted_score'] > 85 else 'Good Alternative'

    except Exception as e:
        st.error(f"Analysis failed: {e}")

def ask_analysis_question(question):
    """Ask a question about the analysis"""
    st.session_state.analysis_chat.append({
        'role': 'user',
        'content': question
    })

    # Generate AI response (simplified - replace with actual agent call)
    responses = {
        'Which offer has the best price?': "Based on the analysis, Offer 1 has the most competitive price at $14,500/tonne, which is 8% below the market average.",
        'Which offer has the fastest delivery?': "Offer 2 offers the fastest delivery at 30 days, compared to 45-60 days for other offers.",
        'What are the main risks for each offer?': "Key risks identified:\n- Offer 1: Moderate financial risk due to payment terms\n- Offer 2: Higher delivery risk due to single source\n- Offer 3: Technical compliance concerns"
    }

    response = responses.get(question, "Let me analyze that for you. Based on the submitted offers, I recommend focusing on the total cost of ownership rather than just the unit price. Consider lead times, payment terms, and supplier reliability in your decision.")

    st.session_state.analysis_chat.append({
        'role': 'assistant',
        'content': response
    })

# ============================================================================
# ADDITIONAL UI COMPONENTS
# ============================================================================

def render_timeline_view(events):
    """Render timeline of RFQ events"""
    st.markdown('<div class="timeline">', unsafe_allow_html=True)

    for event in events:
        st.markdown(f"""
        <div class="timeline-item">
            <div class="timeline-dot"></div>
            <div class="timeline-content">
                <strong>{event.get('title', 'Event')}</strong>
                <div style="color: var(--notion-text-light); font-size: 0.875rem;">
                    {event.get('date', '')} • {event.get('user', 'System')}
                </div>
                <p>{event.get('description', '')}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

def render_comparison_matrix(offers):
    """Render offer comparison matrix"""
    if not offers:
        st.info("No offers to compare")
        return

    # Create comparison dataframe
    comparison_data = []
    for offer in offers:
        comparison_data.append({
            'Supplier': offer.get('supplier_name', 'Unknown'),
            'Price': offer.get('price', 0),
            'Lead Time': offer.get('lead_time', 'N/A'),
            'Payment Terms': offer.get('payment_terms', 'N/A'),
            'Score': offer.get('total_weighted_score', 0)
        })

    df = pd.DataFrame(comparison_data)

    # Highlight best values
    def highlight_best(s):
        if s.name == 'Price':
            is_min = s == s.min()
            return ['background-color: var(--notion-green-bg)' if v else '' for v in is_min]
        elif s.name == 'Score':
            is_max = s == s.max()
            return ['background-color: var(--notion-green-bg)' if v else '' for v in is_max]
        return [''] * len(s)

    styled_df = df.style.apply(highlight_best, subset=['Price', 'Score'])

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

def render_kanban_view(rfq_list):
    """Render Kanban board view of RFQs"""
    statuses = ['draft', 'active', 'completed', 'archived']
    status_names = {'draft': '📝 Draft', 'active': '🔄 Active', 'completed': '✅ Completed', 'archived': '📦 Archived'}

    cols = st.columns(len(statuses))

    for i, status in enumerate(statuses):
        with cols[i]:
            st.markdown(f"### {status_names[status]}")

            status_rfqs = [rfq for rfq in rfq_list if rfq.get('status') == status]

            for rfq in status_rfqs:
                st.markdown(f"""
                <div class="card" style="margin-bottom: 0.5rem;">
                    <div class="card-title" style="font-size: 0.9rem;">{rfq.get('name', 'Untitled')}</div>
                    <div class="card-meta" style="font-size: 0.75rem;">
                        📊 {rfq.get('offer_count', 0)} offers
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(f"View", key=f"kanban_{rfq.get('id')}"):
                    st.session_state.current_rfq = rfq
                    st.session_state.current_view = 'rfq_detail'
                    st.rerun()

def render_activity_feed():
    """Render recent activity feed"""
    st.markdown("### 📢 Recent Activity")

    activities = [
        {'icon': '📝', 'text': 'New RFQ created: Lithium Carbonate Q1 2025', 'time': '2 hours ago'},
        {'icon': '📊', 'text': 'Offer received from Supplier A', 'time': '3 hours ago'},
        {'icon': '✅', 'text': 'Analysis completed for Cobalt RFQ', 'time': '5 hours ago'},
        {'icon': '📈', 'text': 'Price alert: Nickel +5% this week', 'time': '1 day ago'},
    ]

    for activity in activities:
        st.markdown(f"""
        <div class="notion-block" style="padding: 0.75rem; margin-bottom: 0.5rem;">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">{activity['icon']}</span>
                <div style="flex: 1;">
                    <div>{activity['text']}</div>
                    <div style="color: var(--notion-text-light); font-size: 0.875rem;">{activity['time']}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

def render_smart_suggestions():
    """Render AI-powered suggestions"""
    st.markdown("### 💡 Smart Suggestions")

    suggestions = [
        {
            'type': 'opportunity',
            'title': 'Good time to buy Lithium',
            'description': 'Prices are 12% below 90-day average',
            'action': 'Create RFQ'
        },
        {
            'type': 'risk',
            'title': 'Supply chain alert: Cobalt',
            'description': 'Geopolitical risk increased in DRC region',
            'action': 'View Details'
        },
        {
            'type': 'insight',
            'title': 'EV Demand Surge Expected',
            'description': 'Q2 2025 lithium demand projected +18%',
            'action': 'See Forecast'
        }
    ]

    for suggestion in suggestions:
        bg_color = {
            'opportunity': 'var(--notion-green-bg)',
            'risk': 'var(--notion-red-bg)',
            'insight': 'var(--notion-blue-bg)'
        }

        st.markdown(f"""
        <div class="notion-block" style="background: {bg_color.get(suggestion['type'])}; padding: 1rem;">
            <div style="font-weight: 600; margin-bottom: 0.25rem;">{suggestion['title']}</div>
            <div style="font-size: 0.875rem; margin-bottom: 0.5rem;">{suggestion['description']}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button(suggestion['action'], key=f"sugg_{suggestion['title']}"):
            st.info(f"Action: {suggestion['action']}")

def render_analytics_dashboard():
    """Render analytics overview"""
    st.markdown("### 📊 Analytics Overview")

    # Mock data
    col1, col2 = st.columns(2)

    with col1:
        # RFQ trends
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='M')
        rfq_counts = np.random.randint(5, 20, len(dates))

        fig = px.line(
            x=dates,
            y=rfq_counts,
            title="RFQ Creation Trend",
            labels={'x': 'Month', 'y': 'RFQs Created'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Savings analysis
        categories = ['Direct Cost Savings', 'Risk Mitigation', 'Process Efficiency', 'Better Terms']
        values = [125000, 85000, 45000, 30000]

        fig = px.bar(
            x=categories,
            y=values,
            title="Value Generated (USD)",
            labels={'x': 'Category', 'y': 'Value ($)'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_export_options(data):
    """Render export options for data"""
    st.markdown("### 📥 Export Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📄 Export to PDF", use_container_width=True):
            st.info("PDF export functionality")

    with col2:
        if st.button("📊 Export to Excel", use_container_width=True):
            st.info("Excel export functionality")

    with col3:
        if st.button("📋 Copy to Clipboard", use_container_width=True):
            st.info("Data copied!")

def render_collaboration_panel():
    """Render collaboration features"""
    st.markdown("### 👥 Collaboration")

    st.markdown("""
    <div class="notion-block">
        <div style="margin-bottom: 0.5rem;">
            <strong>Team Members</strong>
        </div>
        <div style="display: flex; gap: 0.5rem; margin-bottom: 1rem;">
            <div style="width: 32px; height: 32px; border-radius: 50%; background: var(--notion-blue); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">JD</div>
            <div style="width: 32px; height: 32px; border-radius: 50%; background: var(--notion-green); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">AS</div>
            <div style="width: 32px; height: 32px; border-radius: 50%; background: var(--notion-purple); display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">MK</div>
        </div>
        <div style="font-size: 0.875rem; color: var(--notion-text-light);">
            3 team members • Last activity: 2h ago
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Comments section
    st.markdown("**Comments**")

    comment = st.text_area("Add a comment...", height=80, key="comment_input")
    if st.button("Post Comment", use_container_width=True):
        st.success("Comment posted!")

# ============================================================================
# ENHANCED DASHBOARD WITH ALL FEATURES
# ============================================================================

def render_enhanced_dashboard():
    """Enhanced dashboard with all new UI components"""
    render_dashboard_header()

    # Main content with sidebar
    col1, col2 = st.columns([2, 1])

    with col1:
        # Metrics
        render_dashboard_metrics()

        # Kanban view
        if st.session_state.rfq_list:
            render_kanban_view(st.session_state.rfq_list)

        # Analytics
        render_analytics_dashboard()

    with col2:
        # Activity feed
        render_activity_feed()

        # Smart suggestions
        render_smart_suggestions()

def render_dashboard_metrics():
    """Render dashboard metrics"""
    total_rfqs = len(st.session_state.rfq_list)
    active_rfqs = sum(1 for rfq in st.session_state.rfq_list if rfq.get('status') == 'active')
    completed_rfqs = sum(1 for rfq in st.session_state.rfq_list if rfq.get('status') == 'completed')
    total_value = sum(float(rfq.get('estimated_value', 0)) for rfq in st.session_state.rfq_list)

    st.markdown('<div class="metric-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total RFQs</div>
            <div class="metric-value">{total_rfqs}</div>
            <div class="metric-change metric-positive">+2 this week</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Active</div>
            <div class="metric-value">{active_rfqs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Completed</div>
            <div class="metric-value">{completed_rfqs}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Value</div>
            <div class="metric-value">${total_value:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()