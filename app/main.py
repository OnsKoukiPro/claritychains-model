import streamlit as st
import sys
from pathlib import Path
import logging
import os

# Add the current directory to Python path so we can import from app components
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Add src to Python path for data pipeline imports
src_path = current_dir.parent / 'src'
if src_path.exists():
    sys.path.insert(0, str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Now import components - these are in the same directory or subdirectories
try:
    from components.header import show_header
    from components.sidebar import show_sidebar
    from components.filters import show_filters

    # Import pages
    from pages.global_dashboard import show_global_dashboard
    from pages.enhanced_forecasting import show_enhanced_forecasting
    from pages.ev_adoption import show_ev_adoption_analysis
    from pages.geopolitical_risk import show_geopolitical_risk
    from pages.procurement import show_procurement_analysis
    from pages.supply_chain import show_supply_chain_analysis
    from pages.data_sources import show_data_sources

    # Import utils
    from utils.data_loader import load_data, fetch_real_data
    from utils.config import load_config

    logger.info("✅ All imports successful")

except ImportError as e:
    logger.error(f"❌ Import error: {e}")
    st.error(f"Import error: {e}")

def main():
    """Main Streamlit application with new component structure"""

    # Set page config
    st.set_page_config(
        page_title="Critical Materials AI Platform - Global Enhanced",
        layout="wide",
        page_icon="🌍"
    )

    # Show header
    show_header()

    # Initialize session state
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'prices_df' not in st.session_state:
        st.session_state.prices_df = None
    if 'trade_df' not in st.session_state:
        st.session_state.trade_df = None
    if 'data_source' not in st.session_state:
        st.session_state.data_source = None

    # Show sidebar and get configuration
    config = show_sidebar()

    # Load data if not already loaded
    if not st.session_state.data_loaded or st.session_state.get('refresh_data', False):
        with st.spinner("Loading global data..."):
            if st.session_state.get('use_real_data', True):
                prices_df, trade_df, data_source = fetch_real_data()
            else:
                prices_df, trade_df, data_source = load_data()

            st.session_state.prices_df = prices_df
            st.session_state.trade_df = trade_df
            st.session_state.data_source = data_source
            st.session_state.data_loaded = True
            st.session_state.refresh_data = False

    # Show filters in sidebar if data is loaded
    if st.session_state.data_loaded:
        filters = show_filters(st.session_state.prices_df)
        st.session_state.filters = filters

    # Main content area with tabs
    if st.session_state.data_loaded:
        show_main_content()
    else:
        st.warning("Please wait while data is loading...")

def show_main_content():
    """Display the main content with tabs"""

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Global Dashboard",
        "📈 Enhanced Forecasting",
        "🚗 EV Adoption",
        "🌍 Geopolitical Risk",
        "💳 Procurement",
        "🔗 Supply Chain",
        "🌐 Data Sources"
    ])

    # Get data from session state
    prices_df = st.session_state.prices_df
    trade_df = st.session_state.trade_df
    data_source = st.session_state.data_source
    config = st.session_state.get('config', {})
    filters = st.session_state.get('filters', {})

    # Apply filters to data
    if filters and not prices_df.empty:
        filtered_prices = prices_df.copy()
        if filters.get('material'):
            filtered_prices = filtered_prices[filtered_prices['material'] == filters['material']]
        if filters.get('date_range'):
            start_date, end_date = filters['date_range']
            filtered_prices = filtered_prices[
                (filtered_prices['date'] >= start_date) &
                (filtered_prices['date'] <= end_date)
            ]
        prices_df = filtered_prices

    # Display each tab content
    with tab1:
        show_global_dashboard(prices_df, trade_df, data_source, config)

    with tab2:
        use_enhanced = st.session_state.get('use_enhanced_forecasting', True)
        show_enhanced_forecasting(prices_df, data_source, use_enhanced, config)

    with tab3:
        show_ev_adoption_analysis(config)

    with tab4:
        show_geopolitical_risk(config)

    with tab5:
        show_procurement_analysis(prices_df, config)

    with tab6:
        show_supply_chain_analysis(trade_df, config)

    with tab7:
        show_data_sources(config, prices_df, trade_df, data_source)

if __name__ == "__main__":
    main()