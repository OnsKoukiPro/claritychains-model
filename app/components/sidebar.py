import streamlit as st
import sys
from pathlib import Path

# Add the app directory to path for utils imports
current_dir = Path(__file__).parent
app_dir = current_dir.parent
sys.path.insert(0, str(app_dir))

from utils.config import load_config

def show_sidebar():
    """Display sidebar with configuration options"""

    st.sidebar.header("🌐 Configuration")

    # Data source selection
    use_real_data = st.sidebar.checkbox(
        "Use Real API Data",
        value=True,
        help="Enable to fetch live data from global APIs"
    )
    st.session_state.use_real_data = use_real_data

    # Enhanced features
    use_enhanced_forecasting = st.sidebar.checkbox(
        "Use Enhanced Forecasting",
        value=True,
        help="Include EV adoption and geopolitical risk in forecasts"
    )
    st.session_state.use_enhanced_forecasting = use_enhanced_forecasting

    # Refresh control
    if st.sidebar.button("🔄 Refresh Global Data", type="primary"):
        st.session_state.refresh_data = True
        st.rerun()

    # Load configuration
    config = load_config()
    st.session_state.config = config

    # Display status
    show_sidebar_status(config)

    return config

def show_sidebar_status(config):
    """Display status information in sidebar"""

    st.sidebar.markdown("---")
    st.sidebar.header("🔧 Status")

    # Data source status
    if st.session_state.get('data_loaded'):
        data_source = st.session_state.get('data_source', 'unknown')
        if data_source == "real":
            st.sidebar.success("✅ Global API Data")

            # Show source breakdown if available
            prices_df = st.session_state.get('prices_df')
            if prices_df is not None and not prices_df.empty and 'source' in prices_df.columns:
                sources = prices_df['source'].unique()
                if any('ecb' in str(s).lower() for s in sources) or any('lme' in str(s).lower() for s in sources):
                    st.sidebar.info("🌍 Sources: World Bank, ECB, LME, FRED")
                else:
                    st.sidebar.info("🇺🇸 Sources: FRED, World Bank, Yahoo")
        else:
            st.sidebar.warning("📊 Sample Data")

    # Feature status
    if st.session_state.get('use_enhanced_forecasting'):
        st.sidebar.success("🎯 Enhanced Forecasting: ON")
    else:
        st.sidebar.info("📊 Baseline Forecasting: ON")

    # System info
    if st.session_state.get('prices_df') is not None:
        prices_df = st.session_state.prices_df
        if not prices_df.empty and 'material' in prices_df.columns:
            material_count = len(prices_df['material'].unique())
            st.sidebar.metric("Materials", material_count)

    # Docker environment
    import os
    if os.path.exists('/.dockerenv'):
        st.sidebar.success("🐳 Docker Environment")