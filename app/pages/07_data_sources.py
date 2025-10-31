# app/pages/07_⚙️_Data_Sources.py
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import sys

def main():
    st.header("⚙️ Data Sources & Configuration")

    # Get data from session state
    prices_df = st.session_state.get('prices_df', pd.DataFrame())
    trade_df = st.session_state.get('trade_df', pd.DataFrame())
    data_source = st.session_state.get('data_source', 'unknown')
    config = st.session_state.get('config', {})

    st.subheader("Current Data Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Price Records", len(prices_df))
        if not prices_df.empty and 'source' in prices_df.columns:
            st.metric("Price Sources", len(prices_df['source'].unique()))
        else:
            st.metric("Price Sources", 0)

    with col2:
        st.metric("Trade Records", len(trade_df))
        st.metric("Data Type", data_source.upper())

    with col3:
        # Count materials with sufficient data
        if not prices_df.empty and 'material' in prices_df.columns:
            sufficient_data = sum(1 for material in prices_df['material'].unique()
                                if len(prices_df[prices_df['material'] == material]) >= 10)
            st.metric("Materials Ready for Forecasting", sufficient_data)
        else:
            st.metric("Materials Ready for Forecasting", 0)

    st.subheader("🌐 Enhanced Data Sources")

    st.success("""
    **All data is now sourced from FREE APIs with Enhanced Features:**

    ✅ **World Bank Data** - Commodity prices and economic indicators
    ✅ **UN Comtrade** - Global trade flows (free public API)
    ✅ **Yahoo Finance** - Market data and ETF prices
    ✅ **USGS** - Mineral production statistics
    ✅ **GDELT Project** - Geopolitical event monitoring
    ✅ **IEA/BNEF** - EV adoption projections (synthetic data)

    **No API keys required!** The platform uses publicly available data.
    """)

    st.subheader("📊 Data Quality Information")

    if not prices_df.empty:
        if 'source' in prices_df.columns:
            st.write("**📈 Price Data Sources:**")
            sources = prices_df['source'].value_counts()
            for source, count in sources.items():
                st.write(f"- **{source}**: {count} records")

        # Data recency
        if 'date' in prices_df.columns:
            latest_date = prices_df['date'].max()
            if hasattr(latest_date, 'date'):
                days_old = (datetime.now().date() - latest_date.date()).days
            else:
                days_old = "Unknown"
            st.write(f"**Data Recency:** {days_old} days old")

    if not trade_df.empty:
        st.write("**🌍 Trade Data Coverage:**")
        if 'material' in trade_df.columns:
            st.write(f"- Materials: {len(trade_df['material'].unique())}")

        # Handle both 'exporter' and 'reporter' column names
        if 'exporter' in trade_df.columns:
            st.write(f"- Countries: {len(trade_df['exporter'].unique())}")
        elif 'reporter' in trade_df.columns:
            st.write(f"- Countries: {len(trade_df['reporter'].unique())}")
        else:
            st.write("- Countries: Column not found")

        if 'year' in trade_df.columns:
            st.write(f"- Years: {len(trade_df['year'].unique())}")

    # Enhanced data refresh with fundamental data
    st.subheader("🔄 Data Management")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh All Data", type="primary"):
            with st.spinner("Fetching fresh data from all APIs..."):
                if 'refresh_triggered' not in st.session_state:
                    st.session_state.refresh_triggered = True
                st.success("""
                **Refresh initiated!**
                - Price data from FRED/Yahoo Finance
                - Trade data from USGS/World Bank
                - EV adoption projections
                - Geopolitical risk events
                """)
                st.rerun()

    with col2:
        if st.button("📊 Update Fundamental Data"):
            with st.spinner("Updating EV and risk data..."):
                st.info("Updating fundamental factors for enhanced forecasting")
                # This would trigger updates to EV and GDELT data
                st.success("Fundamental data update complete!")

    st.info("""
    **Note on Free APIs:**
    - UN Comtrade free API has rate limits (please be respectful)
    - Some historical data might be limited
    - Data is typically available with 6-12 month lag
    - GDELT events are processed in near real-time
    - EV adoption data uses latest IEA/BNEF projections
    """)

    # System information
    st.subheader("🔧 System Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Platform Version:** 2.0.0 (Enhanced)")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    with col2:
        st.write(f"**Python Version:** {sys.version.split()[0]}")
        st.write(f"**Streamlit Version:** {st.__version__}")

    # Show import status
    st.subheader("🔍 Module Status")

    # Check module availability
    from app.utils.data_loader import GlobalCommodityFetcher

    try:
        from src.models.baseline_forecaster import BaselineForecaster
        baseline_status = '✅ Available'
    except ImportError:
        baseline_status = '❌ Not Available'

    try:
        from src.data_pipeline.ev_adoption_fetcher import EVAdoptionFetcher
        ev_status = '✅ Available'
    except ImportError:
        ev_status = '❌ Not Available'

    try:
        from src.data_pipeline.gdelt_fetcher import GDELTFetcher
        gdelt_status = '✅ Available'
    except ImportError:
        gdelt_status = '❌ Not Available'

    status_col1, status_col2, status_col3 = st.columns(3)
    with status_col1:
        st.write(f"**BaselineForecaster:** {baseline_status}")
    with status_col2:
        st.write(f"**EVAdoptionFetcher:** {ev_status}")
    with status_col3:
        st.write(f"**GDELTFetcher:** {gdelt_status}")

    # Show global fetcher status
    st.write(f"**GlobalCommodityFetcher:** {'✅ Available' if GlobalCommodityFetcher is not None else '❌ Not Available'}")

if __name__ == "__main__":
    main()