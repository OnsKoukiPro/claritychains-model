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

# Add src to Python path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Health check
def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

try:
    from data_pipeline.real_price_fetcher import RealPriceFetcher
    from data_pipeline.real_trade_fetcher import RealTradeFetcher
    from data_pipeline.usgs_minerals_fetcher import USGSMineralsFetcher
    from models.baseline_forecaster import BaselineForecaster
except ImportError as e:
    logger.warning(f"Import error: {e}")

def load_config():
    """Load configuration"""
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Config load error: {e}")
        return {
            'paths': {'data_dir': './data', 'raw_data': './data/raw'},
            'materials': {
                'lithium': {}, 'cobalt': {}, 'nickel': {}, 'copper': {}, 'rare_earths': {}
            }
        }

def fetch_real_data():
    """Fetch real data using reliable Python libraries"""
    config = load_config()

    # Initialize data fetchers
    price_fetcher = RealPriceFetcher(config)
    trade_fetcher = RealTradeFetcher(config)

    # Create data directory
    data_dir = Path(config['paths']['raw_data'])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Test data libraries first
    st.info("🔧 Testing Python data libraries...")
    lib_success, lib_message = trade_fetcher.test_data_availability()

    if lib_success:
        st.success(f"✅ {lib_message}")
    else:
        st.error(f"❌ {lib_message}")
        st.info("Using statistical data sources instead...")

    # Fetch price data
    st.info("📡 Fetching price data from FRED and market analysis...")
    prices_df = price_fetcher.fetch_all_prices()

    if not prices_df.empty:
        prices_df.to_csv(data_dir / "real_prices.csv", index=False)
        source_info = ", ".join(prices_df['source'].unique())
        st.success(f"✅ Loaded {len(prices_df)} price records from: {source_info}")
    else:
        st.error("❌ Could not load any price data")
        return pd.DataFrame(), pd.DataFrame(), "error"

    # Fetch trade data
    st.info("🌍 Fetching trade data from USGS and World Bank...")
    trade_df = trade_fetcher.fetch_simplified_trade_flows(years=[2023])

    if not trade_df.empty:
        trade_df.to_csv(data_dir / "real_trade_flows.csv", index=False)
        source_info = ", ".join(trade_df['source'].unique())
        st.success(f"✅ Loaded {len(trade_df)} trade records from: {source_info}")
    else:
        st.error("❌ Could not load any trade data")
        return pd.DataFrame(), pd.DataFrame(), "error"

    return prices_df, trade_df, "real"

def load_data():
    """Load data from files or fetch real data - NO SAMPLE DATA"""
    config = load_config()
    data_dir = Path(config['paths']['raw_data'])

    # Check if real data exists
    real_prices_path = data_dir / "real_prices.csv"
    real_trade_path = data_dir / "real_trade_flows.csv"

    if real_prices_path.exists() and real_trade_path.exists():
        prices_df = pd.read_csv(real_prices_path, parse_dates=['date'])
        trade_df = pd.read_csv(real_trade_path)
        return prices_df, trade_df, "real"
    else:
        # Only fetch real data - no sample fallback
        return fetch_real_data()

def main():
    st.set_page_config(
        page_title="Critical Materials AI Platform - Real Data",
        layout="wide",
        page_icon="🛡️"
    )

    st.title("🛡️ Critical Materials AI Platform")
    st.markdown("**Real-time procurement intelligence for critical minerals supply chains**")

    # Environment info
    if os.path.exists('/.dockerenv'):
        st.sidebar.success("🐳 Running in Docker container")

    # Data source selection
    st.sidebar.header("Data Sources")
    use_real_data = st.sidebar.checkbox("Use Real API Data", value=True)
    refresh_data = st.sidebar.button("Refresh Data from APIs")

    # Load data
    if refresh_data or use_real_data:
        with st.spinner("Fetching real data from APIs..."):
            prices_df, trade_df, data_source = fetch_real_data()
    else:
        prices_df, trade_df, data_source = load_data()

    # Display data source info
    if data_source == "real":
        st.sidebar.success("✅ Using Real API Data")
        st.sidebar.info("Sources: World Bank, UN Comtrade, Yahoo Finance")
    else:
        st.sidebar.warning("📊 Using Sample Data")
        st.sidebar.info("Enable 'Use Real API Data' for live data")

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Dashboard", "📈 Real Forecasting", "💳 Procurement",
        "🌍 Supply Chain", "⚙️ Data Sources"
    ])

    with tab1:
        show_live_dashboard(prices_df, trade_df, data_source)

    with tab2:
        show_real_forecasting(prices_df, data_source)

    with tab3:
        show_procurement_analysis(prices_df)

    with tab4:
        show_supply_chain_analysis(trade_df)

    with tab5:
        show_data_sources(config, prices_df, trade_df, data_source)

def show_live_dashboard(prices_df, trade_df, data_source):
    """Live dashboard with real data"""
    st.header("📊 Live Market Dashboard")

    if prices_df.empty:
        st.warning("No price data available")
        return

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        latest_prices = prices_df.groupby('material')['price'].last()
        avg_price = latest_prices.mean()
        st.metric("Average Price", f"${avg_price:,.0f}/t", "Live")

    with col2:
        price_change = prices_df.groupby('material').apply(
            lambda x: (x['price'].iloc[-1] - x['price'].iloc[-2]) / x['price'].iloc[-2] * 100
        ).mean()
        st.metric("Avg Daily Change", f"{price_change:+.1f}%")

    with col3:
        materials_count = len(prices_df['material'].unique())
        st.metric("Materials Tracked", materials_count)

    with col4:
        if not trade_df.empty:
            suppliers_count = len(trade_df['exporter'].unique())
            st.metric("Suppliers Monitored", suppliers_count)
        else:
            st.metric("Data Source", data_source)

    # Real-time price trends
    st.subheader("📈 Live Price Trends")

    # Show latest prices by material
    latest = prices_df.sort_values('date').groupby('material').tail(1)
    fig = px.bar(latest, x='material', y='price', color='material',
                title="Current Prices by Material")
    st.plotly_chart(fig, use_container_width=True)

    # Price history
    st.subheader("📊 Price History")
    fig = px.line(prices_df, x='date', y='price', color='material',
                 title="Historical Price Trends")
    st.plotly_chart(fig, use_container_width=True)

    # Data source info
    if not prices_df.empty:
        st.info(f"**Data Sources:** {', '.join(prices_df['source'].unique())}")

def show_real_forecasting(prices_df, data_source):
    """Real forecasting with live data"""
    st.header("📈 Real-time Price Forecasting")

    if prices_df.empty:
        st.warning("No price data available for forecasting")
        return

    material = st.selectbox("Select Material", prices_df['material'].unique())

    # Filter data for selected material
    material_data = prices_df[prices_df['material'] == material].copy()

    if len(material_data) < 10:
        st.warning(f"Not enough data for {material}. Need at least 10 data points.")
        return

    # Simple forecasting
    st.subheader(f"Forecast for {material.title()}")

    # Calculate basic statistics
    current_price = material_data['price'].iloc[-1]
    volatility = material_data['price'].pct_change().std() * np.sqrt(252)  # Annualized

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:,.0f}")
    with col2:
        st.metric("Volatility", f"{volatility*100:.1f}%")
    with col3:
        trend = "↑ Bullish" if material_data['price'].iloc[-1] > material_data['price'].iloc[-10] else "↓ Bearish"
        st.metric("Short-term Trend", trend)

    # Simple forecast visualization
    st.subheader("6-Month Forecast")

    # Generate simple forecast
    forecast_dates = pd.date_range(material_data['date'].max(), periods=7, freq='M')[1:]
    forecast_prices = []

    for i in range(6):
        # Simple projection based on recent trend
        recent_trend = material_data['price'].tail(30).pct_change().mean()
        forecast_price = current_price * (1 + recent_trend) ** (i + 1)
        forecast_prices.append(forecast_price)

    # Create forecast plot
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=material_data['date'], y=material_data['price'],
        name='Historical', line=dict(color='blue')
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates, y=forecast_prices,
        name='Forecast', line=dict(color='orange', dash='dash')
    ))

    fig.update_layout(title=f"{material.title()} Price Forecast")
    st.plotly_chart(fig, use_container_width=True)

    st.info(f"Forecast based on {data_source} data - {len(material_data)} data points")

def show_procurement_analysis(prices_df):
    """Procurement analysis with real data"""
    st.header("💳 Procurement Strategy")

    if prices_df.empty:
        st.warning("No price data available")
        return

    material = st.selectbox("Select Material for Procurement",
                           prices_df['material'].unique(), key='proc_material')

    material_data = prices_df[prices_df['material'] == material]

    st.subheader("Price Analysis")

    # Price statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${material_data['price'].iloc[-1]:,.0f}")
    with col2:
        st.metric("30-day Avg", f"${material_data['price'].tail(30).mean():,.0f}")
    with col3:
        st.metric("52-week High", f"${material_data['price'].max():,.0f}")

    # Procurement recommendation
    st.subheader("Procurement Recommendation")

    current_price = material_data['price'].iloc[-1]
    avg_30_day = material_data['price'].tail(30).mean()

    if current_price < avg_30_day * 0.9:
        st.success("**🟢 STRONG BUY** - Current price is below 30-day average")
        recommendation = "Consider increasing procurement volume"
    elif current_price < avg_30_day:
        st.info("**🟡 MODERATE BUY** - Price is reasonable")
        recommendation = "Proceed with planned procurement"
    else:
        st.warning("**🔴 CAUTION** - Price is above recent average")
        recommendation = "Consider delaying non-essential purchases"

    st.write(recommendation)

    # Hedging strategy
    st.subheader("Hedging Strategy")

    total_volume = st.number_input("Planned Volume (tonnes)", min_value=100, value=1000)
    risk_tolerance = st.select_slider("Risk Tolerance", options=["Low", "Medium", "High"])

    if st.button("Generate Hedging Plan"):
        if risk_tolerance == "Low":
            allocation = [0.4, 0.3, 0.2, 0.1]  # Front-loaded
            strategy = "Conservative - lock in prices early"
        elif risk_tolerance == "Medium":
            allocation = [0.25, 0.25, 0.25, 0.25]  # Equal
            strategy = "Balanced - average cost over time"
        else:
            allocation = [0.1, 0.2, 0.3, 0.4]  # Back-loaded
            strategy = "Aggressive - bet on lower future prices"

        plan_data = []
        for i, alloc in enumerate(allocation):
            plan_data.append({
                'Month': i + 1,
                'Allocation': f"{alloc*100:.0f}%",
                'Volume (tonnes)': int(total_volume * alloc),
                'Estimated Cost': f"${int(current_price * alloc * total_volume):,}",
                'Strategy': strategy
            })

        st.dataframe(pd.DataFrame(plan_data))

def show_supply_chain_analysis(trade_df):
    """Supply chain risk analysis"""
    st.header("🌍 Global Supply Chain Analysis")

    if trade_df.empty:
        st.warning("No trade data available")
        return

    material = st.selectbox("Select Material for Supply Chain",
                           trade_df['material'].unique(), key='supply_material')

    material_trade = trade_df[trade_df['material'] == material]

    if material_trade.empty:
        st.warning(f"No trade data for {material}")
        return

    # Calculate market concentration
    if 'value_usd' in material_trade.columns:
        total_value = material_trade['value_usd'].sum()
        supplier_shares = (material_trade.groupby('exporter')['value_usd'].sum() / total_value)

        # HHI calculation
        hhi = (supplier_shares ** 2).sum()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Supplier Concentration")
            fig = px.pie(supplier_shares, values=supplier_shares.values,
                        names=supplier_shares.index, title=f"{material.title()} Market Share")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Risk Assessment")
            st.metric("HHI Score", f"{hhi:.3f}")

            if hhi > 0.25:
                st.error("**HIGH RISK** - Market is highly concentrated")
                st.write("Consider diversifying suppliers immediately")
            elif hhi > 0.15:
                st.warning("**MEDIUM RISK** - Moderate concentration")
                st.write("Monitor supply chain and develop alternatives")
            else:
                st.success("**LOW RISK** - Market is diversified")
                st.write("Current supply chain appears resilient")

            # Top suppliers
            st.subheader("Top Suppliers")
            top_suppliers = supplier_shares.nlargest(5)
            for supplier, share in top_suppliers.items():
                st.write(f"- **{supplier}**: {share:.1%}")

def show_data_sources(config, prices_df, trade_df, data_source):
    """Data sources and configuration"""
    st.header("⚙️ Data Sources & Configuration")

    st.subheader("Current Data Status")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Price Records", len(prices_df))
        st.metric("Price Sources", len(prices_df['source'].unique()) if not prices_df.empty else 0)

    with col2:
        st.metric("Trade Records", len(trade_df))
        st.metric("Data Type", data_source.upper())

    st.subheader("🌐 Free Data Sources")

    st.success("""
    **All data is now sourced from FREE APIs:**

    ✅ **World Bank Data** - Commodity prices and economic indicators
    ✅ **UN Comtrade** - Global trade flows (free public API)
    ✅ **Yahoo Finance** - Market data and ETF prices
    ✅ **USGS** - Mineral production statistics

    **No API keys required!** The platform uses publicly available data.
    """)

    st.subheader("Data Quality Information")

    if not prices_df.empty:
        st.write("**📈 Price Data Sources:**")
        sources = prices_df['source'].value_counts()
        for source, count in sources.items():
            st.write(f"- **{source}**: {count} records")

    if not trade_df.empty:
        st.write("**🌍 Trade Data Coverage:**")
        st.write(f"- Materials: {len(trade_df['material'].unique())}")

        # Handle both 'exporter' and 'reporter' column names
        if 'exporter' in trade_df.columns:
            st.write(f"- Countries: {len(trade_df['exporter'].unique())}")
        elif 'reporter' in trade_df.columns:
            st.write(f"- Countries: {len(trade_df['reporter'].unique())}")
        else:
            st.write("- Countries: Column not found")

        st.write(f"- Years: {len(trade_df['year'].unique())}")

    # Data refresh
    st.subheader("Data Management")

    if st.button("🔄 Force Refresh All Data"):
        with st.spinner("Fetching fresh data from free APIs..."):
            # Use global to update the dataframes, or use session state
            # For now, we'll show a message and let user refresh manually
            st.session_state.refresh_triggered = True
            st.success("Refresh triggered! The app will reload with new data.")
            st.rerun()

    st.info("""
    **Note on Free APIs:**
    - UN Comtrade free API has rate limits (please be respectful)
    - Some historical data might be limited
    - Data is typically available with 6-12 month lag
    """)

if __name__ == "__main__":
    main()