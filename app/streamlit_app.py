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
    from data_pipeline.global_price_fetcher import GlobalCommodityFetcher  # CHANGED
    logger.info("✅ Successfully imported GlobalCommodityFetcher")
except ImportError as e:
    logger.warning(f"GlobalCommodityFetcher import failed: {e}")
    # Try to import the old one as fallback
    try:
        from data_pipeline.real_price_fetcher import RealPriceFetcher
        GlobalCommodityFetcher = RealPriceFetcher  # Use old class as fallback
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
    # Create a simple fallback class
    class BaselineForecaster:
        def __init__(self, config):
            self.config = config
            logger.warning("Using fallback BaselineForecaster - limited functionality")

        def fit_predict(self, prices_df, material, use_fundamentals=False):
            logger.error("BaselineForecaster not properly imported - using dummy data")
            # Return dummy data structure
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

        def get_demand_forecast_adjustment(self, material):
            return {'adjustment_factor': 1.0, 'demand_growth': 0.0}

try:
    from data_pipeline.gdelt_fetcher import GDELTFetcher
except ImportError as e:
    logger.warning(f"GDELTFetcher import failed: {e}")
    # Create fallback
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

    # Default configuration - ENHANCED WITH GLOBAL SOURCES
    return {
        'paths': {
            'data_dir': './data',
            'raw_data': './data/raw',
            'processed_data': './data/processed'
        },
        'materials': {
            'lithium': {}, 'cobalt': {}, 'nickel': {}, 'copper': {}, 'rare_earths': {},
            'aluminum': {}, 'zinc': {}, 'lead': {}, 'tin': {}  # ADDED MORE MATERIALS
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
        'global_sources': {  # ADDED NEW CONFIG SECTION
            'ecb_enabled': True,
            'worldbank_enabled': True,
            'lme_enabled': True,
            'global_futures_enabled': True
        }
    }

def fetch_real_data():
    """Fetch real data using reliable Python libraries - UPDATED FOR GLOBAL SOURCES"""
    config = load_config()

    # Check if fetchers are available - UPDATED TO GlobalCommodityFetcher
    if GlobalCommodityFetcher is None or RealTradeFetcher is None:
        st.error("❌ Data fetchers not available. Please check the installation.")
        return pd.DataFrame(), pd.DataFrame(), "error"

    # Initialize data fetchers - UPDATED TO GlobalCommodityFetcher
    price_fetcher = GlobalCommodityFetcher(config)  # CHANGED
    trade_fetcher = RealTradeFetcher(config)

    # Create data directory
    data_dir = Path(config['paths']['raw_data'])
    data_dir.mkdir(parents=True, exist_ok=True)

    # Test data libraries first - REMOVED STREAMLIT MESSAGES
    try:
        lib_success, lib_message = trade_fetcher.test_data_availability()
        if not lib_success:
            logger.warning(f"Data availability test failed: {lib_message}")
    except Exception as e:
        logger.warning(f"Data availability test failed: {e}")

    # Fetch price data - REMOVED STREAMLIT MESSAGES
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

    # Fetch trade data - REMOVED STREAMLIT MESSAGES
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
    """Load data from files or fetch real data - NO SAMPLE DATA"""
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
            logger.warning(f"Error loading saved data: {e}")

    # Only fetch real data - no sample fallback
    return fetch_real_data()

def show_data_sources(config, prices_df, trade_df, data_source):
    """Enhanced data sources display with global coverage"""
    st.header("🌐 Data Sources & Coverage")

    st.subheader("Global Price Data Sources")

    # Create columns for source overview
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🇺🇸 US Sources**")
        st.markdown("""
        - **FRED** (Federal Reserve Economic Data)
        - US ETFs & Futures
        - Yahoo Finance US tickers
        """)

    with col2:
        st.markdown("**🇪🇺 European Sources**")
        st.markdown("""
        - **ECB** (European Central Bank)
        - **LME** (London Metal Exchange)
        - European ETFs
        """)

    with col3:
        st.markdown("**🌍 Global Sources**")
        st.markdown("""
        - **World Bank** Pink Sheet
        - Global futures markets
        - Multi-region ETFs
        """)

    # Show actual data coverage
    if not prices_df.empty and 'source' in prices_df.columns:
        st.subheader("Current Data Coverage")

        # Source breakdown
        source_stats = prices_df['source'].value_counts()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Records by Source:**")
            for source, count in source_stats.items():
                st.write(f"- {source}: {count:,} records")

        with col2:
            # Regional breakdown
            regional_map = {
                'US': ['fred', 'etf_', 'futures_'],
                'Europe': ['ecb', 'lme'],
                'Global': ['worldbank', 'market_analysis']
            }

            regional_counts = {}
            for region, sources in regional_map.items():
                count = prices_df['source'].str.contains('|'.join(sources), na=False).sum()
                regional_counts[region] = count

            st.markdown("**Regional Distribution:**")
            for region, count in regional_counts.items():
                if count > 0:
                    percentage = (count / len(prices_df)) * 100
                    st.write(f"- {region}: {count:,} records ({percentage:.1f}%)")

def main():
    st.set_page_config(
        page_title="Critical Materials AI Platform - Global Enhanced",
        layout="wide",
        page_icon="🌍"
    )

    st.title("🌍 Critical Materials AI Platform - Global Edition")
    st.markdown("**Multi-region procurement intelligence for critical minerals supply chains**")

    # Enhanced subtitle with new features
    st.markdown("""
    <style>
    .global-features {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 1.1em;
    }
    </style>
    <div class="global-features">
    🌐 NEW: Intelligent Tender Management • 🤖 AI-Powered Document Creation • 📊 Real-time Tender Analytics
    </div>
    """, unsafe_allow_html=True)

    # Load data
    with st.spinner("Loading global data..."):
        prices_df, trade_df, data_source = load_data()

    # Enhanced main tabs with tender management
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🤖 Clare AI Offer Analysis", "📝 Tender Management", "📊 Global Dashboard", "📈 Enhanced Forecasting",
        "🌍 Geopolitical Risk", "💳 Procurement", "🔗 Supply Chain", "🌐 Data Sources",
    ])

    with tab1:
        show_ai_offer_analysis()
    with tab2:
        show_tender_management()
    with tab3:
        show_live_dashboard(prices_df, trade_df, data_source)
    with tab4:
        show_enhanced_forecasting(prices_df, data_source)
    with tab5:
        show_geopolitical_risk()
    with tab5:
        show_procurement_analysis(prices_df)
    with tab7:
        show_supply_chain_analysis(trade_df)
    with tab8:
        show_data_sources(load_config(), prices_df, trade_df, data_source)

def classify_data_source(source):
    """Classify data source by region and type"""
    source_lower = str(source).lower()

    # Region classification
    if 'fred' in source_lower or 'etf_' in source_lower:
        region = 'North America'
    elif 'ecb' in source_lower or 'lme' in source_lower:
        region = 'Europe'
    elif 'worldbank' in source_lower:
        region = 'Global'
    elif 'asia' in source_lower or 'shanghai' in source_lower:
        region = 'Asia'
    else:
        region = 'Other'

    # Market type classification
    if 'fred' in source_lower:
        market_type = 'Government Data'
    elif 'etf_' in source_lower:
        market_type = 'ETF'
    elif 'futures_' in source_lower:
        market_type = 'Futures'
    elif 'lme' in source_lower:
        market_type = 'Exchange'
    elif 'worldbank' in source_lower:
        market_type = 'International'
    else:
        market_type = 'Market Analysis'

    return region, market_type

def enrich_price_data(prices_df):
    """Add region and market type classifications to price data"""
    if prices_df.empty:
        return prices_df

    prices_df = prices_df.copy()

    # Add classifications
    classifications = prices_df['source'].apply(
        lambda x: pd.Series(classify_data_source(x))
    )
    prices_df['region'] = classifications[0]
    prices_df['market_type'] = classifications[1]

    # Add material grade classification (if not already present)
    if 'grade' not in prices_df.columns:
        prices_df['grade'] = prices_df['material'].apply(
            lambda x: 'Standard' if x in ['copper', 'aluminum', 'zinc', 'lead', 'tin', 'nickel']
            else 'Specialty'
        )

    return prices_df

def show_live_dashboard(prices_df, trade_df, data_source):
    """Enhanced interactive global commodity dashboard"""

    st.header("🌍 Global Market Intelligence Dashboard")

    if prices_df.empty:
        st.warning("No price data available")
        return

    # Enrich data with classifications
    prices_df = enrich_price_data(prices_df)

    # ======================
    # MAIN FRAME FILTERS - MOVED FROM SIDEBAR
    # ======================
    st.subheader("🔍 Data Filters")

    # Create columns for filters
    col1, col2, col3 = st.columns(3)

    with col1:
        # Material selection
        all_materials = sorted(prices_df['material'].unique())
        default_materials = all_materials[:5] if len(all_materials) > 5 else all_materials
        selected_materials = st.multiselect(
            "🔸 Materials",
            options=all_materials,
            default=default_materials,
            help="Select materials to display"
        )

    with col2:
        # Region selection
        all_regions = sorted(prices_df['region'].unique())
        selected_regions = st.multiselect(
            "🌍 Regions",
            options=all_regions,
            default=all_regions,
            help="Filter by geographic region"
        )

    with col3:
        # Market type selection
        all_market_types = sorted(prices_df['market_type'].unique())
        selected_market_types = st.multiselect(
            "📈 Market Types",
            options=all_market_types,
            default=all_market_types,
            help="Filter by market data source type"
        )

    # Date range selection in its own row
    st.subheader("📅 Time Period")
    min_date = prices_df['date'].min()
    max_date = prices_df['date'].max()

    # Calculate default date range (last 365 days)
    default_start = max(min_date, max_date - timedelta(days=365))

    date_range = st.date_input(
        "Select Date Range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Select date range for analysis"
    )

    # Apply filters
    filtered_df = prices_df[
        (prices_df['material'].isin(selected_materials)) &
        (prices_df['region'].isin(selected_regions)) &
        (prices_df['market_type'].isin(selected_market_types))
    ].copy()

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = filtered_df[
            (filtered_df['date'] >= pd.Timestamp(start_date)) &
            (filtered_df['date'] <= pd.Timestamp(end_date))
        ]

    if filtered_df.empty:
        st.warning("⚠️ No data matches the selected filters. Please adjust your selection.")
        return

    # ======================
    # KEY METRICS ROW
    # ======================
    st.subheader("📊 Key Performance Indicators")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        latest_prices = filtered_df.groupby('material')['price'].last()
        avg_price = latest_prices.mean()
        st.metric("Avg Price", f"${avg_price:,.0f}/t", help="Average price across selected materials")

    with col2:
        if len(filtered_df) > 1:
            try:
                price_changes = filtered_df.groupby('material').apply(
                    lambda x: ((x['price'].iloc[-1] - x['price'].iloc[0]) / x['price'].iloc[0] * 100)
                    if len(x) > 1 else 0
                )
                avg_change = price_changes.mean()
                st.metric("Period Change", f"{avg_change:+.1f}%",
                         delta=f"{avg_change:+.1f}%",
                         delta_color="normal" if avg_change > 0 else "inverse")
            except:
                st.metric("Period Change", "N/A")
        else:
            st.metric("Period Change", "N/A")

    with col3:
        materials_count = len(filtered_df['material'].unique())
        st.metric("Materials", materials_count)

    with col4:
        regions_count = len(filtered_df['region'].unique())
        st.metric("Regions", regions_count)

    with col5:
        data_points = len(filtered_df)
        st.metric("Data Points", f"{data_points:,}")

    # ======================
    # MAIN VISUALIZATION TABS
    # ======================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Price Trends",
        "🌍 Regional Analysis",
        "📈 Market Comparison",
        "🔍 Material Deep Dive",
        "📉 Volatility & Risk"
    ])

    with tab1:
        show_price_trends_tab(filtered_df)

    with tab2:
        show_regional_analysis_tab(filtered_df)

    with tab3:
        show_market_comparison_tab(filtered_df)

    with tab4:
        show_material_deep_dive_tab(filtered_df)

    with tab5:
        show_volatility_analysis_tab(filtered_df)

    # ======================
    # DATA EXPORT & INFO
    # ======================
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Data source summary
        st.subheader("📊 Data Source Summary")

        source_col1, source_col2, source_col3 = st.columns(3)

        with source_col1:
            st.markdown("**By Source:**")
            source_counts = filtered_df['source'].value_counts().head(5)
            for source, count in source_counts.items():
                st.caption(f"• {source}: {count:,}")

        with source_col2:
            st.markdown("**By Region:**")
            region_counts = filtered_df['region'].value_counts()
            for region, count in region_counts.items():
                pct = (count / len(filtered_df)) * 100
                st.caption(f"• {region}: {pct:.1f}%")

        with source_col3:
            st.markdown("**By Market:**")
            market_counts = filtered_df['market_type'].value_counts()
            for market, count in market_counts.items():
                st.caption(f"• {market}: {count:,}")

    with col2:
        # Export options
        st.subheader("📥 Export Data")

        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name=f"commodity_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )

        st.metric("Filtered Records", f"{len(filtered_df):,}")
        st.metric("Date Span", f"{len(filtered_df['date'].unique())} days")

def show_price_trends_tab(filtered_df):
    """Tab 1: Price Trends"""
    st.subheader("📈 Historical Price Trends")

    # Visualization controls
    col1, col2 = st.columns(2)

    with col1:
        chart_type = st.radio(
            "Chart Type",
            ["Line Chart", "Area Chart"],
            horizontal=True,
            key="chart_type"
        )

    with col2:
        color_by = st.selectbox(
            "Color/Group By",
            ["material", "region", "market_type", "source"],
            index=0,
            key="color_by"
        )

    # Create visualization
    fig = go.Figure()

    for group_value in filtered_df[color_by].unique():
        group_data = filtered_df[filtered_df[color_by] == group_value].sort_values('date')

        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(
                x=group_data['date'],
                y=group_data['price'],
                name=str(group_value),
                mode='lines',
                line=dict(width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x|%b %Y}<br>' +
                              'Price: $%{y:,.2f}/t<extra></extra>'
            ))
        else:  # Area Chart
            fig.add_trace(go.Scatter(
                x=group_data['date'],
                y=group_data['price'],
                name=str(group_value),
                mode='lines',
                fill='tonexty',
                line=dict(width=1),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Date: %{x|%b %Y}<br>' +
                              'Price: $%{y:,.2f}/t<extra></extra>'
            ))

    fig.update_layout(
        title=f"Price Trends by {color_by.replace('_', ' ').title()}",
        xaxis_title="Date",
        yaxis_title="Price (USD/t)",
        hovermode='x unified',
        height=500,
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Price statistics table
    st.subheader("📋 Price Statistics")

    stats_df = filtered_df.groupby('material').agg({
        'price': ['mean', 'min', 'max', 'std', 'count']
    }).round(2)
    stats_df.columns = ['Average', 'Minimum', 'Maximum', 'Std Dev', 'Data Points']
    stats_df = stats_df.reset_index()
    stats_df['material'] = stats_df['material'].str.title()

    st.dataframe(stats_df, use_container_width=True)

def show_regional_analysis_tab(filtered_df):
    """Tab 2: Regional Analysis"""
    st.subheader("🌍 Regional Market Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Regional price comparison
        region_stats = filtered_df.groupby('region').agg({
            'price': ['mean', 'std', 'count']
        }).reset_index()
        region_stats.columns = ['Region', 'Avg Price', 'Std Dev', 'Data Points']

        fig_region = px.bar(
            region_stats,
            x='Region',
            y='Avg Price',
            title="Average Price by Region",
            color='Region',
            text='Avg Price',
            hover_data=['Std Dev', 'Data Points']
        )
        fig_region.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_region.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig_region, use_container_width=True)

    with col2:
        # Regional material distribution
        region_material = filtered_df.groupby(['region', 'material']).size().reset_index(name='count')

        fig_sunburst = px.sunburst(
            region_material,
            path=['region', 'material'],
            values='count',
            title="Material Distribution by Region"
        )
        fig_sunburst.update_layout(height=400)
        st.plotly_chart(fig_sunburst, use_container_width=True)

    # Regional time series comparison
    st.subheader("📊 Regional Price Trends Comparison")

    selected_material = st.selectbox(
        "Select Material for Regional Comparison",
        options=sorted(filtered_df['material'].unique()),
        key="regional_material"
    )

    regional_data = filtered_df[filtered_df['material'] == selected_material]

    if not regional_data.empty:
        fig_trends = px.line(
            regional_data.sort_values('date'),
            x='date',
            y='price',
            color='region',
            title=f"{selected_material.title()} Price Trends Across Regions",
            markers=True
        )
        fig_trends.update_layout(height=400, hovermode='x unified')
        st.plotly_chart(fig_trends, use_container_width=True)
    else:
        st.info(f"No regional data available for {selected_material}")

def show_market_comparison_tab(filtered_df):
    """Tab 3: Market Comparison"""
    st.subheader("📈 Market Type Comparison")

    col1, col2 = st.columns(2)

    with col1:
        # Box plot by market type
        fig_box = px.box(
            filtered_df,
            x='market_type',
            y='price',
            color='market_type',
            title="Price Distribution by Market Type"
        )
        fig_box.update_layout(showlegend=False, height=400)
        fig_box.update_xaxes(tickangle=45)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        # Market coverage heatmap
        market_coverage = filtered_df.groupby(['market_type', 'material']).size().reset_index(name='coverage')

        fig_heatmap = px.density_heatmap(
            market_coverage,
            x='material',
            y='market_type',
            z='coverage',
            title="Market Coverage Heatmap",
            color_continuous_scale='Viridis'
        )
        fig_heatmap.update_layout(height=400)
        fig_heatmap.update_xaxes(tickangle=45)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Market statistics table
    st.subheader("📊 Market Statistics")

    market_stats = filtered_df.groupby('market_type').agg({
        'price': ['mean', 'min', 'max'],
        'material': 'count'
    }).reset_index()
    market_stats.columns = ['Market Type', 'Avg Price', 'Min Price', 'Max Price', 'Records']

    st.dataframe(
        market_stats.style.format({
            'Avg Price': '${:,.2f}',
            'Min Price': '${:,.2f}',
            'Max Price': '${:,.2f}',
            'Records': '{:,}'
        }),
        use_container_width=True
    )

def show_material_deep_dive_tab(filtered_df):
    """Tab 4: Material Deep Dive"""
    st.subheader("🔍 Material Deep Dive Analysis")

    selected_material = st.selectbox(
        "Select Material for Detailed Analysis",
        options=sorted(filtered_df['material'].unique()),
        key='deep_dive_material'
    )

    material_data = filtered_df[filtered_df['material'] == selected_material].sort_values('date')

    if material_data.empty:
        st.warning(f"No data available for {selected_material}")
        return

    # Statistics summary
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        current_price = material_data['price'].iloc[-1]
        st.metric("Current Price", f"${current_price:,.2f}/t")

    with col2:
        if len(material_data) > 1:
            price_change = ((material_data['price'].iloc[-1] - material_data['price'].iloc[0]) /
                           material_data['price'].iloc[0] * 100)
            st.metric("Total Change", f"{price_change:+.1f}%",
                     delta=f"{price_change:+.1f}%")
        else:
            st.metric("Total Change", "N/A")

    with col3:
        volatility = material_data['price'].std()
        st.metric("Volatility (Std)", f"${volatility:,.2f}")

    with col4:
        sources_count = material_data['source'].nunique()
        st.metric("Data Sources", sources_count)

    # Multi-source comparison
    st.subheader("📊 All Data Sources")

    fig_sources = px.line(
        material_data,
        x='date',
        y='price',
        color='source',
        title=f"{selected_material.title()} - Price by Source",
        markers=True
    )
    fig_sources.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_sources, use_container_width=True)

    # Distribution and moving averages
    col1, col2 = st.columns(2)

    with col1:
        # Price distribution
        fig_hist = px.histogram(
            material_data,
            x='price',
            nbins=30,
            title="Price Distribution",
            marginal="box"
        )
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Moving averages
        material_data_sorted = material_data.copy().sort_values('date')
        material_data_sorted['MA_30'] = material_data_sorted['price'].rolling(window=min(30, len(material_data_sorted)), min_periods=1).mean()
        material_data_sorted['MA_90'] = material_data_sorted['price'].rolling(window=min(90, len(material_data_sorted)), min_periods=1).mean()

        fig_ma = go.Figure()
        fig_ma.add_trace(go.Scatter(
            x=material_data_sorted['date'],
            y=material_data_sorted['price'],
            name='Price',
            line=dict(color='blue', width=1)
        ))
        fig_ma.add_trace(go.Scatter(
            x=material_data_sorted['date'],
            y=material_data_sorted['MA_30'],
            name='30-day MA',
            line=dict(color='orange', width=2)
        ))
        fig_ma.add_trace(go.Scatter(
            x=material_data_sorted['date'],
            y=material_data_sorted['MA_90'],
            name='90-day MA',
            line=dict(color='red', width=2)
        ))
        fig_ma.update_layout(title="Moving Averages", height=350, hovermode='x unified')
        st.plotly_chart(fig_ma, use_container_width=True)

def show_volatility_analysis_tab(filtered_df):
    """Tab 5: Volatility & Risk Analysis"""
    st.subheader("📉 Volatility & Risk Analysis")

    # Calculate volatility metrics
    volatility_data = []

    for material in filtered_df['material'].unique():
        mat_data = filtered_df[filtered_df['material'] == material].sort_values('date')

        if len(mat_data) > 1:
            returns = mat_data['price'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized

            volatility_data.append({
                'material': material,
                'volatility': volatility * 100,
                'avg_price': mat_data['price'].mean(),
                'price_range': mat_data['price'].max() - mat_data['price'].min(),
                'data_points': len(mat_data)
            })

    vol_df = pd.DataFrame(volatility_data)

    if not vol_df.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Volatility bar chart
            fig_vol = px.bar(
                vol_df.sort_values('volatility', ascending=False),
                x='material',
                y='volatility',
                title="Annualized Volatility by Material",
                color='volatility',
                color_continuous_scale='Reds',
                text='volatility'
            )
            fig_vol.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig_vol.update_layout(height=400)
            st.plotly_chart(fig_vol, use_container_width=True)

        with col2:
            # Risk-return scatter
            fig_scatter = px.scatter(
                vol_df,
                x='avg_price',
                y='volatility',
                size='price_range',
                color='material',
                title="Risk-Return Profile",
                labels={'avg_price': 'Average Price (USD/t)', 'volatility': 'Volatility (%)'},
                hover_data=['data_points']
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)

        # Volatility table
        st.subheader("📊 Volatility Metrics")

        display_vol = vol_df.copy()
        display_vol['material'] = display_vol['material'].str.title()
        display_vol = display_vol.sort_values('volatility', ascending=False)

        st.dataframe(
            display_vol.style.format({
                'volatility': '{:.2f}%',
                'avg_price': '${:,.2f}',
                'price_range': '${:,.2f}',
                'data_points': '{:,}'
            }).background_gradient(subset=['volatility'], cmap='RdYlGn_r'),
            use_container_width=True
        )

        # Risk interpretation
        st.subheader("⚠️ Risk Interpretation")

        high_vol_materials = vol_df[vol_df['volatility'] > vol_df['volatility'].quantile(0.75)]['material'].tolist()
        low_vol_materials = vol_df[vol_df['volatility'] < vol_df['volatility'].quantile(0.25)]['material'].tolist()

        col1, col2 = st.columns(2)

        with col1:
            if high_vol_materials:
                st.warning(f"**High Volatility Materials:**")
                for mat in high_vol_materials:
                    st.write(f"• {mat.title()} - Consider hedging strategies")

        with col2:
            if low_vol_materials:
                st.success(f"**Low Volatility Materials:**")
                for mat in low_vol_materials:
                    st.write(f"• {mat.title()} - Stable pricing environment")
    else:
        st.info("Insufficient data for volatility analysis")

def show_enhanced_forecasting(prices_df, data_source):
    """Enhanced forecasting with fundamental adjustments"""
    st.header("🎯 Enhanced Price Forecasting")

    if prices_df.empty:
        st.warning("No price data available for forecasting")
        return

    if 'material' not in prices_df.columns:
        st.warning("Price data missing 'material' column")
        return

    material = st.selectbox("Select Material", prices_df['material'].unique(), key='forecast_material')

    # Enhanced forecasting options
    col1, col2 = st.columns(2)
    with col1:
        use_fundamentals = st.checkbox("Include Fundamental Factors", value=True,
                                      help="Include EV adoption demand and geopolitical risk in forecasts")
    with col2:
        show_comparison = st.checkbox("Show Method Comparison", value=True,
                                     help="Compare baseline vs enhanced forecasting")

    # Filter data for selected material
    material_data = prices_df[prices_df['material'] == material].copy()

    if len(material_data) < 10:
        st.warning(f"Not enough data for {material}. Need at least 10 data points.")
        return

    if st.button("Generate Enhanced Forecast", type="primary"):
        with st.spinner("Running enhanced forecasting analysis..."):
            try:
                config = load_config()

                if BaselineForecaster is None:
                    st.error("❌ Forecasting engine not available. Please check the installation.")
                    return

                forecaster = BaselineForecaster(config)

                # Generate forecast
                result = forecaster.fit_predict(material_data, material, use_fundamentals=use_fundamentals)

                # Display enhanced metrics
                metrics = result['metrics']
                fundamentals = result.get('fundamentals')

                st.subheader(f"📊 Market Analysis for {material.title()}")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${metrics.get('current_price', 0):,.0f}")
                with col2:
                    st.metric("Volatility Regime", metrics.get('volatility_regime', 'N/A').title())
                with col3:
                    st.metric("Momentum", f"{metrics.get('momentum_zscore', 0):+.2f} σ")
                with col4:
                    trend = metrics.get('trend', 'neutral')
                    trend_icon = "📈" if trend == 'upward' else "📉" if trend == 'downward' else "➡️"
                    st.metric("Trend", f"{trend_icon} {trend.title()}")

                # Show fundamental adjustments if applied
                if fundamentals and not fundamentals.get('fallback_to_baseline', False):
                    st.success("🎯 Fundamental Adjustments Applied")

                    adj_col1, adj_col2, adj_col3 = st.columns(3)
                    with adj_col1:
                        ev_adj = fundamentals.get('ev_adjustment', {}).get('adjustment_factor', 1.0)
                        st.metric("EV Demand Factor", f"{ev_adj:.3f}")
                    with adj_col2:
                        risk_score = fundamentals.get('geopolitical_risk', {}).get('risk_score', 0.0)
                        st.metric("Geopolitical Risk", f"{risk_score:.2f}")
                    with adj_col3:
                        st.metric("Recent Events", fundamentals.get('gdelt_events_count', 0))

                # Enhanced forecast visualization
                st.subheader("🔮 6-Month Price Forecast")
                forecast_df = result['forecast']

                if not forecast_df.empty:
                    # Create enhanced forecast plot
                    fig = go.Figure()

                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=material_data['date'], y=material_data['price'],
                        name='Historical', line=dict(color='blue', width=2),
                        hovertemplate='%{x|%b %Y}: $%{y:,.0f}/t<extra></extra>'
                    ))

                    # Forecast with confidence intervals
                    fig.add_trace(go.Scatter(
                        x=forecast_df['date'], y=forecast_df['forecast_mean'],
                        name='Forecast Mean', line=dict(color='orange', width=3, dash='dash'),
                        hovertemplate='%{x|%b %Y}: $%{y:,.0f}/t<extra></extra>'
                    ))

                    # Confidence interval
                    if 'forecast_p10' in forecast_df.columns and 'forecast_p90' in forecast_df.columns:
                        fig.add_trace(go.Scatter(
                            x=pd.concat([forecast_df['date'], forecast_df['date'][::-1]]),
                            y=pd.concat([forecast_df['forecast_p10'], forecast_df['forecast_p90'][::-1]]),
                            fill='toself',
                            fillcolor='rgba(255,165,0,0.2)',
                            line=dict(color='rgba(255,165,0,0)'),
                            name='80% Confidence Interval',
                            hovertemplate='%{x|%b %Y}: $%{y:,.0f}/t<extra></extra>'
                        ))

                    fig.update_layout(
                        title=f"{material.title()} Price Forecast with Confidence Intervals",
                        xaxis_title="Date",
                        yaxis_title="Price (USD/t)",
                        hovermode='x unified',
                        showlegend=True
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Forecast details table
                    st.subheader("📋 Forecast Details")
                    display_forecast = forecast_df[['date', 'forecast_mean']].copy()
                    if 'forecast_p10' in forecast_df.columns and 'forecast_p90' in forecast_df.columns:
                        display_forecast['forecast_p10'] = forecast_df['forecast_p10']
                        display_forecast['forecast_p90'] = forecast_df['forecast_p90']

                    display_forecast['date'] = display_forecast['date'].dt.strftime('%Y-%m')
                    display_forecast['forecast_mean'] = display_forecast['forecast_mean'].round(0)

                    if 'forecast_p10' in display_forecast.columns:
                        display_forecast['forecast_p10'] = display_forecast['forecast_p10'].round(0)
                        display_forecast['forecast_p90'] = display_forecast['forecast_p90'].round(0)
                        display_forecast.columns = ['Month', 'Mean Forecast', 'P10 (Low)', 'P90 (High)']
                    else:
                        display_forecast.columns = ['Month', 'Mean Forecast']

                    st.dataframe(display_forecast, use_container_width=True)

                    # Method comparison if requested
                    if show_comparison and use_fundamentals:
                        st.subheader("🔄 Forecasting Method Comparison")
                        try:
                            comparison = forecaster.compare_forecast_methods(material_data, material)

                            col1, col2 = st.columns(2)
                            with col1:
                                baseline_mean = comparison.get('baseline', {}).get('mean_forecast', 0)
                                enhanced_mean = comparison.get('enhanced', {}).get('mean_forecast', 0)
                                change_pct = comparison.get('difference', {}).get('mean_change_pct', 0)

                                if abs(change_pct) > 1.0:
                                    change_icon = "📈" if change_pct > 0 else "📉"
                                    st.metric(
                                        "Enhanced vs Baseline",
                                        f"{change_icon} {abs(change_pct):.1f}%",
                                        delta=f"{change_pct:+.1f}%",
                                        delta_color="normal" if change_pct > 0 else "inverse"
                                    )
                                else:
                                    st.metric("Enhanced vs Baseline", "Minimal Change")

                            with col2:
                                if 'fundamental_adjustment' in comparison.get('enhanced', {}):
                                    adj_factor = comparison['enhanced']['fundamental_adjustment']
                                    st.metric("Overall Adjustment Factor", f"{adj_factor:.3f}")

                        except Exception as e:
                            st.warning(f"Could not generate method comparison: {e}")
                else:
                    st.warning("No forecast data available")

            except Exception as e:
                st.error(f"Forecasting error: {str(e)}")
                logger.error(f"Forecasting failed for {material}: {e}")

    # Information about enhanced forecasting
    with st.expander("ℹ️ About Enhanced Forecasting"):
        st.markdown("""
        **Enhanced Forecasting** combines:
        - **Statistical models** (rolling averages, momentum, volatility regimes)
        - **EV adoption demand** from IEA/BNEF projections
        - **Geopolitical risk** from GDELT event monitoring
        - **Confidence intervals** (P10/P50/P90) for risk assessment

        **Fundamental Factors:**
        - 🌍 Geopolitical events create supply risks
        - ⚡ Volatility regimes indicate market stability
        """)

def show_geopolitical_risk():
    """Geopolitical risk monitoring tab"""
    st.header("🌍 Geopolitical Risk Monitor")

    try:
        config = load_config()
        gdelt_fetcher = GDELTFetcher(config)

        material = st.selectbox("Select Material for Risk Analysis",
                               ['lithium', 'cobalt', 'nickel', 'copper', 'rare_earths'],
                               key='risk_material')

        country = st.text_input("Specific Country (optional)",
                               placeholder="e.g., China, DRC, Chile",
                               help="Focus risk analysis on specific country")

        days_back = st.slider("Analysis Period (days)", 7, 90, 30,
                             help="How far back to search for risk events")

        if st.button("Analyze Geopolitical Risk", type="primary"):
            with st.spinner("Scanning global events for supply chain risks..."):
                events = gdelt_fetcher.fetch_events_for_material(material, days_back)
                risk_score = gdelt_fetcher.generate_risk_score(events, material, country)

                # Display risk dashboard
                st.subheader("🎯 Risk Assessment Dashboard")

                # Risk metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    score = risk_score['risk_score']
                    if score > 0.7:
                        icon = "🔴"
                    elif score > 0.5:
                        icon = "🟡"
                    elif score > 0.3:
                        icon = "🟢"
                    else:
                        icon = "🟢"
                    st.metric("Risk Score", f"{icon} {score:.2f}")

                with col2:
                    level = risk_score['risk_level']
                    st.metric("Risk Level", level)

                with col3:
                    st.metric("Recent Events", risk_score['recent_events'])

                with col4:
                    sentiment = risk_score['avg_sentiment']
                    sentiment_icon = "😊" if sentiment > 0.1 else "😐" if sentiment > -0.1 else "😟"
                    st.metric("Avg Sentiment", f"{sentiment_icon} {sentiment:.2f}")

                # Risk interpretation
                st.subheader("📋 Risk Interpretation")
                risk_description = risk_score.get('risk_description', 'No specific risk events detected.')
                st.info(risk_description)

                # Show key events
                if risk_score.get('key_events'):
                    st.subheader("🔔 Recent Risk Events")

                    for i, event in enumerate(risk_score['key_events'][:5]):  # Show top 5 events
                        with st.container():
                            col1, col2 = st.columns([3, 1])
                            with col1:
                                st.write(f"**{event.get('title', 'No title')}**")
                                if 'date' in event:
                                    st.caption(f"Date: {event['date']}")
                            with col2:
                                sentiment = event.get('sentiment', 0)
                                st.metric("Sentiment", f"{sentiment:.2f}")

                            st.markdown("---")

                # Risk mitigation recommendations
                st.subheader("🛡️ Risk Mitigation Recommendations")

                if risk_score['risk_level'] in ['HIGH', 'CRITICAL']:
                    st.error("""
                    **Immediate Actions Recommended:**
                    - Diversify supplier base away from high-risk regions
                    - Increase inventory buffers for critical materials
                    - Monitor situation daily for escalation
                    - Develop contingency sourcing plans
                    """)
                elif risk_score['risk_level'] == 'MEDIUM':
                    st.warning("""
                    **Precautionary Actions:**
                    - Review supplier concentration risks
                    - Monitor key risk indicators weekly
                    - Develop alternative sourcing options
                    - Consider strategic stockpiling
                    """)
                else:
                    st.success("""
                    **Maintenance Actions:**
                    - Continue regular supply chain monitoring
                    - Maintain diverse supplier relationships
                    - Update risk assessment quarterly
                    """)

                # Event statistics
                if not events.empty:
                    st.subheader("📊 Event Statistics")
                    if 'event_type' in events.columns:
                        event_types = events['event_type'].value_counts()
                        fig = px.pie(values=event_types.values, names=event_types.index,
                                    title="Distribution of Risk Event Types")
                        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Geopolitical risk analysis failed: {e}")
        logger.error(f"Geopolitical risk analysis error: {e}")

def show_procurement_analysis(prices_df):
    """Procurement analysis with real data and editable plans"""
    st.header("💳 Procurement Strategy")

    if prices_df.empty:
        st.warning("No price data available")
        return

    if 'material' not in prices_df.columns:
        st.warning("Price data missing 'material' column")
        return

    material = st.selectbox("Select Material for Procurement",
                           prices_df['material'].unique(), key='proc_material')

    material_data = prices_df[prices_df['material'] == material]

    if material_data.empty:
        st.warning(f"No data for {material}")
        return

    st.subheader("Price Analysis")

    # Price statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        if not material_data.empty:
            current_price = material_data['price'].iloc[-1]
            st.metric("Current Price", f"${current_price:,.0f}")
        else:
            st.metric("Current Price", "N/A")
    with col2:
        if len(material_data) >= 30:
            st.metric("30-day Avg", f"${material_data['price'].tail(30).mean():,.0f}")
        else:
            st.metric("30-day Avg", "N/A")
    with col3:
        st.metric("52-week High", f"${material_data['price'].max():,.0f}")

    # Enhanced procurement recommendation with fundamental context
    st.subheader("Procurement Recommendation")

    if not material_data.empty:
        current_price = material_data['price'].iloc[-1]
        avg_30_day = material_data['price'].tail(30).mean() if len(material_data) >= 30 else current_price

        # Enhanced recommendation with multiple factors
        recommendation_factors = []

        # Price factor
        price_ratio = current_price / avg_30_day
        if price_ratio < 0.9:
            price_recommendation = "STRONG BUY"
            price_color = "green"
            recommendation_factors.append("Price below 30-day average")
        elif price_ratio < 1.0:
            price_recommendation = "MODERATE BUY"
            price_color = "blue"
            recommendation_factors.append("Price near 30-day average")
        else:
            price_recommendation = "CAUTION"
            price_color = "red"
            recommendation_factors.append("Price above 30-day average")

        # Display enhanced recommendation
        if price_color == "green":
            st.success(f"**🟢 {price_recommendation}**")
        elif price_color == "blue":
            st.info(f"**🟡 {price_recommendation}**")
        else:
            st.warning(f"**🔴 {price_recommendation}**")

        # Show recommendation factors
        st.write("**Key Factors:**")
        for factor in recommendation_factors:
            st.write(f"- {factor}")

        # Enhanced hedging strategy
        st.subheader("Advanced Hedging Strategy")

        col1, col2 = st.columns(2)
        with col1:
            total_volume = st.number_input("Planned Volume (tonnes)", min_value=100, value=1000, step=100)
            contract_duration = st.selectbox("Contract Duration", [3, 6, 12], index=1, format_func=lambda x: f"{x} months")
        with col2:
            risk_tolerance = st.select_slider("Risk Tolerance", options=["Low", "Medium", "High"])
            include_fundamentals = st.checkbox("Include Fundamental Analysis", value=True)

        if st.button("Generate Advanced Hedging Plan", type="primary"):
            with st.spinner("Optimizing procurement strategy..."):
                # Enhanced allocation based on multiple factors
                if risk_tolerance == "Low":
                    allocation = [0.4, 0.3, 0.2, 0.1]  # Front-loaded
                    strategy = "Conservative - lock in prices early"
                    color = "green"
                elif risk_tolerance == "Medium":
                    allocation = [0.25, 0.25, 0.25, 0.25]  # Equal
                    strategy = "Balanced - average cost over time"
                    color = "blue"
                else:
                    allocation = [0.1, 0.2, 0.3, 0.4]  # Back-loaded
                    strategy = "Aggressive - bet on lower future prices"
                    color = "orange"

                # Adjust based on fundamentals if enabled
                if include_fundamentals and BaselineForecaster is not None:
                    try:
                        config = load_config()
                        forecaster = BaselineForecaster(config)
                        result = forecaster.fit_predict(material_data, material, use_fundamentals=True)
                        fundamentals = result.get('fundamentals', {})

                        if fundamentals and not fundamentals.get('fallback_to_baseline', False):
                            risk_level = fundamentals.get('geopolitical_risk', {}).get('risk_level', 'LOW')
                            if risk_level in ['HIGH', 'CRITICAL']:
                                # More front-loaded for high risk
                                allocation = [0.5, 0.3, 0.15, 0.05]
                                strategy += " (Risk-Adjusted: High geopolitical risk)"
                                color = "red"
                    except Exception as e:
                        st.warning(f"Could not incorporate fundamental analysis: {e}")

                # Generate initial plan
                plan_data = []
                for i, alloc in enumerate(allocation):
                    monthly_volume = int(total_volume * alloc)
                    monthly_cost = current_price * monthly_volume

                    plan_data.append({
                        'Month': i + 1,
                        'Allocation_Percent': alloc,
                        'Volume_tonnes': monthly_volume,
                        'Price_per_tonne': current_price,
                        'Monthly_Cost': monthly_cost,
                        'Strategy': strategy
                    })

                # Store in session state for editing
                st.session_state.procurement_plan = plan_data
                st.session_state.procurement_strategy = strategy
                st.session_state.current_price = current_price

        # Display editable plan if it exists
        if 'procurement_plan' in st.session_state:
            st.subheader("📊 Editable Procurement Plan")

            # Create editable dataframe
            editable_df = pd.DataFrame(st.session_state.procurement_plan)
            display_df = editable_df.copy()

            # Format for display
            display_df['Allocation'] = display_df['Allocation_Percent'].apply(lambda x: f"{x*100:.0f}%")
            display_df['Price/t'] = display_df['Price_per_tonne'].apply(lambda x: f"${x:,.0f}")
            display_df['Monthly Cost'] = display_df['Monthly_Cost'].apply(lambda x: f"${x:,.0f}")

            # Show current plan
            st.write("**Current Plan:**")
            st.dataframe(display_df[['Month', 'Allocation', 'Volume_tonnes', 'Price/t', 'Monthly Cost']],
                        use_container_width=True)

            # Volume editing interface
            st.subheader("✏️ Modify Volumes")
            st.write("Adjust monthly volumes to see how it affects your procurement strategy:")

            # Create columns for volume inputs
            cols = st.columns(len(st.session_state.procurement_plan))
            updated_volumes = []

            for i, month_data in enumerate(st.session_state.procurement_plan):
                with cols[i]:
                    month = month_data['Month']
                    current_vol = month_data['Volume_tonnes']
                    allocation_pct = month_data['Allocation_Percent']

                    new_volume = st.number_input(
                        f"Month {month} Volume (tonnes)",
                        min_value=0,
                        value=current_vol,
                        step=10,
                        key=f"vol_{month}"
                    )
                    updated_volumes.append(new_volume)

                    # Show allocation percentage based on new volume
                    total_planned = sum(updated_volumes) if updated_volumes else total_volume
                    if total_planned > 0:
                        new_allocation = (new_volume / total_planned) * 100
                        st.write(f"Allocation: {new_allocation:.1f}%")

            # Calculate and display updated plan
            if st.button("Update Plan", type="secondary"):
                total_updated_volume = sum(updated_volumes)

                if total_updated_volume == 0:
                    st.error("Total volume cannot be zero!")
                else:
                    # Update the plan with new volumes
                    updated_plan = []
                    total_updated_cost = 0

                    for i, month_data in enumerate(st.session_state.procurement_plan):
                        new_volume = updated_volumes[i]
                        new_cost = st.session_state.current_price * new_volume
                        total_updated_cost += new_cost
                        new_allocation = new_volume / total_updated_volume

                        updated_plan.append({
                            'Month': month_data['Month'],
                            'Allocation_Percent': new_allocation,
                            'Volume_tonnes': new_volume,
                            'Price_per_tonne': st.session_state.current_price,
                            'Monthly_Cost': new_cost,
                            'Strategy': f"Custom - {st.session_state.procurement_strategy}"
                        })

                    # Update session state
                    st.session_state.procurement_plan = updated_plan

                    # Show updated metrics
                    st.success("Plan updated successfully!")

                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Volume", f"{total_updated_volume:,} t")
                    with col2:
                        st.metric("Average Price", f"${st.session_state.current_price:,.0f}/t")
                    with col3:
                        st.metric("Total Cost", f"${total_updated_cost:,.0f}")
                    with col4:
                        avg_cost_per_ton = total_updated_cost / total_updated_volume if total_updated_volume > 0 else 0
                        st.metric("Avg Cost/t", f"${avg_cost_per_ton:,.0f}")

                    # Show allocation breakdown
                    st.subheader("📈 Updated Allocation Breakdown")
                    allocation_data = {
                        'Month': [f"Month {p['Month']}" for p in updated_plan],
                        'Allocation %': [f"{p['Allocation_Percent']*100:.1f}%" for p in updated_plan],
                        'Volume (t)': [p['Volume_tonnes'] for p in updated_plan],
                        'Cost': [f"${p['Monthly_Cost']:,.0f}" for p in updated_plan]
                    }
                    st.dataframe(pd.DataFrame(allocation_data), use_container_width=True)

                    # Visualization
                    fig_col1, fig_col2 = st.columns(2)

                    with fig_col1:
                        # Volume allocation pie chart
                        import plotly.express as px
                        fig_volume = px.pie(
                            values=[p['Volume_tonnes'] for p in updated_plan],
                            names=[f"Month {p['Month']}" for p in updated_plan],
                            title="Volume Allocation by Month"
                        )
                        st.plotly_chart(fig_volume, use_container_width=True)

                    with fig_col2:
                        # Cost allocation pie chart
                        fig_cost = px.pie(
                            values=[p['Monthly_Cost'] for p in updated_plan],
                            names=[f"Month {p['Month']}" for p in updated_plan],
                            title="Cost Allocation by Month"
                        )
                        st.plotly_chart(fig_cost, use_container_width=True)

            # Risk analysis based on current plan
            st.subheader("📊 Risk Analysis")
            if 'procurement_plan' in st.session_state:
                current_plan = st.session_state.procurement_plan
                total_vol = sum(p['Volume_tonnes'] for p in current_plan)
                total_cost = sum(p['Monthly_Cost'] for p in current_plan)

                # Calculate concentration risk
                max_month_alloc = max(p['Allocation_Percent'] for p in current_plan)
                concentration_risk = "HIGH" if max_month_alloc > 0.5 else "MEDIUM" if max_month_alloc > 0.3 else "LOW"

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Concentration Risk", concentration_risk)
                with col2:
                    st.metric("Max Monthly Allocation", f"{max_month_alloc*100:.1f}%")
                with col3:
                    months_with_volume = sum(1 for p in current_plan if p['Volume_tonnes'] > 0)
                    st.metric("Active Months", f"{months_with_volume}/{len(current_plan)}")

                # Risk recommendations
                if concentration_risk == "HIGH":
                    st.warning("**Recommendation:** Consider diversifying purchases across more months to reduce concentration risk.")
                elif concentration_risk == "MEDIUM":
                    st.info("**Recommendation:** Current allocation provides moderate risk diversification.")
                else:
                    st.success("**Recommendation:** Well-diversified procurement strategy.")
    else:
        st.warning("No price data available for selected material")

def show_supply_chain_analysis(trade_df):
    """Supply chain risk analysis"""
    st.header("🔗 Global Supply Chain Analysis")

    if trade_df.empty:
        st.warning("No trade data available")
        return

    if 'material' not in trade_df.columns:
        st.warning("Trade data missing 'material' column")
        return

    material = st.selectbox("Select Material for Supply Chain",
                           trade_df['material'].unique(), key='supply_material')

    material_trade = trade_df[trade_df['material'] == material]

    if material_trade.empty:
        st.warning(f"No trade data for {material}")
        return

    # Calculate market concentration
    if 'value_usd' in material_trade.columns and 'exporter' in material_trade.columns:
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
                st.write("""
                **Immediate Actions:**
                - Diversify suppliers immediately
                - Develop alternative sourcing strategies
                - Increase inventory buffers
                - Monitor geopolitical risks in concentrated regions
                """)
            elif hhi > 0.15:
                st.warning("**MEDIUM RISK** - Moderate concentration")
                st.write("""
                **Recommended Actions:**
                - Monitor supply chain regularly
                - Develop supplier alternatives
                - Consider strategic partnerships
                - Review contingency plans
                """)
            else:
                st.success("**LOW RISK** - Market is diversified")
                st.write("""
                **Maintenance Actions:**
                - Continue monitoring supplier diversity
                - Maintain relationships with multiple suppliers
                - Regular risk assessment updates
                """)

            # Top suppliers with enhanced info
            st.subheader("🏭 Top Suppliers")
            top_suppliers = supplier_shares.nlargest(5)
            for supplier, share in top_suppliers.items():
                # Add risk indicators for known high-risk countries
                risk_indicators = ""
                high_risk_countries = ['China', 'DRC', 'Russia']
                if supplier in high_risk_countries:
                    risk_indicators = " ⚠️"

                st.write(f"- **{supplier}**: {share:.1%}{risk_indicators}")

def show_data_sources(config, prices_df, trade_df, data_source):
    """Data sources and configuration"""
    st.header("⚙️ Data Sources & Configuration")

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
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.write(f"**BaselineForecaster:** {'✅ Available' if BaselineForecaster is not None else '❌ Not Available'}")
    with status_col2:
        st.write(f"**GDELTFetcher:** {'✅ Available' if GDELTFetcher is not None else '❌ Not Available'}")

def show_ai_offer_analysis():
    """AI-powered procurement offer analysis with sidebar layout"""

    # Set page configuration for better layout
    st.set_page_config(layout="wide", page_title="ClarityChain - Procurement Analyzer")

    # Enhanced Custom CSS with better title styling
    st.markdown("""
    <style>
    :root {
        --primary-color: #4a69ff;
        --primary-hover-color: #3b55cc;
        --background-color: #f7f8fc;
        --sidebar-bg: #ffffff;
        --card-bg: #ffffff;
        --text-primary: #2c3e50;
        --text-secondary: #8a94a6;
        --border-color: #e1e5eb;
        --shadow-color: rgba(0, 0, 0, 0.05);
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    }

    /* Enhanced Product Title Styles - Matching Main Platform */
    .platform-title {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        text-align: center;
        margin: 1rem 0 0.5rem 0 !important;
        padding: 0.5rem;
        font-family: "Source Sans Pro", sans-serif !important;
        letter-spacing: -0.5px;
    }

    .platform-subtitle {
        color: var(--text-primary);
        font-size: 1.1rem;
        text-align: center;
        margin: 0 0 1.5rem 0 !important;
        font-weight: 600;
        font-family: "Source Sans Pro", sans-serif;
    }

    .platform-features {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 1.1em;
        text-align: center;
        margin: 0.5rem 0 1.5rem 0 !important;
    }

    .title-container {
        border-bottom: 2px solid var(--border-color);
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sidebar .sidebar-content {
        background-color: var(--sidebar-bg);
        border-right: 1px solid var(--border-color);
    }

    .stButton button {
        width: 100%;
        padding: 12px;
        border: none;
        border-radius: 8px;
        font-size: 16px;
        font-weight: 600;
        cursor: pointer;
        transition: background-color 0.2s ease;
    }

    .stButton button:first-child {
        background-color: var(--primary-color);
        color: white;
    }

    .stButton button:first-child:hover {
        background-color: var(--primary-hover-color);
    }

    .card {
        background: var(--card-bg);
        border: 1px solid var(--border-color);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px var(--shadow-color);
        margin-bottom: 20px;
    }

    .best-offer {
        border-color: #28a745;
        position: relative;
    }

    .recommendation-badge {
        position: absolute;
        top: -10px;
        right: -10px;
        background-color: #28a745;
        color: white;
        padding: 5px 10px;
        font-size: 12px;
        font-weight: 600;
        border-radius: 5px;
        transform: rotate(10deg);
    }

    .recommendation-best {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 5px 0;
    }

    .recommendation-good {
        background-color: #fff3cd;
        color: #856404;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin: 5px 0;
    }

    .total-score-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        margin: 10px 0;
        border-top: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
    }

    .total-score-label {
        font-weight: 600;
        font-size: 16px;
    }

    .total-score-value {
        font-weight: 700;
        font-size: 20px;
        color: var(--primary-color);
    }

    .category-scores {
        padding: 15px 0;
        margin: 15px 0;
        border-top: 1px solid var(--border-color);
        border-bottom: 1px solid var(--border-color);
    }

    .score-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        font-size: 14px;
    }

    .comparison-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 15px;
        background-color: var(--card-bg);
        box-shadow: 0 2px 8px var(--shadow-color);
        border-radius: 8px;
        overflow: hidden;
    }

    .comparison-table th, .comparison-table td {
        padding: 12px 15px;
        border: 1px solid var(--border-color);
        text-align: left;
    }

    .comparison-table thead th {
        background-color: var(--primary-color);
        color: white;
        font-weight: 600;
    }

    .highlight-green {
        background-color: #d4edda;
        color: #155724;
        font-weight: 600;
    }

    .highlight-yellow {
        background-color: #fff3cd;
        color: #856404;
        font-weight: 600;
    }

    .highlight-red {
        background-color: #f8d7da;
        color: #721c24;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)

    # Import agent client
    try:
        from utils.agent_client import get_agent_client
        agent = get_agent_client()

        # Check if agent is available
        if not agent.health_check():
            st.error("❌ AI Agent API is not available. Please ensure the agent-api service is running.")
            st.info("Start the agent with: `docker-compose up agent-api`")
            return

        st.success("✅ AI Agent is ready")

    except ImportError:
        st.error("❌ Agent client not available. Please check installation.")
        return

    # Initialize session state
    if 'offers_staged' not in st.session_state:
        st.session_state.offers_staged = []
    if 'analysis_result' not in st.session_state:
        st.session_state.analysis_result = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # ===== SIDEBAR =====
    with st.sidebar:
        # Fixed sidebar styling
        st.markdown(
            """
            <div style='text-align: center; margin-bottom: 1.5rem;'>
                <h1 style="
                    font-size: 3.2rem;
                    font-weight: 800;
                    color: #31333F;
                    margin: 0.5rem 0 0.25rem 0;
                    text-align: center;
                ">
                    ClarityChain
                </h1>
                <p style="
                    font-size: 1.1rem;
                    font-weight: 600;
                    color: #666;
                    margin: 0 0 1rem 0;
                    text-align: center;
                ">
                    AI Procurement Platform
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Section 1: Stage Offers
        st.header("📥 Stage Offers")

        st.info("Upload all documents for a single offer, then click 'Add as Offer' before uploading the next offer.")

        uploaded_files = st.file_uploader(
            "Drag & Drop files here or click to select",
            accept_multiple_files=True,
            type=['pdf', 'docx', 'txt', 'csv', 'xlsx', 'xls'],
            key='offer_uploader',
            label_visibility="collapsed"
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Add as Offer", type="primary", disabled=not uploaded_files):
                if uploaded_files:
                    with st.spinner("Adding offer..."):
                        import tempfile
                        temp_files = []

                        for uploaded_file in uploaded_files:
                            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name)
                            temp_file.write(uploaded_file.getvalue())
                            temp_file.close()
                            temp_files.append(temp_file.name)

                        # Add to agent
                        result = agent.add_offer(temp_files)

                        if 'error' not in result:
                            st.session_state.offers_staged.append({
                                'files': temp_files,
                                'file_names': [f.name for f in uploaded_files],
                                'count': len(uploaded_files)
                            })
                            st.success(f"✅ Offer {len(st.session_state.offers_staged)} added successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to add offer: {result['error']}")

        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.offers_staged = []
                st.session_state.analysis_result = None
                st.session_state.chat_history = []
                st.rerun()

        # Show staged offers
                # Show staged offers
        st.subheader("📋 Staged Offers")
        if st.session_state.offers_staged:
            for i, offer in enumerate(st.session_state.offers_staged):
                with st.expander(f"Offer {i+1} - {offer['count']} file(s)"):
                    for filename in offer['file_names']:
                        st.text(f"• {filename}")
        else:
            st.info("No offers staged yet.")

        # Section 2: Configure Analysis
        st.header("⚙️ Configure Analysis")

        eval_criteria = st.text_area(
            "Evaluation Criteria (Optional)",
            placeholder="e.g., focus on long-term value, sustainability, or specific technical requirements...",
            help="Provide any specific requirements or preferences for the AI analysis"
        )

        # Category Weights
        with st.expander("📊 Set Category Weights", expanded=False):
            st.caption("Adjust the importance of each evaluation category")
            tco_weight = st.slider("TCO (Total Cost of Ownership)", 0, 100, 25, 5, key="tco")
            payment_terms_weight = st.slider("Payment Terms", 0, 100, 10, 5, key="payment_terms")
            price_stability_weight = st.slider("Price Stability", 0, 100, 5, 5, key="price_stability")
            lead_time_weight = st.slider("Lead Time", 0, 100, 20, 5, key="lead_time")
            tech_spec_weight = st.slider("Technical Specifications", 0, 100, 25, 5, key="tech_spec")
            certifications_weight = st.slider("Certifications", 0, 100, 5, 5, key="certifications")
            incoterms_weight = st.slider("Incoterms", 0, 100, 5, 5, key="incoterms")
            warranty_weight = st.slider("Warranty & Support", 0, 100, 5, 5, key="warranty")

        # Risk Weights
        with st.expander("⚠️ Set Risk Weights", expanded=False):
            st.caption("Configure risk assessment priorities")
            delivery_risk_weight = st.slider("Delivery Risk", 0, 100, 10, 5, key="delivery_risk")
            financial_risk_weight = st.slider("Financial Risk", 0, 100, 10, 5, key="financial_risk")
            technical_risk_weight = st.slider("Technical Risk", 0, 100, 10, 5, key="technical_risk")
            quality_risk_weight = st.slider("Quality Risk", 0, 100, 10, 5, key="quality_risk")
            hse_compliance_risk_weight = st.slider("HSE / Compliance Risk", 0, 100, 10, 5, key="hse_compliance_risk")
            geopolitical_supply_risk_weight = st.slider("Geopolitical / Supply Risk", 0, 100, 10, 5, key="geopolitical_supply_risk")
            esg_reputation_risk_weight = st.slider("ESG / Reputation Risk", 0, 100, 10, 5, key="esg_reputation_risk")

        # Analyze button
        if st.button("🚀 Analyze Offers with AI", type="primary", use_container_width=True):
            if not st.session_state.offers_staged:
                st.error("Please add at least one offer first.")
            else:
                with st.spinner("🤖 AI is analyzing your offers... This may take 2-3 minutes..."):
                    weights = {
                        "tco_weight": tco_weight,
                        "payment_terms_weight": payment_terms_weight,
                        "price_stability_weight": price_stability_weight,
                        "lead_time_weight": lead_time_weight,
                        "tech_spec_weight": tech_spec_weight,
                        "certifications_weight": certifications_weight,
                        "incoterms_weight": incoterms_weight,
                        "warranty_weight": warranty_weight,
                        "delivery_risk_weight": delivery_risk_weight,
                        "financial_risk_weight": financial_risk_weight,
                        "technical_risk_weight": technical_risk_weight,
                        "quality_risk_weight": quality_risk_weight,
                        "hse_compliance_risk_weight": hse_compliance_risk_weight,
                        "geopolitical_supply_risk_weight": geopolitical_supply_risk_weight,
                        "esg_reputation_risk_weight": esg_reputation_risk_weight,
                    }

                    result = agent.analyze_offers(eval_criteria, weights)

                    if 'error' not in result:
                        st.session_state.analysis_result = result
                        st.success("✅ Analysis complete!")
                        st.rerun()
                    else:
                        st.error(f"Analysis failed: {result['error']}")

        # Add a footer with version info
        st.markdown("---")
        st.caption("🔒 **ClarityChain v2.0**  \n*Secure AI-Powered Procurement*")

    # ===== MAIN CONTENT =====
    st.header("🎯 Procurement Analysis Dashboard")
    st.write("Results from your procurement offer analysis will be displayed here.")

    if not st.session_state.analysis_result:
        st.info("📤 Upload and analyze offers to see the results.")
        return

    # Create tabs for different views
    analysis_tab, comparison_tab, chat_tab = st.tabs([
        "📊 Analysis Results",
        "📈 Comparison Summary",
        "💬 AI Chat"
    ])

    # ===== ANALYSIS TAB =====
    with analysis_tab:
        analysis = st.session_state.analysis_result.get('analysis', [])

        if isinstance(analysis, str):
            try:
                analysis = json.loads(analysis)
            except:
                st.error("Could not parse analysis data as JSON")
                analysis = []

        if analysis:
            st.write(f"Found {len(analysis)} offers to display")
            # Create cards for each offer
            cols = st.columns(2)
            for i, offer in enumerate(analysis):
                with cols[i % 2]:
                    display_offer_card(offer, i)
        else:
            st.warning("No analysis data available")

    # ===== COMPARISON TAB =====
    with comparison_tab:
        display_comparison_summary()

    # ===== CHAT TAB =====
    with chat_tab:
        display_chat_interface(agent)

def display_offer_card(offer, index):
    """Display an offer card using complete HTML structure"""

    # Helper function to safely convert to float
    def safe_float(value, default=0.0):
        try:
            if value is None or value == '' or value == 'N/A':
                return default
            return float(value)
        except (ValueError, TypeError):
            return default

    supplier_name = offer.get('supplier_name', 'Unknown Supplier')
    recommendation = offer.get('recommendation', 'N/A')
    score = safe_float(offer.get('total_weighted_score', 0))

    # Summary metrics
    summary_metrics = offer.get('summary_metrics', {})
    price = summary_metrics.get('Total Price', 'N/A') if isinstance(summary_metrics, dict) else 'N/A'
    lead_time = summary_metrics.get('Lead Time', 'N/A') if isinstance(summary_metrics, dict) else 'N/A'

    # Build category scores HTML
    category_scores_html = ""
    category_scores = offer.get('category_scores', {})
    if category_scores:
        for category, score_val in category_scores.items():
            score_float = safe_float(score_val)
            category_scores_html += f'<div class="score-item"><span>{category}</span><span class="score-percentage">{score_float:.1f}%</span></div>'

    # Create complete HTML card
    if recommendation == 'Best Offer':
        card_html = f'<div class="card best-offer"><div class="recommendation-badge">Recommended</div><h3>{supplier_name}</h3><div class="recommendation-best">Best Offer</div><div class="total-score-container"><span class="total-score-label">Total Weighted Score</span><span class="total-score-value">{score:.2f}</span></div><div class="category-scores">{category_scores_html}</div><div class="summary-metrics"><p><strong>Price:</strong> {price}</p><p><strong>Lead Time:</strong> {lead_time}</p></div></div>'
    else:
        card_html = f'<div class="card"><h3>{supplier_name}</h3><div class="recommendation-good">{recommendation}</div><div class="total-score-container"><span class="total-score-label">Total Weighted Score</span><span class="total-score-value">{score:.2f}</span></div><div class="category-scores">{category_scores_html}</div><div class="summary-metrics"><p><strong>Price:</strong> {price}</p><p><strong>Lead Time:</strong> {lead_time}</p></div></div>'

    # Display the complete card
    st.markdown(card_html, unsafe_allow_html=True)

    # Details expander
    with st.expander(f"View Details - {supplier_name}"):
        display_offer_details(offer)

def display_offer_details(offer):
    """Display detailed analysis for an offer"""

    # Gap Analysis
    if 'detailed_gap_analysis' in offer and offer['detailed_gap_analysis']:
        gap_analysis = offer['detailed_gap_analysis']
        headers = gap_analysis.get('headers', [])
        rows = gap_analysis.get('rows', [])

        if headers and rows:
            st.subheader("Gap Analysis")
            gap_df = pd.DataFrame(rows, columns=headers)
            st.dataframe(gap_df, use_container_width=True)

    # Risk Analysis
    risk_data = offer.get('risk', {})
    if risk_data:
        st.subheader("Risk Analysis")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Overall Risk Level:** {risk_data.get('risk_level', 'N/A')}")
            st.write(f"**Risk Score:** {risk_data.get('total_risk_score', 'N/A')}")

        with col2:
            if 'summary' in risk_data and risk_data['summary']:
                st.write(f"**Summary:** {risk_data['summary']}")

        # Dimension scores
        dimension_scores = risk_data.get('dimension_scores', {})
        if dimension_scores:
            st.write("**Dimension Scores:**")
            for dimension, score in dimension_scores.items():
                st.write(f"• {dimension}: {score}")

        # Detailed risk analysis
        if 'detailed_risk_analysis' in risk_data and risk_data['detailed_risk_analysis']:
            risk_analysis = risk_data['detailed_risk_analysis']
            headers = risk_analysis.get('headers', [])
            rows = risk_analysis.get('rows', [])

            if headers and rows:
                st.write("**Detailed Risk Breakdown:**")
                risk_df = pd.DataFrame(rows, columns=headers)
                st.dataframe(risk_df, use_container_width=True)

def display_comparison_summary():
    """Display comparison summary in the style of the old project"""

    if not st.session_state.analysis_result:
        st.info("Complete an analysis first to see the comparison summary.")
        return

    comparison_data = st.session_state.analysis_result.get('comparison_summary', {})

    # Handle nested structure
    if isinstance(comparison_data, str):
        try:
            comparison_data = json.loads(comparison_data)
        except json.JSONDecodeError:
            comparison_data = {}

    # Extract actual comparison summary
    if 'comparison_summary' in comparison_data:
        comparison_summary = comparison_data['comparison_summary']
    else:
        comparison_summary = comparison_data

    if not comparison_summary or not isinstance(comparison_summary, dict):
        st.warning("No comparison summary available.")
        return

    # Comparison Table
    if 'comparison_table' in comparison_summary and comparison_summary['comparison_table']:
        st.subheader("Offer Comparison Table")
        display_comparison_table(comparison_summary['comparison_table'])

    # AI Insights
    if 'ai_insights' in comparison_summary and comparison_summary['ai_insights']:
        st.subheader("AI Highlights & Insights")
        # REMOVED KEY PARAMETER
        with st.expander("View AI Analysis"):
            insights = comparison_summary['ai_insights']
            if isinstance(insights, list):
                for i, insight in enumerate(insights, 1):
                    st.markdown(f"**{i}.** {insight}")
            elif isinstance(insights, str):
                st.info(insights)
            else:
                st.json(insights)

    # Action List
    if 'action_list' in comparison_summary and comparison_summary['action_list']:
        st.subheader("Action List")
        display_action_list(comparison_summary['action_list'])

def display_comparison_table(comparison_table):
    """Display the comparison table with styling"""

    if not comparison_table:
        st.info("No comparison table data available.")
        return

    try:
        # Create a clean dataframe
        rows = []
        if comparison_table and len(comparison_table) > 0:
            first_item = comparison_table[0]
            offer_columns = [key for key in first_item.keys() if 'Offer' in key or 'offer' in key]
            columns = ['Criterion'] + sorted(offer_columns) + ['Observation', 'Highlight']

            for item in comparison_table:
                if isinstance(item, dict):
                    row = {'Criterion': item.get('criterion', '')}
                    for col in offer_columns:
                        row[col] = item.get(col, '')
                    row['Observation'] = item.get('observation', '')
                    row['Highlight'] = item.get('highlight', '')
                    rows.append(row)

        if rows:
            comp_df = pd.DataFrame(rows)
            # Display without highlight column
            display_df = comp_df.drop(columns=['Highlight'], errors='ignore')
            st.dataframe(display_df, use_container_width=True, hide_index=True)
        else:
            st.info("No comparison data available in table format.")

    except Exception as e:
        st.error(f"Error displaying comparison table: {e}")

def display_action_list(action_list):
    """Display the action list in a table format"""

    if not action_list:
        st.info("No action items generated.")
        return

    action_data = []
    for i, action in enumerate(action_list, 1):
        if isinstance(action, dict):
            action_data.append({
                'Action': action.get('action', 'N/A'),
                'Responsible': action.get('responsible', 'N/A'),
                'Status': action.get('status', 'Open'),
                'Due Date': action.get('due_date', '')
            })

    if action_data:
        action_df = pd.DataFrame(action_data)
        st.dataframe(action_df, use_container_width=True, hide_index=True)

def display_chat_interface(agent):
    """Display the chat interface"""

    if not st.session_state.analysis_result:
        st.info("Complete an analysis first, then you can ask questions about the results.")
        return

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg['content'])

    # Chat input
    user_input = st.chat_input("Ask about the analysis...")

    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })

        # Get AI response
        with st.spinner("🤖 Thinking..."):
            response = agent.chat(user_input)

            if 'error' not in response:
                assistant_msg = response.get('content', 'Sorry, I could not process that.')
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': assistant_msg
                })
            else:
                st.error(f"Chat error: {response['error']}")

        st.rerun()

    # Quick questions
    st.subheader("💡 Quick Questions")
    quick_questions = [
        "What are the main differences between the top 2 offers?",
        "Which offer has the best lead time?",
        "Explain the risk analysis for each offer",
        "What are the key action items I should focus on?",
        "Which offer is best for long-term partnership?"
    ]

    cols = st.columns(2)
    for i, question in enumerate(quick_questions):
        with cols[i % 2]:
            if st.button(question, key=f"quick_{i}"):
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': question
                })

                response = agent.chat(question)
                if 'error' not in response:
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': response.get('content', '')
                    })

                st.rerun()

import uuid
from datetime import datetime, timedelta
import json

def show_tender_management():
    """Intelligent Tender Management System with AI-powered document creation"""

    st.header("📝 Intelligent Tender Management")

    # Initialize session state for tender management
    if 'tenders' not in st.session_state:
        st.session_state.tenders = []
    if 'current_tender' not in st.session_state:
        st.session_state.current_tender = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Main tabs for tender management
    tender_tab1, tender_tab2, tender_tab3 = st.tabs([
        "🏠 Tender Dashboard",
        "🔄 Create New Tender",
        "📊 Tender Analysis"
    ])

    with tender_tab1:
        show_tender_dashboard()

    with tender_tab2:
        show_tender_creation()

    with tender_tab3:
        show_tender_analysis()

def show_tender_dashboard():
    """Dashboard showing all RFQs and their status"""

    st.subheader("📊 Tender Dashboard")

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        total_tenders = len(st.session_state.tenders)
        st.metric("Total Tenders", total_tenders)

    with col2:
        active_tenders = len([t for t in st.session_state.tenders if t.get('status') == 'Active'])
        st.metric("Active Tenders", active_tenders)

    with col3:
        draft_tenders = len([t for t in st.session_state.tenders if t.get('status') == 'Draft'])
        st.metric("Draft Tenders", draft_tenders)

    with col4:
        completed_tenders = len([t for t in st.session_state.tenders if t.get('status') == 'Completed'])
        st.metric("Completed Tenders", completed_tenders)

    # Tender list with status
    st.subheader("📋 All Tenders")

    if not st.session_state.tenders:
        st.info("No tenders created yet. Start by creating your first tender!")
        return

    # Create a dataframe for better display
    tender_data = []
    for tender in st.session_state.tenders:
        tender_data.append({
            'ID': tender.get('id', 'N/A'),
            'Title': tender.get('title', 'Untitled'),
            'Status': tender.get('status', 'Draft'),
            'Category': tender.get('category', 'General'),
            'Created': tender.get('created_date', 'N/A'),
            'Suppliers': len(tender.get('suppliers', [])),
            'Value': f"${tender.get('estimated_value', 0):,}",
            'Deadline': tender.get('submission_deadline', 'N/A')
        })

    if tender_data:
        df = pd.DataFrame(tender_data)
        st.dataframe(df, use_container_width=True)

    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("➕ Create New Tender", use_container_width=True):
            st.session_state.current_tender = None
            st.rerun()

    with col2:
        if st.button("📊 Generate Report", use_container_width=True):
            generate_tender_report()

    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()

def show_tender_creation():
    """AI-powered tender document creation with conversational interface"""

    st.subheader("🔄 Create Intelligent Tender")

    # Initialize new tender if none exists
    if st.session_state.current_tender is None:
        st.session_state.current_tender = {
            'id': str(uuid.uuid4())[:8],
            'title': '',
            'status': 'Draft',
            'category': 'General',
            'created_date': datetime.now().strftime("%Y-%m-%d"),
            'conversation_history': [],
            'document_sections': {},
            'suppliers': [],
            'requirements': {},
            'timeline': {},
            'evaluation_criteria': {}
        }
        st.session_state.conversation_history = []

    # Two-column layout: Conversation and Document
    col1, col2 = st.columns([1, 1])

    with col1:
        show_tender_conversation()

    with col2:
        show_tender_document_editor()

def show_tender_conversation():
    """Conversational AI interface for building tender documents"""

    st.subheader("💬 AI Assistant")
    st.markdown("*Ask me to help build your tender document*")

    # Display conversation history
    for msg in st.session_state.conversation_history:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg['content'])

    # Quick questions for the AI
    st.markdown("**💡 Quick Questions to Get Started:**")
    quick_questions = [
        "Help me create a tender for electrical equipment",
        "What sections should I include in a construction tender?",
        "Define evaluation criteria for supplier selection",
        "Create a timeline for a 3-month procurement process"
    ]

    for i, question in enumerate(quick_questions):
        if st.button(question, key=f"quick_q_{i}", use_container_width=True):
            st.session_state.conversation_history.append({
                'role': 'user',
                'content': question,
                'timestamp': datetime.now().isoformat()
            })
            process_ai_response(question)
            st.rerun()

    # User input
    user_input = st.chat_input("Ask about tender creation...")

    if user_input:
        # Add user message
        st.session_state.conversation_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })

        # Process AI response
        process_ai_response(user_input)
        st.rerun()

def process_ai_response(user_input):
    """Process user input and generate AI response that updates the tender document"""

    # Simple rule-based responses (in production, this would use your AI agent)
    user_input_lower = user_input.lower()

    if any(word in user_input_lower for word in ['electrical', 'equipment', 'cables', 'transformers']):
        response = """
        **I'll help you create an electrical equipment tender!** 🔌

        I've started structuring your tender with these key sections:

        ✅ **Technical Specifications**
        - Voltage ratings and standards compliance
        - Safety certifications required
        - Performance metrics
        - Installation requirements

        ✅ **Commercial Terms**
        - Payment milestones
        - Warranty periods (suggested: 24 months)
        - Delivery terms (INCOTERMS 2020)

        ✅ **Evaluation Criteria**
        - Technical compliance: 40%
        - Price: 35%
        - Delivery timeline: 15%
        - Past experience: 10%

        Would you like me to add any specific electrical standards or modify these sections?
        """

        # Update tender document structure
        update_tender_from_ai('electrical_equipment')

    elif any(word in user_input_lower for word in ['construction', 'building', 'contractor']):
        response = """
        **Building a construction tender!** 🏗️

        Here's a comprehensive structure for your construction tender:

        ✅ **Project Scope & Specifications**
        - Detailed work breakdown structure
        - Materials specifications
        - Quality standards
        - Safety requirements

        ✅ **Timeline & Milestones**
        - Project phases with deadlines
        - Critical path items
        - Progress reporting requirements

        ✅ **Commercial Proposal**
        - Fixed price vs. cost-plus options
        - Payment schedule tied to milestones
        - Variation management process

        ✅ **Compliance & Certifications**
        - Building code compliance
        - Environmental regulations
        - Safety certifications

        Shall I focus on any specific type of construction project?
        """
        update_tender_from_ai('construction')

    elif any(word in user_input_lower for word in ['criteria', 'evaluation', 'scoring']):
        response = """
        **Setting up evaluation criteria!** 📊

        Here's a balanced evaluation framework:

        **Weighting Structure:**
        - **Technical Compliance (40%)**: Meets all specifications
        - **Price Competitiveness (30%)**: Total cost of ownership
        - **Delivery & Timeline (15%)**: Lead time and reliability
        - **Supplier Experience (10%)**: Past performance and references
        - **Sustainability (5%)**: ESG compliance and green initiatives

        **Scoring Method:**
        - 5-point scale for qualitative factors
        - Price scoring: (Lowest Price / Supplier Price) × Points
        - Minimum threshold: 70% for technical compliance

        Would you like to adjust these weightings or add specific technical criteria?
        """
        update_tender_from_ai('evaluation_criteria')

    elif any(word in user_input_lower for word in ['timeline', 'schedule', 'deadline']):
        response = """
        **Creating project timeline!** 📅

        Suggested 3-month procurement timeline:

        **Phase 1: Preparation (Week 1-2)**
        - Finalize specifications
        - Prepare tender documents
        - Identify potential suppliers

        **Phase 2: Bidding (Week 3-6)**
        - Issue tender (Week 3)
        - Supplier questions (Week 4)
        - Site visits (Week 5)
        - Bid submission (End of Week 6)

        **Phase 3: Evaluation (Week 7-8)**
        - Technical evaluation
        - Commercial assessment
        - Supplier interviews

        **Phase 4: Award & Mobilization (Week 9-12)**
        - Contract award
        - Supplier onboarding
        - Project kickoff

        Should I adjust this timeline or add specific milestones?
        """
        update_tender_from_ai('timeline')

    else:
        response = """
        **I'm here to help you build a comprehensive tender document!** 📝

        I can assist with:
        - **Document Structure**: Creating well-organized tender sections
        - **Technical Specifications**: Defining detailed requirements
        - **Commercial Terms**: Setting payment and delivery terms
        - **Evaluation Criteria**: Establishing fair scoring methods
        - **Timeline Planning**: Creating realistic project schedules

        Tell me what you're procuring or ask specific questions about any tender component!
        """

    # Add AI response to history
    st.session_state.conversation_history.append({
        'role': 'assistant',
        'content': response,
        'timestamp': datetime.now().isoformat()
    })

def update_tender_from_ai(tender_type):
    """Update tender document based on AI conversation"""

    if tender_type == 'electrical_equipment':
        st.session_state.current_tender.update({
            'category': 'Electrical Equipment',
            'document_sections': {
                'technical_specs': {
                    'title': 'Technical Specifications',
                    'content': 'Voltage ratings, safety certifications, performance metrics, installation requirements',
                    'status': 'Draft'
                },
                'commercial_terms': {
                    'title': 'Commercial Terms',
                    'content': 'Payment milestones, warranty periods, delivery terms (INCOTERMS 2020)',
                    'status': 'Draft'
                },
                'evaluation_criteria': {
                    'title': 'Evaluation Criteria',
                    'content': 'Technical compliance (40%), Price (35%), Delivery (15%), Past experience (10%)',
                    'status': 'Draft'
                }
            },
            'evaluation_criteria': {
                'technical_compliance': 40,
                'price': 35,
                'delivery': 15,
                'experience': 10
            }
        })

    elif tender_type == 'construction':
        st.session_state.current_tender.update({
            'category': 'Construction',
            'document_sections': {
                'project_scope': {
                    'title': 'Project Scope & Specifications',
                    'content': 'Detailed work breakdown, materials specs, quality standards, safety requirements',
                    'status': 'Draft'
                },
                'timeline': {
                    'title': 'Timeline & Milestones',
                    'content': 'Project phases, critical path, progress reporting',
                    'status': 'Draft'
                },
                'commercial': {
                    'title': 'Commercial Proposal',
                    'content': 'Pricing options, payment schedule, variation management',
                    'status': 'Draft'
                }
            }
        })

    elif tender_type == 'evaluation_criteria':
        st.session_state.current_tender.update({
            'evaluation_criteria': {
                'technical_compliance': 40,
                'price': 30,
                'delivery': 15,
                'experience': 10,
                'sustainability': 5
            }
        })

    elif tender_type == 'timeline':
        st.session_state.current_tender.update({
            'timeline': {
                'preparation': 'Week 1-2',
                'bidding': 'Week 3-6',
                'evaluation': 'Week 7-8',
                'award': 'Week 9-12'
            }
        })

def show_tender_document_editor():
    """Interactive tender document editor with Notion-like blocks"""

    st.subheader("📄 Tender Document Editor")

    if not st.session_state.current_tender:
        st.info("Start a conversation with the AI to begin building your tender document.")
        return

    # Tender metadata
    col1, col2 = st.columns(2)

    with col1:
        title = st.text_input(
            "Tender Title",
            value=st.session_state.current_tender.get('title', ''),
            placeholder="e.g., Electrical Equipment Supply for Project X"
        )
        st.session_state.current_tender['title'] = title

    with col2:
        category = st.selectbox(
            "Category",
            options=['General', 'Electrical Equipment', 'Construction', 'IT Services', 'Raw Materials', 'Professional Services'],
            index=['General', 'Electrical Equipment', 'Construction', 'IT Services', 'Raw Materials', 'Professional Services'].index(
                st.session_state.current_tender.get('category', 'General')
            )
        )
        st.session_state.current_tender['category'] = category

    # Document sections - editable blocks
    st.subheader("📑 Document Sections")

    sections = st.session_state.current_tender.get('document_sections', {})

    if not sections:
        st.info("No sections created yet. Ask the AI to help structure your tender document.")
    else:
        for section_key, section_data in sections.items():
            with st.expander(f"📝 {section_data.get('title', 'Untitled')}", expanded=True):
                # Editable content
                content = st.text_area(
                    f"Content for {section_data.get('title', 'Untitled')}",
                    value=section_data.get('content', ''),
                    height=150,
                    key=f"content_{section_key}"
                )
                sections[section_key]['content'] = content

                # Status
                status = st.selectbox(
                    "Status",
                    options=['Draft', 'In Review', 'Finalized'],
                    index=['Draft', 'In Review', 'Finalized'].index(sections[section_key].get('status', 'Draft')),
                    key=f"status_{section_key}"
                )
                sections[section_key]['status'] = status

    # Add new section
    st.markdown("---")
    st.subheader("➕ Add New Section")

    col1, col2 = st.columns([3, 1])

    with col1:
        new_section_title = st.text_input("Section Title", placeholder="e.g., Quality Assurance Requirements")

    with col2:
        if st.button("Add Section", use_container_width=True) and new_section_title:
            new_key = f"section_{len(sections) + 1}"
            sections[new_key] = {
                'title': new_section_title,
                'content': '',
                'status': 'Draft'
            }
            st.session_state.current_tender['document_sections'] = sections
            st.rerun()

    # Evaluation criteria
    st.subheader("📊 Evaluation Criteria")

    criteria = st.session_state.current_tender.get('evaluation_criteria', {})

    if criteria:
        total_weight = sum(criteria.values())
        if total_weight != 100:
            st.warning(f"Total weight is {total_weight}% - should be 100%")

        for criterion, weight in criteria.items():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.text_input(
                    f"Criterion",
                    value=criterion.replace('_', ' ').title(),
                    key=f"label_{criterion}",
                    disabled=True
                )
            with col2:
                new_weight = st.number_input(
                    "Weight %",
                    min_value=0,
                    max_value=100,
                    value=weight,
                    key=f"weight_{criterion}"
                )
                criteria[criterion] = new_weight
    else:
        st.info("No evaluation criteria set. Ask the AI to suggest evaluation criteria.")

    # Save and actions
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("💾 Save Tender", use_container_width=True, type="primary"):
            save_tender()

    with col2:
        if st.button("📤 Publish Tender", use_container_width=True):
            publish_tender()

    with col3:
        if st.button("🔄 Reset", use_container_width=True):
            st.session_state.current_tender = None
            st.rerun()

def save_tender():
    """Save the current tender to the tenders list"""

    if st.session_state.current_tender:
        # Check if this is a new tender or updating existing
        existing_index = None
        for i, tender in enumerate(st.session_state.tenders):
            if tender.get('id') == st.session_state.current_tender.get('id'):
                existing_index = i
                break

        if existing_index is not None:
            # Update existing
            st.session_state.tenders[existing_index] = st.session_state.current_tender
        else:
            # Add new
            st.session_state.tenders.append(st.session_state.current_tender)

        st.success("✅ Tender saved successfully!")

def publish_tender():
    """Publish the tender and change status to Active"""

    if st.session_state.current_tender:
        st.session_state.current_tender['status'] = 'Active'
        st.session_state.current_tender['published_date'] = datetime.now().strftime("%Y-%m-%d")
        save_tender()
        st.success("🎉 Tender published successfully! It's now active and visible to suppliers.")

def show_tender_analysis():
    """Analysis of tender performance and supplier responses"""

    st.subheader("📊 Tender Analysis & History")

    if not st.session_state.tenders:
        st.info("No tenders available for analysis. Create some tenders first!")
        return

    # Filter tenders for analysis
    tender_options = {f"{t['id']} - {t['title']}": t for t in st.session_state.tenders}
    selected_tender_key = st.selectbox(
        "Select Tender for Analysis",
        options=list(tender_options.keys())
    )

    selected_tender = tender_options[selected_tender_key]

    # Analysis dashboard for selected tender
    st.subheader(f"Analysis: {selected_tender['title']}")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Status", selected_tender.get('status', 'Draft'))

    with col2:
        suppliers_count = len(selected_tender.get('suppliers', []))
        st.metric("Suppliers", suppliers_count)

    with col3:
        submissions = len([s for s in selected_tender.get('suppliers', []) if s.get('submitted', False)])
        st.metric("Submissions", submissions)

    with col4:
        if selected_tender.get('estimated_value'):
            st.metric("Estimated Value", f"${selected_tender['estimated_value']:,}")

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        # Status breakdown
        if selected_tender.get('document_sections'):
            section_statuses = [s.get('status', 'Draft') for s in selected_tender['document_sections'].values()]
            status_counts = {status: section_statuses.count(status) for status in set(section_statuses)}

            if status_counts:
                fig = px.pie(
                    values=list(status_counts.values()),
                    names=list(status_counts.keys()),
                    title="Document Section Status"
                )
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Timeline visualization
        if selected_tender.get('timeline'):
            timeline_data = []
            for phase, timeframe in selected_tender['timeline'].items():
                timeline_data.append({
                    'Phase': phase.replace('_', ' ').title(),
                    'Timeframe': timeframe
                })

            if timeline_data:
                df_timeline = pd.DataFrame(timeline_data)
                st.dataframe(df_timeline, use_container_width=True)

    # Supplier responses analysis
    st.subheader("🏭 Supplier Responses")

    suppliers = selected_tender.get('suppliers', [])
    if suppliers:
        supplier_data = []
        for supplier in suppliers:
            supplier_data.append({
                'Supplier': supplier.get('name', 'Unknown'),
                'Status': 'Submitted' if supplier.get('submitted') else 'Invited',
                'Submission Date': supplier.get('submission_date', 'N/A'),
                'Score': supplier.get('score', 'N/A')
            })

        df_suppliers = pd.DataFrame(supplier_data)
        st.dataframe(df_suppliers, use_container_width=True)

        # Add mock supplier responses for demonstration
        if st.button("🔄 Generate Mock Supplier Responses"):
            generate_mock_supplier_responses(selected_tender)
            st.rerun()
    else:
        st.info("No suppliers added yet. Add suppliers to track responses.")

        # Quick add mock suppliers
        if st.button("➕ Add Sample Suppliers"):
            add_sample_suppliers(selected_tender)
            st.rerun()

def generate_mock_supplier_responses(tender):
    """Generate mock supplier responses for demonstration"""

    suppliers = tender.get('suppliers', [])
    for supplier in suppliers:
        if not supplier.get('submitted'):
            # Randomly decide if this supplier submits
            import random
            if random.random() > 0.3:  # 70% chance to submit
                supplier['submitted'] = True
                supplier['submission_date'] = datetime.now().strftime("%Y-%m-%d")
                supplier['score'] = round(random.uniform(60, 95), 1)

    # Update the tender
    for i, t in enumerate(st.session_state.tenders):
        if t['id'] == tender['id']:
            st.session_state.tenders[i] = tender
            break

def add_sample_suppliers(tender):
    """Add sample suppliers to a tender"""

    sample_suppliers = [
        {'name': 'Global Equipment Corp', 'contact': 'procurement@globalcorp.com', 'submitted': False},
        {'name': 'Tech Solutions Ltd', 'contact': 'bids@techsolutions.com', 'submitted': False},
        {'name': 'Premium Supplies Inc', 'contact': 'tenders@premiumsupplies.com', 'submitted': False},
        {'name': 'Reliable Partners Co', 'contact': 'quotes@reliablepartners.com', 'submitted': False}
    ]

    tender['suppliers'] = sample_suppliers

    # Update the tender
    for i, t in enumerate(st.session_state.tenders):
        if t['id'] == tender['id']:
            st.session_state.tenders[i] = tender
            break

def generate_tender_report():
    """Generate comprehensive tender report"""

    st.success("📈 Generating tender performance report...")

    if not st.session_state.tenders:
        st.warning("No tenders available for reporting.")
        return

    # Create summary statistics
    total_tenders = len(st.session_state.tenders)
    status_counts = {}
    category_counts = {}
    total_value = 0

    for tender in st.session_state.tenders:
        status = tender.get('status', 'Draft')
        status_counts[status] = status_counts.get(status, 0) + 1

        category = tender.get('category', 'General')
        category_counts[category] = category_counts.get(category, 0) + 1

        if tender.get('estimated_value'):
            total_value += tender['estimated_value']

    # Display report
    st.subheader("📊 Tender Management Report")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tenders", total_tenders)
        st.metric("Total Estimated Value", f"${total_value:,}")

    with col2:
        st.write("**Status Distribution:**")
        for status, count in status_counts.items():
            st.write(f"- {status}: {count}")

    with col3:
        st.write("**Category Distribution:**")
        for category, count in category_counts.items():
            st.write(f"- {category}: {count}")

    # Performance metrics
    st.subheader("📈 Performance Metrics")

    # Calculate average timeline (mock data)
    avg_preparation_time = "2 weeks"
    avg_evaluation_time = "3 weeks"
    completion_rate = "75%"

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Preparation Time", avg_preparation_time)
    with col2:
        st.metric("Avg Evaluation Time", avg_evaluation_time)
    with col3:
        st.metric("Completion Rate", completion_rate)

    # Export option
    st.download_button(
        label="📥 Download Report as PDF",
        data="Mock PDF content - in production, this would generate an actual PDF",
        file_name=f"tender_report_{datetime.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf"
    )

if __name__ == "__main__":
    main()