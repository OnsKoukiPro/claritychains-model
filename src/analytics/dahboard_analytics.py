# src/analytics/dashboard_analytics.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class DashboardAnalytics:
    def __init__(self, config):
        self.config = config

    def render_global_dashboard(self, prices_df, trade_df, data_source):
        """Render the complete global dashboard"""
        st.header("📊 Global Market Dashboard")

        if prices_df.empty:
            st.warning("No price data available")
            return

        # Key metrics
        self._render_key_metrics(prices_df)

        # Real-time price trends
        self._render_price_trends(prices_df)

        # Data source info
        self._render_data_source_info(prices_df)

    def _render_key_metrics(self, prices_df):
        """Render key metrics header"""
        st.subheader("📊 Global Market Overview")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if 'material' in prices_df.columns and 'price' in prices_df.columns:
                latest_prices = prices_df.groupby('material')['price'].last()
                avg_price = latest_prices.mean()
                st.metric("Average Price", f"${avg_price:,.0f}/t", "Live")
            else:
                st.metric("Average Price", "N/A")

        with col2:
            if len(prices_df) > 1:
                try:
                    price_change = prices_df.groupby('material').apply(
                        lambda x: (x['price'].iloc[-1] - x['price'].iloc[-2]) / x['price'].iloc[-2] * 100
                    ).mean()
                    st.metric("Avg Daily Change", f"{price_change:+.1f}%")
                except:
                    st.metric("Avg Daily Change", "N/A")
            else:
                st.metric("Avg Daily Change", "N/A")

        with col3:
            if 'material' in prices_df.columns:
                materials_count = len(prices_df['material'].unique())
                st.metric("Materials Tracked", materials_count)
            else:
                st.metric("Materials Tracked", 0)

        with col4:
            if not prices_df.empty and 'source' in prices_df.columns:
                # Count unique regions
                regional_sources = {
                    'US': ['fred', 'etf_', 'futures_'],
                    'Europe': ['ecb', 'lme'],
                    'Global': ['worldbank', 'market_analysis']
                }
                regions_covered = []
                for region, sources in regional_sources.items():
                    if prices_df['source'].str.contains('|'.join(sources), na=False).any():
                        regions_covered.append(region)
                st.metric("Regions Covered", len(regions_covered))

    def _render_price_trends(self, prices_df):
        """Render price trends section"""
        st.subheader("📈 Global Price Trends")

        if 'material' in prices_df.columns and 'price' in prices_df.columns and 'source' in prices_df.columns:
            # Show latest prices by material with source info
            latest = prices_df.sort_values('date').groupby('material').tail(1)

            # Add source type for coloring
            def get_source_type(source):
                if 'fred' in str(source).lower():
                    return 'US'
                elif 'ecb' in str(source).lower() or 'lme' in str(source).lower():
                    return 'Europe'
                elif 'worldbank' in str(source).lower():
                    return 'Global'
                else:
                    return 'Other'

            latest['source_type'] = latest['source'].apply(get_source_type)

            fig = px.bar(latest, x='material', y='price', color='source_type',
                        title="Current Prices by Material and Region",
                        color_discrete_map={'US': '#1f77b4', 'Europe': '#ff7f0e', 'Global': '#2ca02c', 'Other': '#7f7f7f'})
            st.plotly_chart(fig, use_container_width=True)

            # Price history with enhanced visualization by source
            st.subheader("📊 Multi-Source Price History")

            # Create price history plot with source differentiation
            fig = go.Figure()

            # Group by material and source for better visualization
            for material in prices_df['material'].unique():
                material_data = prices_df[prices_df['material'] == material]

                # Show different sources with different line styles
                sources = material_data['source'].unique()
                for i, source in enumerate(sources):
                    source_data = material_data[material_data['source'] == source]
                    line_style = 'solid' if 'fred' in source else 'dash' if 'ecb' in source else 'dot'

                    fig.add_trace(go.Scatter(
                        x=source_data['date'],
                        y=source_data['price'],
                        name=f'{material} ({source})',
                        line=dict(width=2, dash=line_style),
                        opacity=0.7
                    ))

            fig.update_layout(
                title="Historical Price Trends from Multiple Sources",
                xaxis_title="Date",
                yaxis_title="Price (USD/t)",
                hovermode='x unified',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

    def _render_data_source_info(self, prices_df):
        """Render data source information"""
        if not prices_df.empty and 'source' in prices_df.columns:
            source_counts = prices_df['source'].value_counts()
            st.info(f"**Global Data Sources:** {', '.join([f'{k} ({v} recs)' for k, v in source_counts.items()])}")

    def detect_available_regions(self, prices_df):
        """Detect available regions from data sources"""
        region_mapping = {
            'US': ['fred', 'etf_', 'futures_', 'comex', 'nyse'],
            'Europe': ['ecb', 'lme', 'europe', 'euronext', 'euwax'],
            'Asia': ['shfe', 'tfex', 'sgx', 'japan', 'china', 'hongkong'],
            'Global': ['worldbank', 'market_analysis', 'bloomberg', 'reuters'],
            'Other': ['other', 'unknown', 'synthetic']
        }

        available_regions = []
        for region, keywords in region_mapping.items():
            for source in prices_df['source'].unique():
                if any(keyword in str(source).lower() for keyword in keywords):
                    if region not in available_regions:
                        available_regions.append(region)
                    break

        return available_regions

    def apply_dashboard_filters(self, prices_df, filters):
        """Apply comprehensive filters to the dataset"""
        filtered_df = prices_df.copy()

        # Material filter
        if 'materials' in filters and filters['materials']:
            filtered_df = filtered_df[filtered_df['material'].isin(filters['materials'])]

        # Source filter
        if 'sources' in filters and filters['sources']:
            filtered_df = filtered_df[filtered_df['source'].isin(filters['sources'])]

        # Time range filter
        if filters['time_range'] != 'All':
            end_date = filtered_df['date'].max()
            if filters['time_range'] == '1M':
                start_date = end_date - pd.Timedelta(days=30)
            elif filters['time_range'] == '3M':
                start_date = end_date - pd.Timedelta(days=90)
            elif filters['time_range'] == '6M':
                start_date = end_date - pd.Timedelta(days=180)
            elif filters['time_range'] == '1Y':
                start_date = end_date - pd.Timedelta(days=365)
            elif filters['time_range'] == '2Y':
                start_date = end_date - pd.Timedelta(days=730)
            else:  # 5Y
                start_date = end_date - pd.Timedelta(days=1825)

            filtered_df = filtered_df[filtered_df['date'] >= start_date]

        return filtered_df

    def calculate_price_momentum(self, df):
        """Calculate overall price momentum"""
        recent = df[df['date'] > df['date'].max() - pd.Timedelta(days=30)]
        older = df[df['date'] <= df['date'].max() - pd.Timedelta(days=30)]

        if len(recent) == 0 or len(older) == 0:
            return 0

        recent_avg = recent.groupby('material')['price'].last().mean()
        older_avg = older.groupby('material')['price'].last().mean()

        return ((recent_avg - older_avg) / older_avg) * 100

    def map_source_to_region(self, source):
        """Map data source to region"""
        source_str = str(source).lower()

        if any(x in source_str for x in ['fred', 'etf_', 'futures_', 'comex', 'nyse']):
            return 'US'
        elif any(x in source_str for x in ['ecb', 'lme', 'europe', 'euronext']):
            return 'Europe'
        elif any(x in source_str for x in ['shfe', 'tfex', 'sgx', 'japan', 'china']):
            return 'Asia'
        elif any(x in source_str for x in ['worldbank', 'market_analysis']):
            return 'Global'
        else:
            return 'Other'