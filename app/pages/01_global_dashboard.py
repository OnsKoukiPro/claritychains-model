# app/pages/01_🌍_Global_Dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.header("📊 Global Market Dashboard")

    # Get data from session state
    prices_df = st.session_state.get('prices_df', pd.DataFrame())
    trade_df = st.session_state.get('trade_df', pd.DataFrame())
    data_source = st.session_state.get('data_source', 'unknown')

    if prices_df.empty:
        st.warning("No price data available")
        return

    # Key metrics - enhanced with global indicators
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

    # Real-time price trends with source coloring
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

    # Enhanced data source info
    if not prices_df.empty and 'source' in prices_df.columns:
        source_counts = prices_df['source'].value_counts()
        st.info(f"**Global Data Sources:** {', '.join([f'{k} ({v} recs)' for k, v in source_counts.items()])}")

if __name__ == "__main__":
    main()