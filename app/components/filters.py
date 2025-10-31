import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def show_filters(prices_df):
    """Display data filters in sidebar"""

    if prices_df is None or prices_df.empty:
        return {}

    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    filters = {}

    # Material filter
    if 'material' in prices_df.columns:
        materials = prices_df['material'].unique()
        selected_material = st.sidebar.selectbox(
            "Select Material",
            options=['All'] + list(materials),
            index=0
        )
        if selected_material != 'All':
            filters['material'] = selected_material

    # Date range filter
    if 'date' in prices_df.columns:
        if pd.api.types.is_datetime64_any_dtype(prices_df['date']):
            min_date = prices_df['date'].min()
            max_date = prices_df['date'].max()

            date_range = st.sidebar.date_input(
                "Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )

            if len(date_range) == 2:
                filters['date_range'] = date_range

    # Source filter (if multiple sources available)
    if 'source' in prices_df.columns:
        sources = prices_df['source'].unique()
        if len(sources) > 1:
            selected_sources = st.sidebar.multiselect(
                "Data Sources",
                options=list(sources),
                default=list(sources)
            )
            if selected_sources:
                filters['sources'] = selected_sources

    return filters