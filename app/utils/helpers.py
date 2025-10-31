# app/utils/helpers.py
import streamlit as st
import os

def show_import_status():
    """Show import status in sidebar"""
    from app.utils.data_loader import GlobalCommodityFetcher

    if GlobalCommodityFetcher is None:
        st.sidebar.error("⚠️ GlobalCommodityFetcher not available - data sourcing limited")
    else:
        st.sidebar.success("✅ GlobalCommodityFetcher loaded")

    # Environment info
    if os.path.exists('/.dockerenv'):
        st.sidebar.success("🐳 Running in Docker container")