# src/analytics/supply_chain_analyzer.py
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SupplyChainAnalyzer:
    def __init__(self, config):
        self.config = config

    def render_supply_chain_analysis(self, trade_df):
        """Render supply chain risk analysis"""
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

    def calculate_supply_chain_metrics(self, trade_df):
        """Calculate comprehensive supply chain metrics"""
        if trade_df.empty:
            return {}

        metrics = {}

        if 'material' in trade_df.columns and 'value_usd' in trade_df.columns:
            for material in trade_df['material'].unique():
                material_trade = trade_df[trade_df['material'] == material]

                if not material_trade.empty:
                    # Basic metrics
                    total_trade_value = material_trade['value_usd'].sum()
                    country_count = material_trade['exporter'].nunique() if 'exporter' in material_trade.columns else 0

                    # Concentration metrics
                    if 'exporter' in material_trade.columns:
                        supplier_shares = (material_trade.groupby('exporter')['value_usd'].sum() / total_trade_value)
                        hhi = (supplier_shares ** 2).sum()
                        top_3_concentration = supplier_shares.nlargest(3).sum()
                        top_supplier = supplier_shares.idxmax()
                        top_supplier_share = supplier_shares.max()
                    else:
                        hhi = 0
                        top_3_concentration = 0
                        top_supplier = "Unknown"
                        top_supplier_share = 0

                    metrics[material] = {
                        'total_trade_value': total_trade_value,
                        'country_count': country_count,
                        'hhi': hhi,
                        'top_3_concentration': top_3_concentration,
                        'top_supplier': top_supplier,
                        'top_supplier_share': top_supplier_share,
                        'risk_level': self._classify_supply_chain_risk(hhi, top_3_concentration)
                    }

        return metrics

    def _classify_supply_chain_risk(self, hhi, top_3_concentration):
        """Classify supply chain risk level"""
        if hhi > 0.25 or top_3_concentration > 0.8:
            return 'HIGH'
        elif hhi > 0.15 or top_3_concentration > 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'

    def generate_supply_chain_recommendations(self, metrics):
        """Generate supply chain recommendations based on metrics"""
        recommendations = {}

        for material, metric in metrics.items():
            material_recs = []

            if metric['risk_level'] == 'HIGH':
                material_recs.extend([
                    "Immediate supplier diversification required",
                    "Consider strategic stockpiling",
                    "Develop alternative sourcing strategies",
                    "Monitor geopolitical risks in key supplier regions"
                ])
            elif metric['risk_level'] == 'MEDIUM':
                material_recs.extend([
                    "Monitor supplier concentration regularly",
                    "Develop contingency plans for key suppliers",
                    "Explore new supplier relationships",
                    "Consider long-term contracts for stability"
                ])
            else:
                material_recs.extend([
                    "Maintain current supplier relationships",
                    "Continue monitoring market developments",
                    "Consider strategic partnerships for future security"
                ])

            recommendations[material] = material_recs

        return recommendations