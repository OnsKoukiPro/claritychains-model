# src/analytics/procurement_optimizer.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class ProcurementOptimizer:
    def __init__(self, config):
        self.config = config

    def render_procurement_analysis(self, prices_df):
        """Render complete procurement analysis"""
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

        # Price analysis
        self._render_price_analysis(material_data)

        # Procurement recommendation
        self._render_procurement_recommendation(material_data)

        # Hedging strategy
        self._render_hedging_strategy(material_data, material)

    def _render_price_analysis(self, material_data):
        """Render price analysis section"""
        st.subheader("Price Analysis")

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

    def _render_procurement_recommendation(self, material_data):
        """Render procurement recommendation"""
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

    def _render_hedging_strategy(self, material_data, material):
        """Render hedging strategy with editable plan"""
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
                current_price = material_data['price'].iloc[-1]

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
                if include_fundamentals:
                    try:
                        from src.models.baseline_forecaster import BaselineForecaster
                        forecaster = BaselineForecaster(self.config)
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
            self._render_editable_plan()

    def _render_editable_plan(self):
        """Render editable procurement plan"""
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
                total_planned = sum(updated_volumes) if updated_volumes else st.session_state.get('total_volume', 1000)
                if total_planned > 0:
                    new_allocation = (new_volume / total_planned) * 100
                    st.write(f"Allocation: {new_allocation:.1f}%")

        # Calculate and display updated plan
        if st.button("Update Plan", type="secondary"):
            self._update_procurement_plan(updated_volumes)

        # Risk analysis
        self._render_risk_analysis()

    def _update_procurement_plan(self, updated_volumes):
        """Update procurement plan with new volumes"""
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

    def _render_risk_analysis(self):
        """Render risk analysis for current plan"""
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