# app/pages/02_📈_Enhanced_Forecasting.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime

def main():
    st.header("🎯 Enhanced Price Forecasting")

    # Get data from session state
    prices_df = st.session_state.get('prices_df', pd.DataFrame())
    use_enhanced_forecasting = st.session_state.get('use_enhanced_forecasting', True)

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
        use_fundamentals = st.checkbox("Include Fundamental Factors", value=use_enhanced_forecasting,
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
                config = st.session_state.get('config', {})

                # Try to import the forecaster
                try:
                    from src.models.baseline_forecaster import BaselineForecaster
                    forecaster = BaselineForecaster(config)
                except ImportError:
                    # Use fallback
                    from app.utils.data_loader import FallbackBaselineForecaster
                    forecaster = FallbackBaselineForecaster(config)
                    st.warning("Using fallback forecaster - some features limited")

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
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")

    # Information about enhanced forecasting
    with st.expander("ℹ️ About Enhanced Forecasting"):
        st.markdown("""
        **Enhanced Forecasting** combines:
        - **Statistical models** (rolling averages, momentum, volatility regimes)
        - **EV adoption demand** from IEA/BNEF projections
        - **Geopolitical risk** from GDELT event monitoring
        - **Confidence intervals** (P10/P50/P90) for risk assessment

        **Fundamental Factors:**
        - 📈 EV adoption drives long-term demand
        - 🌍 Geopolitical events create supply risks
        - ⚡ Volatility regimes indicate market stability
        """)

if __name__ == "__main__":
    main()