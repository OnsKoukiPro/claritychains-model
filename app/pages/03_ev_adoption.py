# app/pages/03_🚗_EV_Adoption.py
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.header("🚗 EV Adoption & Demand Impact")

    try:
        config = st.session_state.get('config', {})

        # Try to import EV fetcher
        try:
            from src.data_pipeline.ev_adoption_fetcher import EVAdoptionFetcher
            ev_fetcher = EVAdoptionFetcher(config)
        except ImportError:
            # Use fallback
            from app.utils.data_loader import EVAdoptionFetcher
            ev_fetcher = EVAdoptionFetcher(config)
            st.info("Using fallback EV adoption data")

        material = st.selectbox("Select Material",
                               ['lithium', 'cobalt', 'nickel', 'copper', 'rare_earths'],
                               key='ev_material')

        scenario = st.radio("EV Adoption Scenario",
                           ['conservative', 'stated_policies', 'sustainable'],
                           horizontal=True,
                           help="IEA Global EV Outlook scenarios")

        # Show material intensity information
        with st.expander("📊 Material Intensity Factors"):
            if hasattr(ev_fetcher, 'material_intensity'):
                intensity_data = []
                for mat, specs in ev_fetcher.material_intensity.items():
                    intensity_data.append({
                        'Material': mat.title(),
                        'kg per EV': specs.get('per_ev_kg', 0),
                        'Annual Growth': f"{((specs.get('growth_factor', 1) - 1) * 100):.1f}%",
                        'Data Source': specs.get('data_source', 'N/A')
                    })
                st.dataframe(pd.DataFrame(intensity_data), use_container_width=True)

        if st.button("Generate EV Demand Forecast", type="primary"):
            with st.spinner("Calculating EV-driven demand impact..."):
                demand_data = ev_fetcher.calculate_material_demand(material, scenario)

                if not demand_data.empty:
                    # Display demand chart
                    fig = px.line(demand_data, x='year', y='material_demand_tons',
                                 title=f"{material.title()} Demand from EV Adoption ({scenario.replace('_', ' ').title()} Scenario)",
                                 labels={'material_demand_tons': 'Demand (tons)', 'year': 'Year'})

                    # Add scenario comparison
                    if scenario != 'stated_policies':
                        comp_scenario = 'stated_policies'
                        comp_data = ev_fetcher.calculate_material_demand(material, comp_scenario)
                        if not comp_data.empty:
                            fig.add_scatter(x=comp_data['year'], y=comp_data['material_demand_tons'],
                                          name=f"{comp_scenario.replace('_', ' ').title()} Scenario",
                                          line=dict(dash='dot'))

                    fig.update_layout(hovermode='x unified')
                    st.plotly_chart(fig, use_container_width=True)

                    # Show key metrics
                    st.subheader("📈 Demand Projection Metrics")
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        current_demand = demand_data[demand_data['year'] == 2024]['material_demand_tons'].iloc[0]
                        st.metric("2024 Demand", f"{current_demand:,.0f} tons")

                    with col2:
                        future_demand = demand_data[demand_data['year'] == 2030]['material_demand_tons'].iloc[0]
                        st.metric("2030 Projected", f"{future_demand:,.0f} tons")

                    with col3:
                        growth = ((future_demand - current_demand) / current_demand) * 100
                        st.metric("Growth 2024-2030", f"{growth:.1f}%")

                    with col4:
                        # Calculate annualized growth rate
                        years = 2030 - 2024
                        cagr = ((future_demand / current_demand) ** (1/years) - 1) * 100
                        st.metric("CAGR", f"{cagr:.1f}%")

                    # Price impact analysis
                    st.subheader("💰 Potential Price Impact")
                    try:
                        price_impact = ev_fetcher.get_demand_forecast_adjustment(material)
                        adj_col1, adj_col2, adj_col3 = st.columns(3)

                        with adj_col1:
                            st.metric("Demand Growth", f"{price_impact.get('demand_growth_pct', 0):.1f}%")
                        with adj_col2:
                            st.metric("Price Elasticity", f"{price_impact.get('price_elasticity', 0.3):.2f}")
                        with adj_col3:
                            adj_factor = price_impact.get('adjustment_factor', 1.0)
                            price_impact_pct = (adj_factor - 1) * 100
                            st.metric("Price Impact", f"{price_impact_pct:+.1f}%")

                    except Exception as e:
                        st.info("Price impact analysis not available")

                else:
                    st.warning("No demand data available for the selected material and scenario")

    except Exception as e:
        st.error(f"EV adoption analysis failed: {e}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()