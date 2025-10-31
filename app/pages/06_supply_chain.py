# app/pages/06_🔗_Supply_Chain.py
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.header("🔗 Global Supply Chain Analysis")

    # Get data from session state
    trade_df = st.session_state.get('trade_df', pd.DataFrame())

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

    else:
        st.warning("Trade data missing required columns for analysis")

if __name__ == "__main__":
    main()