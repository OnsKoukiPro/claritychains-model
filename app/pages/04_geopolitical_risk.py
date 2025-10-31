# app/pages/04_🌐_Geopolitical_Risk.py
import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.header("🌍 Geopolitical Risk Monitor")

    try:
        config = st.session_state.get('config', {})

        # Try to import GDELT fetcher
        try:
            from src.data_pipeline.gdelt_fetcher import GDELTFetcher
            gdelt_fetcher = GDELTFetcher(config)
        except ImportError:
            # Use fallback
            from app.utils.data_loader import GDELTFetcher
            gdelt_fetcher = GDELTFetcher(config)
            st.info("Using fallback geopolitical risk data")

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
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")

if __name__ == "__main__":
    main()