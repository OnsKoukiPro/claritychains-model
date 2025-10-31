import streamlit as st

def show_header():
    """Display the application header"""

    st.title("🌍 Critical Materials AI Platform - Global Edition")
    st.markdown("**Multi-region procurement intelligence for critical minerals supply chains**")

    # Enhanced subtitle with global features
    st.markdown("""
    <style>
    .global-features {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4, #45b7d1, #96ceb4, #feca57);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
        font-size: 1.1em;
    }
    </style>
    <div class="global-features">
    🌐 NEW: Global Data Sources (US, Europe, World Bank) • 🚀 Multi-Region Coverage • 📊 Enhanced Reliability
    </div>
    """, unsafe_allow_html=True)