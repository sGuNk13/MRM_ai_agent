"""
Model Risk Management AI Agent - Landing Page
==============================================
One-stop service for model assessment and performance monitoring
"""

import streamlit as st
import sys
import os

# Add utils to path
sys.path.insert(0, os.path.dirname(__file__))

from utils.google_sheets import get_gsheet_client, get_quick_stats
from oauth2client.service_account import ServiceAccountCredentials
import gspread

st.set_page_config(
    page_title="MRM AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Google Sheets client
if 'gsheet_client' not in st.session_state:
    st.session_state.gsheet_client = get_gsheet_client()

# Main landing page
st.title("🤖 Model Risk Management AI Agent")
st.markdown("### Your AI-powered assistant for model performance management")

st.markdown("---")

# Service cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; height: 280px;">
        <h3 style="margin-top: 0;">📊 Model Assessment</h3>
        <p>Assess individual model performance with:</p>
        <ul>
            <li>Real-time risk rating</li>
            <li>Degradation analysis</li>
            <li>Mitigation planning</li>
            <li>Automatic logging</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to Assessment →", use_container_width=True, key="btn_assessment"):
        st.switch_page("pages/1_📊_Model_Assessment.py")

with col2:
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; height: 280px;">
        <h3 style="margin-top: 0;">📈 Performance Dashboard</h3>
        <p>Visualize model performance with:</p>
        <ul>
            <li>Single model trends</li>
            <li>Model group comparisons</li>
            <li>Product analytics</li>
            <li>Risk distribution</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button("Go to Dashboard →", use_container_width=True, key="btn_dashboard"):
        st.switch_page("pages/2_📈_Performance_Dashboard.py")

with col3:
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; height: 280px;">
        <h3 style="margin-top: 0;">🔍 Analytics</h3>
        <p style="font-style: italic; opacity: 0.9;">Coming Soon</p>
        <p>Advanced insights:</p>
        <ul>
            <li>Predictive alerts</li>
            <li>Anomaly detection</li>
            <li>Executive reports</li>
            <li>Trend forecasting</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.button("Coming Soon", use_container_width=True, disabled=True, key="btn_analytics")

st.markdown("---")

# Quick stats
st.subheader("📋 Quick Overview")

try:
    stats = get_quick_stats(st.session_state.gsheet_client)
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Total Assessments", stats['total'])
    with metric_col2:
        st.metric("This Month", stats['this_month'])
    with metric_col3:
        st.metric("High Risk", stats['high_risk'], delta=stats['high_risk_change'])
    with metric_col4:
        st.metric("Avg Deviation", f"{stats['avg_deviation']:.2f}%")
        
except Exception as e:
    st.info("📊 Connect to Google Sheets to see statistics")

# Sidebar navigation
with st.sidebar:
    st.header("🧭 Navigation")
    
    st.page_link("streamlit_app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/1_📊_Model_Assessment.py", label="📊 Model Assessment")
    st.page_link("pages/2_📈_Performance_Dashboard.py", label="📈 Performance Dashboard")
    
    st.divider()
    
    st.subheader("🔌 System Status")
    
    # Check connections
    if st.session_state.gsheet_client:
        st.success("✅ Google Sheets Connected")
    else:
        st.error("❌ Google Sheets Disconnected")
    
    if 'GROQ_API_KEY' in st.secrets:
        st.success("✅ AI Assistant Active")
    else:
        st.error("❌ AI Assistant Inactive")
    
    st.divider()
    
    st.subheader("ℹ️ About")
    st.caption("MRM AI Agent v1.0")
    st.caption("Powered by Llama 3.1")
