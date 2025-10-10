"""
Performance Dashboard Module
Interactive visualizations and analytics for model assessments
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.google_sheets import get_gsheet_client, load_all_assessments

st.set_page_config(
    page_title="Performance Dashboard",
    page_icon="📈",
    layout="wide"
)

# Initialize
if 'gsheet_client' not in st.session_state:
    st.session_state.gsheet_client = get_gsheet_client()

st.title("📈 Performance Dashboard")
st.caption("Interactive analytics and visualizations")

# Load data
try:
    df = load_all_assessments(st.session_state.gsheet_client)
    
    # DEBUG: Show what we loaded
    st.write(f"🔍 DEBUG: Total rows loaded from sheets: {len(df)}")
    st.write(f"🔍 DEBUG: Sample data:")
    st.dataframe(df.head(10))
    
    if df.empty:
        st.warning("⚠️ No assessment data available yet.")
        st.stop()
    
    # Check for model_id before normalization
    st.write(f"🔍 DEBUG: model_id values BEFORE normalization:")
    st.write(df['model_id'].value_counts())
    
    # NORMALIZE
    df['model_id'] = df['model_id'].str.lower()
    
    st.write(f"🔍 DEBUG: model_id values AFTER normalization:")
    st.write(df['model_id'].value_counts())
    
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.stop()

# Sidebar - Dashboard Type Selection
with st.sidebar:
    st.header("📊 Dashboard Type")
    
    dashboard_type = st.radio(
        "Select View:",
        ["🎯 Single Model", "👥 Model Group", "📦 Product Dashboard", "⚠️ Risk Overview"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # Filters based on dashboard type
    if dashboard_type == "🎯 Single Model":
        st.subheader("Model Selection")
        available_models = sorted(df['model_id'].astype(str).unique(), key=str.lower)
        selected_model = st.selectbox("Choose Model:", available_models)
        
    elif dashboard_type == "👥 Model Group":
        st.subheader("Group Selection")
        group_by = st.selectbox("Group By:", ["Metric Type", "Risk Level", "Custom"])
        
        if group_by == "Metric Type":
            available_metrics = sorted(df['metric'].astype(str).unique(), key=str.lower)
            selected_metric = st.multiselect("Select Metrics:", available_metrics, default=available_metrics[:2] if len(available_metrics) >= 2 else available_metrics)
        elif group_by == "Risk Level":
            risk_levels = st.multiselect("Select Risk Levels:", 
                                        ["Low", "Medium", "High", "Critical"],
                                        default=["High", "Critical"])
        else:
            selected_models = st.multiselect("Select Models:", df['model_id'].unique())
    
    elif dashboard_type == "📦 Product Dashboard":
        st.subheader("Product Selection")
        df['product'] = df['model_id'].astype(str).str.split('_').str[0]
        available_products = sorted(df['product'].unique(), key=str.lower)
        selected_product = st.selectbox("Choose Product:", available_products)
    
    st.divider()
    
    # Date filter
    st.subheader("📅 Time Period")
    date_filter = st.selectbox(
        "Period:",
        ["Last 7 Days", "Last 30 Days", "Last 3 Months", "Last 6 Months", "All Time"]
    )
    
    if date_filter != "All Time":
        days_map = {
            "Last 7 Days": 7,
            "Last 30 Days": 30,
            "Last 3 Months": 90,
            "Last 6 Months": 180
        }
        days = days_map[date_filter]
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= start_date]
    
    st.divider()
    
    if st.button("🏠 Back to Home", use_container_width=True):
        st.switch_page("streamlit_app.py")

# ============================================================================
# DASHBOARD FUNCTIONS
# ============================================================================

def show_single_model_dashboard(df, model_id):
    """Single model detailed view"""
    st.header(f"📊 {model_id} Dashboard")
    
    # DEBUG: Check filtering
    st.write(f"🔍 DEBUG: Total rows in full df: {len(df)}")
    st.write(f"🔍 DEBUG: Looking for model_id: {model_id}")
    
    model_data = df[df['model_id'] == model_id].sort_values('timestamp')
    
    st.write(f"🔍 DEBUG: Rows after filtering for {model_id}: {len(model_data)}")
    st.write(f"🔍 DEBUG: model_data sample:")
    st.dataframe(model_data[['model_id', 'timestamp', 'deviation']])
    
    if model_data.empty:
        st.warning(f"No assessment data for {model_id}")
        return
    
    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Assessments", len(model_data))
    with col2:
        latest_risk = model_data.iloc[-1]['risk_rating']
        st.metric("Current Risk", latest_risk)
    with col3:
        model_data['deviation_numeric'] = pd.to_numeric(
            model_data['deviation'].astype(str).str.rstrip('%'), 
            errors='coerce'
        )
        avg_deviation = model_data['deviation_numeric'].mean()
        st.metric("Avg Deviation", f"{avg_deviation:.2f}%")
    with col4:
        high_risk_count = len(model_data[model_data['risk_rating'].isin(['High', 'Critical'])])
        st.metric("High Risk Events", high_risk_count)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Performance Trend")
        fig = px.line(model_data, x='timestamp', y='deviation_numeric',
                     title='Deviation Over Time',
                     markers=True)
        fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="Baseline")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution")
        risk_counts = model_data['risk_rating'].value_counts()
        fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                    title='Risk Levels',
                    color=risk_counts.index,
                    color_discrete_map={'Low':'green', 'Medium':'yellow', 
                                       'High':'orange', 'Critical':'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Recent assessments
    st.subheader("Recent Assessments")
    recent = model_data.tail(10)[['timestamp', 'current_performance', 'deviation', 'risk_rating']]
    st.dataframe(recent, use_container_width=True)
    
    # High risk events
    high_risk_data = model_data[model_data['risk_rating'].isin(['High', 'Critical'])]
    if not high_risk_data.empty and 'degradation_reason' in high_risk_data.columns:
        st.subheader("⚠️ High Risk Events")
        st.dataframe(high_risk_data[['timestamp', 'deviation', 'degradation_reason', 'mitigation_plan']], 
                    use_container_width=True)

def show_metric_group_dashboard(df, selected_metrics):
    """Compare models by metric type"""
    st.header("👥 Metric Group Comparison")
    
    filtered_df = df[df['metric'].isin(selected_metrics)]
    
    # Group stats
    st.subheader("Group Statistics")
    
    for metric in selected_metrics:
        metric_data = filtered_df[filtered_df['metric'] == metric]
        
        with st.expander(f"📊 {metric} Models ({len(metric_data['model_id'].unique())} models)"):
            col1, col2, col3 = st.columns(3)
            
            metric_data['deviation_numeric'] = pd.to_numeric(
                metric_data['deviation'].astype(str).str.rstrip('%'), 
                errors='coerce'
            )
            
            with col1:
                avg_dev = metric_data['deviation_numeric'].mean()
                st.metric("Avg Deviation", f"{avg_dev:.2f}%")
            
            with col2:
                high_risk = len(metric_data[metric_data['risk_rating'].isin(['High', 'Critical'])])
                st.metric("High Risk Count", high_risk)
            
            with col3:
                total_assessments = len(metric_data)
                st.metric("Total Assessments", total_assessments)
    
    # Comparison chart
    st.subheader("Performance Comparison")
    
    filtered_df['deviation_numeric'] = pd.to_numeric(
        filtered_df['deviation'].astype(str).str.rstrip('%'), 
        errors='coerce'
    )
    
    fig = px.box(filtered_df, x='metric', y='deviation_numeric', color='risk_rating',
                title='Deviation Distribution by Metric',
                color_discrete_map={'Low':'green', 'Medium':'yellow', 
                                   'High':'orange', 'Critical':'red'})
    st.plotly_chart(fig, use_container_width=True)

def show_product_dashboard(df, product):
    """Product-level aggregated view"""
    st.header(f"📦 {product} Product Dashboard")
    
    product_df = df[df['product'] == product]
    
    # Product KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Models", len(product_df['model_id'].unique()))
    with col2:
        st.metric("Total Assessments", len(product_df))
    with col3:
        high_risk = len(product_df[product_df['risk_rating'].isin(['High', 'Critical'])])
        st.metric("High Risk", high_risk)
    with col4:
        product_df['deviation_numeric'] = pd.to_numeric(
            product_df['deviation'].astype(str).str.rstrip('%'), 
            errors='coerce'
        )
        avg_dev = product_df['deviation_numeric'].mean()
        st.metric("Avg Deviation", f"{avg_dev:.2f}%")
    
    # Model performance heatmap
    st.subheader("Model Performance Heatmap")
    
    product_df['month'] = pd.to_datetime(product_df['timestamp']).dt.to_period('M').astype(str)
    
    pivot = product_df.pivot_table(
        values='deviation_numeric',
        index='model_id',
        columns='month',
        aggfunc='mean'
    )
    
    fig = px.imshow(pivot, 
                    title='Monthly Average Deviation by Model',
                    labels=dict(x="Month", y="Model ID", color="Deviation %"),
                    aspect="auto",
                    color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig, use_container_width=True)

def show_risk_overview_dashboard(df):
    """Enterprise risk overview"""
    st.header("⚠️ Risk Overview Dashboard")
    
    # Executive summary
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total = len(df)
        st.metric("Total Assessments", total)
    
    with col2:
        critical = len(df[df['risk_rating'] == 'Critical'])
        st.metric("Critical Risk", critical, delta=f"{(critical/total*100):.1f}%" if total > 0 else "0%")
    
    with col3:
        high = len(df[df['risk_rating'] == 'High'])
        st.metric("High Risk", high, delta=f"{(high/total*100):.1f}%" if total > 0 else "0%")
    
    with col4:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        this_month = len(df[df['timestamp'].dt.month == datetime.now().month])
        st.metric("This Month", this_month)
    
    # Risk trend over time
    st.subheader("Risk Trend")
    
    df['month'] = pd.to_datetime(df['timestamp']).dt.to_period('M').astype(str)
    risk_by_month = df.groupby(['month', 'risk_rating']).size().reset_index(name='count')
    
    fig = px.bar(risk_by_month, x='month', y='count', color='risk_rating',
                title='Risk Distribution Over Time',
                color_discrete_map={'Low':'green', 'Medium':'yellow', 
                                   'High':'orange', 'Critical':'red'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Top risk models
    st.subheader("🚨 Top Risk Models")
    
    high_risk_df = df[df['risk_rating'].isin(['High', 'Critical'])]
    if not high_risk_df.empty:
        top_risk = high_risk_df['model_id'].value_counts().head(10)
        
        fig = px.bar(x=top_risk.index, y=top_risk.values,
                    title='Models with Most High/Critical Assessments',
                    labels={'x':'Model ID', 'y':'Count'})
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No high-risk assessments found.")

# ============================================================================
# RENDER DASHBOARD
# ============================================================================

if dashboard_type == "🎯 Single Model":
    show_single_model_dashboard(df, selected_model)

elif dashboard_type == "👥 Model Group":
    if group_by == "Metric Type" and selected_metric:
        show_metric_group_dashboard(df, selected_metric)
    elif group_by == "Risk Level" and risk_levels:
        filtered = df[df['risk_rating'].isin(risk_levels)]
        st.header("Risk Level Analysis")
        st.dataframe(filtered, use_container_width=True)
    elif group_by == "Custom" and selected_models:
        filtered = df[df['model_id'].isin(selected_models)]
        st.header("Custom Model Group")
        st.dataframe(filtered, use_container_width=True)
    else:
        st.info("Please select at least one option from the sidebar.")

elif dashboard_type == "📦 Product Dashboard":
    show_product_dashboard(df, selected_product)

elif dashboard_type == "⚠️ Risk Overview":
    show_risk_overview_dashboard(df)
