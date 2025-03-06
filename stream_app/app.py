import streamlit as st

# Set page config - must be the first Streamlit command
st.set_page_config(
    page_title="Association Rule Mining App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'About': "An advanced association rule mining application"
    }
)

import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import time
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import os
import json
from typing import List, Dict, Tuple, Set, Optional, Union
import base64
from io import BytesIO
from datetime import datetime
import sys

# Add utils directory to sys.path to allow importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import utility modules
from utils.data_loader import load_and_prep_data, create_time_based_datasets, segment_customers
from utils.rule_mining import get_rules, process_rules_batch, analyze_rules_over_time, prune_redundant_rules, detect_insights, mine_rules_by_segment
from utils.visualizations import (
    create_rule_scatterplot, create_3d_rule_visualization, create_rule_network, 
    create_item_frequency_chart, create_metric_distribution_plots, visualize_rules_over_time,
    create_top_rules_table
)
from utils.insights import (
    generate_business_insights, create_segment_recommendations, 
    identify_cross_sell_opportunities, create_executive_summary
)
from utils.reporting import generate_pdf_report
from utils.performance import (
    profile_memory_usage, time_function, convert_pandas_to_polars,
    convert_polars_to_pandas, optimize_pandas_dtypes
)

# Add a custom theme and styling for better UI
st.markdown("""
<style>
    /* Main theme colors - dark mode compatible */
    :root {
        --primary-color: #1E88E5;
        --secondary-color: #4CAF50;
        --background-color: rgba(30, 34, 45, 0.1);
        --text-color: rgba(250, 250, 250, 0.9);
        --card-bg-color: rgba(30, 34, 45, 0.5);
        --highlight-color: rgba(30, 136, 229, 0.2);
    }
    
    /* Main container styling */
    .main {
        color: var(--text-color);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Card styling */
    .stCard {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        padding: 20px;
        margin-bottom: 20px;
        background-color: var(--card-bg-color);
        color: var(--text-color);
    }
    
    /* Section headers */
    .section-header {
        color: var(--primary-color);
        font-weight: 600;
        font-size: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--primary-color);
    }
    
    /* Workflow steps */
    .workflow-step {
        background-color: var(--card-bg-color);
        border-left: 5px solid var(--primary-color);
        padding: 15px;
        margin-bottom: 15px;
        border-radius: 0 5px 5px 0;
        color: var(--text-color);
    }
    
    .workflow-step-number {
        background-color: var(--primary-color);
        color: white;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        margin-right: 10px;
        font-weight: bold;
    }
    
    /* Metrics display */
    .metric-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin: 20px 0;
    }
    
    .metric-card {
        flex: 1;
        min-width: 200px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        text-align: center;
        border-radius: 8px;
        background-color: var(--card-bg-color);
        color: var(--text-color);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
    }
    
    .metric-label {
        font-size: 1rem;
        color: rgba(200, 200, 200, 0.8);
    }
    
    /* Highlight important info */
    .highlight {
        background-color: var(--highlight-color);
        padding: 10px;
        border-radius: 5px;
        border-left: 5px solid var(--primary-color);
        color: var(--text-color);
    }
    
    /* Success messages */
    .success-message {
        background-color: rgba(76, 175, 80, 0.2);
        border-left: 5px solid var(--secondary-color);
        padding: 10px;
        border-radius: 0 5px 5px 0;
        margin: 10px 0;
        color: var(--text-color);
    }
    
    /* Parameters section */
    .parameter-section {
        background-color: var(--card-bg-color);
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        color: var(--text-color);
    }
    
    /* Tooltip styling */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: rgba(50, 50, 50, 0.95);
        color: #fff;
        text-align: center;
        border-radius: 6px;
        padding: 5px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Additional dark theme compatibility styles */
    div.stButton button {
        background-color: var(--primary-color);
        color: white;
        border: none;
    }
    
    div.stButton button:hover {
        background-color: #1565C0;
        color: white;
    }
    
    .stTextInput>div>div>input {
        color: var(--text-color);
        background-color: rgba(50, 50, 50, 0.2);
    }
    
    .stSelectbox>div>div>div {
        color: var(--text-color);
        background-color: rgba(50, 50, 50, 0.2);
    }
    
    .stSlider>div>div>div {
        color: var(--text-color);
    }
    
    /* Ensure tab text is visible */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(50, 50, 50, 0.2);
        color: var(--text-color);
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color) !important;
        color: white !important;
    }
    
    /* Fix table and dataframe text color */
    .stDataFrame table, .stTable table {
        color: var(--text-color);
    }
    
    .stDataFrame th, .stTable th {
        background-color: rgba(30, 136, 229, 0.2);
        color: var(--text-color);
    }
    
    /* Fix expander styling */
    .streamlit-expanderHeader {
        color: var(--text-color);
        background-color: rgba(50, 50, 50, 0.1);
    }
    
    /* Make sure all text elements inherit the text color */
    p, h1, h2, h3, h4, h5, h6, span, label, .stMarkdown {
        color: var(--text-color) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state if not already initialized
if 'df' not in st.session_state:
    st.session_state.df = None
if 'rules' not in st.session_state:
    st.session_state.rules = None
if 'time_rules' not in st.session_state:
    st.session_state.time_rules = None
if 'segment_rules' not in st.session_state:
    st.session_state.segment_rules = None
if 'insights' not in st.session_state:
    st.session_state.insights = []
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = {}
if 'filtered_rules' not in st.session_state:
    st.session_state.filtered_rules = None
if 'basket_encoded' not in st.session_state:
    st.session_state.basket_encoded = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

# Apply performance optimizations
optimize_streamlit_performance()

# Main application header
st.title("🚀 Association Rule Mining Application")
st.markdown("""
<div class="highlight">
    <p>This application allows you to discover hidden patterns in transaction data using association rule mining. 
    Find out which products are frequently purchased together and gain valuable business insights.</p>
</div>
""", unsafe_allow_html=True)

# Create a workflow guide
st.markdown('<div class="section-header">📋 Workflow Guide</div>', unsafe_allow_html=True)
st.markdown("""
<div class="workflow-step">
    <span class="workflow-step-number">1</span>
    <strong>Load your data</strong> - Upload a transaction dataset or use our sample data
</div>
<div class="workflow-step">
    <span class="workflow-step-number">2</span>
    <strong>Configure parameters</strong> - Set support, confidence, and lift thresholds
</div>
<div class="workflow-step">
    <span class="workflow-step-number">3</span>
    <strong>Generate rules</strong> - Run the association rule mining algorithm
</div>
<div class="workflow-step">
    <span class="workflow-step-number">4</span>
    <strong>Explore insights</strong> - Visualize patterns and extract business recommendations
</div>
""", unsafe_allow_html=True)

# Data loading section with tabs for better organization
st.markdown('<div class="section-header">📥 Data Input</div>', unsafe_allow_html=True)

data_tabs = st.tabs(["Upload Data", "Use Sample Data", "Data Preview"])

with data_tabs[0]:  # Upload Data tab
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your transaction data (CSV or Excel)", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        st.info(f"File '{uploaded_file.name}' ready to load")
        
        # Add a button to load the data
        if st.button("Load Uploaded Data"):
            with st.spinner("Processing uploaded data..."):
                try:
                    start_time = time.time()
                    df, basket_encoded = load_and_prep_data(file=uploaded_file)
                    st.session_state.df = df
                    st.session_state.basket_encoded = basket_encoded
                    st.session_state.processing_time['data_loading'] = time.time() - start_time
                    
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ Successfully loaded data with {df.shape[0]} transactions and {df.shape[1]} columns
                    </div>
                    """, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error loading data: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[1]:  # Sample Data tab
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        <p>The sample data is the UCI Online Retail dataset, containing transactions from a UK-based online retailer 
        between December 2010 and December 2011.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Add a button to load sample data
    if st.button("Load Sample Data"):
        with st.spinner("Loading sample data..."):
            try:
                start_time = time.time()
                # Load sample data
                sample_data_path = os.path.join(os.path.dirname(__file__), "Online Retail.xlsx")
                if os.path.exists(sample_data_path):
                    df, basket_encoded = load_and_prep_data(file_path=sample_data_path)
                    st.session_state.df = df
                    st.session_state.basket_encoded = basket_encoded
                    st.session_state.processing_time['data_loading'] = time.time() - start_time
                    
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ Successfully loaded sample data with {df.shape[0]} transactions
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"Sample data file not found at {sample_data_path}. Please upload a file instead.")
            except Exception as e:
                st.error(f"Error loading sample data: {str(e)}")
    st.markdown('</div>', unsafe_allow_html=True)

with data_tabs[2]:  # Data Preview tab
    if st.session_state.df is not None:
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        st.subheader("Data Summary")
        
        # Create metrics for data overview
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Transactions", f"{len(st.session_state.df['InvoiceNo'].unique()):,}")
        with col2:
            st.metric("Products", f"{len(st.session_state.df['Description'].unique()):,}")
        with col3:
            st.metric("Customers", f"{len(st.session_state.df['CustomerID'].dropna().unique()):,}" if 'CustomerID' in st.session_state.df.columns else "N/A")
        with col4:
            st.metric("Countries", f"{len(st.session_state.df['Country'].unique()):,}" if 'Country' in st.session_state.df.columns else "N/A")
        
        st.subheader("Data Preview (First 10 rows)")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        # Show data types
        st.subheader("Column Data Types")
        df_types = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes.astype(str),
            'Non-Null Values': st.session_state.df.count().values,
            'Null %': (st.session_state.df.isna().mean() * 100).round(2).astype(str) + '%'
        })
        st.dataframe(df_types, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Load data first to see a preview")

# Only show mining controls if data is loaded
if st.session_state.df is not None:
    # Mining parameters section with collapsible container for cleaner UI
    st.markdown('<div class="section-header">⚙️ Mining Parameters</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            # Algorithm selection with tooltip
            st.markdown("""
            <div class="tooltip">
                Mining Algorithm <span class="tooltiptext">Apriori is more intuitive, FP-Growth is faster for large datasets</span>
            </div>
            """, unsafe_allow_html=True)
            algorithm = st.selectbox("Algorithm", ["Apriori", "FP-Growth"], index=0, label_visibility="collapsed")
            
            # Support parameter with tooltip
            st.markdown("""
            <div class="tooltip">
                Minimum Support <span class="tooltiptext">Higher values create fewer rules. Suggested: 0.01-0.05</span>
            </div>
            """, unsafe_allow_html=True)
            min_support = st.slider("Minimum Support", 0.001, 0.5, 0.01, 0.001, format="%.3f", label_visibility="collapsed")
        
        with col2:
            # Confidence parameter with tooltip
            st.markdown("""
            <div class="tooltip">
                Minimum Confidence <span class="tooltiptext">Higher values mean more reliable rules. Suggested: 0.3-0.7</span>
            </div>
            """, unsafe_allow_html=True)
            min_confidence = st.slider("Minimum Confidence", 0.1, 1.0, 0.3, 0.05, format="%.2f", label_visibility="collapsed")
            
            # Lift parameter with tooltip
            st.markdown("""
            <div class="tooltip">
                Minimum Lift <span class="tooltiptext">Values > 1 indicate items appear together more often than expected. Suggested: 1.0-3.0</span>
            </div>
            """, unsafe_allow_html=True)
            min_lift = st.slider("Minimum Lift", 1.0, 10.0, 1.0, 0.5, format="%.1f", label_visibility="collapsed")
        
        # Advanced options in an expander
        with st.expander("Advanced Options"):
            max_len = st.slider("Maximum Rule Length", 2, 10, 5, 
                              help="Maximum number of items in a rule (antecedent + consequent)")
            
            prune_redundant = st.checkbox("Prune Redundant Rules", True,
                                        help="Remove redundant or less interesting rules")
            
            if 'InvoiceDate' in st.session_state.df.columns:
                analyze_time = st.checkbox("Analyze Rules Over Time", False,
                                         help="Discover how rules change over different time periods")
                
                if analyze_time:
                    time_granularity = st.selectbox("Time Granularity", 
                                                  ["day", "week", "month", "quarter"], 
                                                  index=2)
            else:
                analyze_time = False
                time_granularity = "month"
                
            if 'CustomerID' in st.session_state.df.columns:
                segment_customers_opt = st.checkbox("Segment Customers", False,
                                                  help="Group customers and analyze segment-specific patterns")
                
                if segment_customers_opt:
                    segmentation_method = st.selectbox("Segmentation Method", 
                                                     ["RFM", "spending", "frequency"], 
                                                     index=0)
            else:
                segment_customers_opt = False
                segmentation_method = "RFM"
                
        # Generate rules button with prominent styling
        if st.button("🔍 Generate Association Rules", type="primary", use_container_width=True):
            with st.spinner("Mining association rules..."):
                start_time = time.time()
                
                # Process options based on user selections
                additional_params = {
                    "max_len": max_len,
                    "prune_redundant": prune_redundant
                }
                
                # Call the rule mining function
                if algorithm == "FP-Growth":
                    rules_df = get_rules(st.session_state.basket_encoded, 
                                       min_support=min_support, 
                                       min_confidence=min_confidence,
                                       min_lift=min_lift,
                                       algorithm='fpgrowth',
                                       max_len=max_len)
                else:
                    rules_df = get_rules(st.session_state.basket_encoded, 
                                       min_support=min_support, 
                                       min_confidence=min_confidence,
                                       min_lift=min_lift,
                                       algorithm='apriori',
                                       max_len=max_len)
                
                # Prune redundant rules if selected
                if prune_redundant and not rules_df.empty:
                    rules_df = prune_redundant_rules(rules_df)
                
                # Store in session state
                st.session_state.rules = rules_df
                st.session_state.processing_time['rule_mining'] = time.time() - start_time
                
                # Process time-based analysis if selected
                if analyze_time and 'InvoiceDate' in st.session_state.df.columns:
                    with st.spinner("Analyzing rules over time..."):
                        time_start = time.time()
                        time_datasets = create_time_based_datasets(st.session_state.df, granularity=time_granularity)
                        time_rules = analyze_rules_over_time(time_datasets, min_support, min_confidence, min_lift)
                        st.session_state.time_rules = time_rules
                        st.session_state.processing_time['time_analysis'] = time.time() - time_start
                
                # Process customer segmentation if selected
                if segment_customers_opt and 'CustomerID' in st.session_state.df.columns:
                    with st.spinner("Segmenting customers..."):
                        segment_start = time.time()
                        segments = segment_customers(st.session_state.df, method=segmentation_method)
                        st.session_state.customer_segments = segments
                        st.session_state.processing_time['segmentation'] = time.time() - segment_start
                        
                        # Mine rules for each segment
                        with st.spinner("Mining rules for each segment..."):
                            segment_rules_start = time.time()
                            segment_rules = mine_rules_by_segment(
                                st.session_state.df, 
                                segments, 
                                min_support=min_support,
                                min_confidence=min_confidence,
                                min_lift=min_lift
                            )
                            st.session_state.segment_rules = segment_rules
                            st.session_state.processing_time['segment_rules'] = time.time() - segment_rules_start
                
                # Show success message with rule count
                if rules_df.empty:
                    st.warning("No rules found with the current parameters. Try lowering the minimum support or confidence.")
                else:
                    st.markdown(f"""
                    <div class="success-message">
                        ✅ Successfully generated {len(rules_df)} association rules in {st.session_state.processing_time['rule_mining']:.2f} seconds
                    </div>
                    """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# Main content area
if st.session_state.df is not None:
    # Create a dashboard layout with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard", "🔍 Detailed Analysis", "⏱️ Time Analysis", 
        "👥 Customer Segments", "🛒 Recommendations"
    ])
    
    # Dashboard tab
    with tab1:
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Transactions</div>
            </div>
            """.format(len(st.session_state.df['InvoiceNo'].unique())), unsafe_allow_html=True)
            
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Products</div>
            </div>
            """.format(len(st.session_state.df['Description'].unique())), unsafe_allow_html=True)
            
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Customers</div>
            </div>
            """.format(len(st.session_state.df['CustomerID'].dropna().unique())), unsafe_allow_html=True)
            
        with col4:
            if st.session_state.rules is not None:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">{:,}</div>
                    <div class="metric-label">Association Rules</div>
                </div>
                """.format(len(st.session_state.rules)), unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-value">-</div>
                    <div class="metric-label">Association Rules</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Overview charts
        st.markdown("<h2 class='sub-header'>Overview</h2>", unsafe_allow_html=True)
        
        if st.session_state.rules is not None and not st.session_state.rules.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.plotly_chart(
                    create_rule_scatterplot(st.session_state.rules), 
                    use_container_width=True,
                    key="dashboard_overview_scatter"
                )
                
            with col2:
                st.dataframe(
                    create_top_rules_table(st.session_state.rules, sort_by='lift'),
                    use_container_width=True
                )
        
        # Key insights
        st.markdown("<h2 class='sub-header'>Key Insights</h2>", unsafe_allow_html=True)
        
        if st.session_state.insights:
            for i, insight in enumerate(st.session_state.insights[:5]):
                st.markdown(f"<div class='highlight'>{insight}</div>", unsafe_allow_html=True)
        else:
            st.info("Generate association rules to see key insights")
    
    # Detailed Analysis tab
    with tab2:
        if st.session_state.rules is not None and not st.session_state.rules.empty:
            # Rule filtering
            st.markdown("<h2 class='sub-header'>Filter Rules</h2>", unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                min_support_filter = st.slider(
                    "Min Support",
                    min_value=float(st.session_state.rules['support'].min()),
                    max_value=float(st.session_state.rules['support'].max()),
                    value=float(st.session_state.rules['support'].min()),
                    step=0.001,
                    format="%.3f"
                )
                
            with col2:
                min_confidence_filter = st.slider(
                    "Min Confidence",
                    min_value=float(st.session_state.rules['confidence'].min()),
                    max_value=float(st.session_state.rules['confidence'].max()),
                    value=float(st.session_state.rules['confidence'].min()),
                    step=0.01,
                    format="%.2f"
                )
                
            with col3:
                min_lift_filter = st.slider(
                    "Min Lift",
                    min_value=float(st.session_state.rules['lift'].min()),
                    max_value=float(st.session_state.rules['lift'].max()),
                    value=float(st.session_state.rules['lift'].min()),
                    step=0.1,
                    format="%.1f"
                )
            
            # Apply filters
            filtered_rules = st.session_state.rules[
                (st.session_state.rules['support'] >= min_support_filter) &
                (st.session_state.rules['confidence'] >= min_confidence_filter) &
                (st.session_state.rules['lift'] >= min_lift_filter)
            ]
            
            st.session_state.filtered_rules = filtered_rules
            
            # Display number of rules after filtering
            st.info(f"Showing {len(filtered_rules)} rules after filtering")
            
            # Visualizations
            st.markdown("<h2 class='sub-header'>Visualizations</h2>", unsafe_allow_html=True)
            
            # Select visualization type
            viz_type = st.selectbox(
                "Visualization Type",
                ["Scatter Plot", "3D Visualization", "Network Graph", "Metric Distributions"],
                index=0
            )
            
            if viz_type == "Scatter Plot":
                color_by = st.selectbox("Color by", ["lift", "confidence", "support"], index=0)
                size_by = st.selectbox("Size by", ["lift", "confidence", "support"], index=0)
                
                st.plotly_chart(
                    create_rule_scatterplot(filtered_rules, colorby=color_by, sizeby=size_by),
                    use_container_width=True,
                    key="detailed_scatter_plot"
                )
                
            elif viz_type == "3D Visualization":
                st.plotly_chart(
                    create_3d_rule_visualization(filtered_rules),
                    use_container_width=True,
                    key="detailed_3d_viz"
                )
                
            elif viz_type == "Network Graph":
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    min_lift_network = st.slider("Minimum Lift for Network", 1.0, 10.0, 2.0, step=0.1)
                    max_rules_network = st.slider("Max Rules to Show", 5, 50, 20, step=5)
                
                with col2:
                    st.plotly_chart(
                        create_rule_network(filtered_rules, min_lift=min_lift_network, max_rules=max_rules_network),
                        use_container_width=True,
                        key="detailed_network_graph"
                    )
                    
            elif viz_type == "Metric Distributions":
                metric = st.selectbox("Metric to Analyze", ["lift", "confidence", "support"], index=0)
                nbins = st.slider("Number of Bins", 5, 50, 20, step=5)
                
                st.plotly_chart(
                    create_metrics_distribution(filtered_rules, metric=metric, nbins=nbins),
                    use_container_width=True,
                    key="detailed_metrics_distribution"
                )
            
            # Rules table
            st.markdown("<h2 class='sub-header'>Rules Table</h2>", unsafe_allow_html=True)
            
            # Sorting options
            sort_by = st.selectbox("Sort by", ["lift", "confidence", "support"], index=0)
            ascending = st.checkbox("Ascending Order", value=False)
            
            # Display top rules table
            st.dataframe(
                create_top_rules_table(
                    filtered_rules,
                    sort_by=sort_by,
                    ascending=ascending,
                    top_n=100  # Show more rules in the detailed tab
                ),
                use_container_width=True
            )
            
        else:
            st.info("Generate association rules to see detailed analysis")
    
    # Time Analysis tab
    with tab3:
        if st.session_state.time_rules is not None:
            st.markdown("<h2 class='sub-header'>Rule Evolution Over Time</h2>", unsafe_allow_html=True)
            
            # Visualization of rule metrics over time
            st.plotly_chart(
                create_temporal_analysis_chart(st.session_state.time_rules),
                use_container_width=True,
                key="temporal_analysis_chart"
            )
            
            # Compare time periods
            st.markdown("<h2 class='sub-header'>Time Period Comparison</h2>", unsafe_allow_html=True)
            
            # Select periods to compare
            periods = list(st.session_state.time_rules.keys())
            if len(periods) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    period1 = st.selectbox("First Period", periods, index=len(periods)-1)
                
                with col2:
                    period2 = st.selectbox("Second Period", periods, index=max(0, len(periods)-2))
                
                # Compare the selected periods
                if period1 != period2:
                    comparison_insights = compare_time_periods(
                        st.session_state.time_rules[period1],
                        st.session_state.time_rules[period2]
                    )
                    
                    for insight in comparison_insights:
                        st.markdown(f"<div class='highlight'>{insight}</div>", unsafe_allow_html=True)
                else:
                    st.warning("Please select different time periods to compare")
            else:
                st.info("Not enough time periods to compare")
                
            # Time period specific rules
            st.markdown("<h2 class='sub-header'>Rules by Time Period</h2>", unsafe_allow_html=True)
            
            selected_period = st.selectbox("Select Time Period", periods)
            
            st.dataframe(
                create_top_rules_table(st.session_state.time_rules[selected_period]),
                use_container_width=True
            )
            
        else:
            st.info("Generate association rules with time analysis to see temporal patterns")
    
    # Customer Segments tab
    with tab4:
        if st.session_state.segment_rules:
            st.markdown("<h2 class='sub-header'>Segment Comparison</h2>", unsafe_allow_html=True)
            
            # Metrics comparison
            segments = list(st.session_state.segment_rules.keys())
            
            # Prepare data for segment comparison
            segment_metrics = {
                'Segment': [],
                'Rule Count': [],
                'Avg Support': [],
                'Avg Confidence': [],
                'Avg Lift': []
            }
            
            for segment, rules in st.session_state.segment_rules.items():
                if not rules.empty:
                    segment_metrics['Segment'].append(segment)
                    segment_metrics['Rule Count'].append(len(rules))
                    segment_metrics['Avg Support'].append(rules['support'].mean())
                    segment_metrics['Avg Confidence'].append(rules['confidence'].mean())
                    segment_metrics['Avg Lift'].append(rules['lift'].mean())
            
            segment_df = pd.DataFrame(segment_metrics)
            
            # Bar chart for comparing segments
            st.plotly_chart(
                px.bar(
                    segment_df,
                    x='Segment',
                    y=['Avg Support', 'Avg Confidence', 'Avg Lift'],
                    barmode='group',
                    title="Average Rule Metrics by Segment",
                    labels={'value': 'Average Value', 'variable': 'Metric'},
                    color_discrete_sequence=['#1E88E5', '#5E35B1', '#43A047']
                ),
                use_container_width=True,
                key="segment_comparison_chart"
            )
            
            # Segment-specific insights
            st.markdown("<h2 class='sub-header'>Segment Insights</h2>", unsafe_allow_html=True)
            
            # Get insights for each segment
            segment_insights = segment_based_insights(st.session_state.segment_rules)
            
            # Create expanders for each segment
            for segment, insights in segment_insights.items():
                with st.expander(f"Insights for {segment} Segment"):
                    for insight in insights:
                        st.markdown(f"<div class='highlight'>{insight}</div>", unsafe_allow_html=True)
            
            # Segment-specific rules
            st.markdown("<h2 class='sub-header'>Rules by Segment</h2>", unsafe_allow_html=True)
            
            selected_segment = st.selectbox("Select Segment", segments)
            
            st.dataframe(
                create_top_rules_table(st.session_state.segment_rules[selected_segment]),
                use_container_width=True
            )
            
        else:
            st.info("Generate association rules with customer segmentation to see segment-specific patterns")
    
    # Recommendations tab
    with tab5:
        if st.session_state.recommendations:
            st.markdown("<h2 class='sub-header'>Product Recommendations</h2>", unsafe_allow_html=True)
            
            # Product lookup
            st.markdown("##### Find recommendations for a specific product")
            
            # Get all unique products from the recommendations
            all_products = list(st.session_state.recommendations.keys())
            
            # Allow user to search for a product
            product_search = st.text_input("Search for a product")
            
            if product_search:
                # Filter products based on search
                filtered_products = [p for p in all_products if product_search.lower() in p.lower()]
                
                if filtered_products:
                    selected_product = st.selectbox("Select a product", filtered_products)
                    
                    if selected_product in st.session_state.recommendations:
                        # Display recommendations
                        st.markdown("##### Top recommendations for this product:")
                        
                        # Create a table of recommendations
                        rec_data = []
                        for item, lift in st.session_state.recommendations[selected_product][:10]:
                            rec_data.append({
                                "Product": item,
                                "Lift Score": lift
                            })
                        
                        rec_df = pd.DataFrame(rec_data)
                        
                        # Display as a table
                        st.dataframe(rec_df, use_container_width=True)
                        
                        # Visualize the recommendations
                        st.plotly_chart(
                            px.bar(
                                rec_df,
                                x="Product",
                                y="Lift Score",
                                title=f"Recommendation Strength for {selected_product}",
                                labels={"Lift Score": "Lift (Recommendation Strength)", "Product": "Recommended Product"},
                                color="Lift Score",
                                color_continuous_scale="Blues"
                            ),
                            use_container_width=True,
                            key=f"product_recommendation_chart_{selected_product.replace(' ', '_')}"
                        )
                else:
                    st.warning("No products found matching your search")
            
            # Top product combinations
            st.markdown("<h2 class='sub-header'>Top Product Combinations</h2>", unsafe_allow_html=True)
            
            if st.session_state.rules is not None and not st.session_state.rules.empty:
                # Get top rules by lift
                top_combinations = create_top_rules_table(
                    st.session_state.rules,
                    sort_by='lift',
                    top_n=10
                )
                
                # Display as a table
                st.dataframe(top_combinations, use_container_width=True)
                
        else:
            st.info("Generate association rules to see product recommendations")

else:
    # Display instructions when no data is loaded
    st.info("Please load data using the sidebar controls to begin analysis")
    
    # Sample images of what the app can do
    st.markdown("<h2 class='sub-header'>Features</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Data Analysis")
        st.markdown("""
        * Load transaction data
        * Generate association rules
        * Filter rules by metrics
        * Extract business insights
        """)
    
    with col2:
        st.markdown("### Visualizations")
        st.markdown("""
        * Interactive scatter plots
        * 3D rule visualizations
        * Network graphs
        * Metric distributions
        """)
    
    with col3:
        st.markdown("### Advanced Analytics")
        st.markdown("""
        * Time-based analysis
        * Customer segmentation
        * Product recommendations
        * Performance optimization
        """)

# Footer
st.markdown("---")
st.markdown("Association Rule Mining Dashboard | Created with Streamlit | v2.0")

# Run the app: streamlit run app.py 