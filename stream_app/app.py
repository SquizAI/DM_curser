import streamlit as st
import sys
import os
import time
import pandas as pd
import numpy as np
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, fpgrowth
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import json
from typing import List, Dict, Tuple, Set, Optional, Union
import base64
from io import BytesIO
from datetime import datetime
import gc

# Print debugging information
print(f"Python version: {sys.version}")
print(f"Running app.py from: {__file__}")

# Add utils directory to sys.path to allow importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'utils'))

print(f"Python path: {sys.path}")

# Define the import_utils_when_needed function first
def import_utils_when_needed():
    # Create global variables for the modules
    global load_and_prep_data, create_time_based_datasets, segment_customers
    global get_rules, process_rules_batch, analyze_rules_over_time, prune_redundant_rules, detect_insights, mine_rules_by_segment
    global create_rule_scatterplot, create_3d_rule_visualization, create_rule_network, create_item_frequency_chart, create_metric_distribution_plots, visualize_rules_over_time, create_top_rules_table
    global generate_business_insights, create_segment_recommendations, identify_cross_sell_opportunities, create_executive_summary
    global generate_pdf_report
    global profile_memory_usage, time_function, convert_pandas_to_polars
    
    # Import utility modules
    print("Importing utility modules...")
    try:
        from utils.data_loader import load_and_prep_data, create_time_based_datasets, segment_customers
        print("Imported data_loader")
    except Exception as e:
        print(f"Error importing data_loader: {e}")
    
    try:
        from utils.rule_mining import get_rules, process_rules_batch, analyze_rules_over_time, prune_redundant_rules, detect_insights, mine_rules_by_segment
        print("Imported rule_mining")
    except Exception as e:
        print(f"Error importing rule_mining: {e}")
        
        # Provide backup definition for get_rules
        @st.cache_data
        def get_rules(basket_encoded, min_support=0.01, min_confidence=0.3, min_lift=1.0, algorithm='apriori', max_len=None):
            print("Using backup get_rules function")
            print(f"basket_encoded type: {type(basket_encoded)}")
            
            try:
                # Convert dictionary to proper format if needed
                if isinstance(basket_encoded, dict):
                    # Extract transaction lists from the dictionary
                    transactions = list(basket_encoded.values())
                    
                    # Ensure all items in transactions are strings to prevent type comparison errors
                    str_transactions = []
                    for transaction in transactions:
                        # Convert each item in the transaction to a string
                        str_transaction = [str(item) for item in transaction]
                        str_transactions.append(str_transaction)
                    
                    # Use TransactionEncoder to convert to binary format
                    te = TransactionEncoder()
                    te_ary = te.fit_transform(str_transactions)
                    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
                elif isinstance(basket_encoded, set):
                    # Handle when basket_encoded is a set
                    print("Converting set to proper format for apriori")
                    # For a set, create individual transactions 
                    # (one item per transaction works better than putting all in one transaction)
                    transactions = [[item] for item in basket_encoded]
                    
                    # Ensure all items are strings
                    str_transactions = []
                    for transaction in transactions:
                        str_transaction = [str(item) for item in transaction]
                        str_transactions.append(str_transaction)
                    
                    # Use TransactionEncoder to convert to binary format
                    te = TransactionEncoder()
                    te_ary = te.fit_transform(str_transactions)
                    basket_df = pd.DataFrame(te_ary, columns=te.columns_)
                else:
                    # Check if it's actually a DataFrame
                    if hasattr(basket_encoded, 'values') and hasattr(basket_encoded, 'columns'):
                        basket_df = basket_encoded
                    else:
                        # If it's not a recognized type, create an empty DataFrame with proper columns
                        print(f"Unrecognized basket_encoded type: {type(basket_encoded)}")
                        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 
                                                'confidence', 'lift', 'leverage', 'conviction'])
                
                # Apply selected algorithm
                if algorithm == 'fpgrowth':
                    frequent_itemsets = fpgrowth(basket_df, 
                                              min_support=min_support, 
                                              use_colnames=True,
                                              max_len=max_len)
                else:  # default to apriori
                    frequent_itemsets = apriori(basket_df, 
                                            min_support=min_support, 
                                            use_colnames=True,
                                            max_len=max_len)
                
                # If no frequent itemsets found, return empty DataFrame
                if frequent_itemsets.empty:
                    return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 
                                             'confidence', 'lift', 'leverage', 'conviction'])
                
                # Generate association rules
                rules = association_rules(frequent_itemsets, 
                                       metric="confidence", 
                                       min_threshold=min_confidence)
                
                # Filter by minimum lift
                rules = rules[rules['lift'] >= min_lift]
                
                return rules
            except Exception as e:
                print(f"Error in rule mining: {e}")
                # Return empty DataFrame on error
                return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 
                                          'confidence', 'lift', 'leverage', 'conviction'])
            
        def prune_redundant_rules(rules):
            if rules.empty:
                return rules
            
            # Sort by lift (higher is better)
            sorted_rules = rules.sort_values('lift', ascending=False)
            
            # Identify rules with same consequent
            pruned_rules = []
            seen_consequents = set()
            
            for _, rule in sorted_rules.iterrows():
                # Convert frozenset to tuple for hashability
                consequent = tuple(rule['consequents'])
                
                # Simple pruning: if we've seen this exact consequent before with higher lift, skip
                if consequent in seen_consequents:
                    continue
                    
                pruned_rules.append(rule)
                seen_consequents.add(consequent)
            
            return pd.DataFrame(pruned_rules)
    
    try:
        from utils.visualizations import (
            create_rule_scatterplot, create_3d_rule_visualization, create_rule_network, 
            create_item_frequency_chart, create_metric_distribution_plots, visualize_rules_over_time,
            create_top_rules_table
        )
        print("Imported visualizations")
    except Exception as e:
        print(f"Error importing visualizations: {e}")
    
    try:
        from utils.insights import (
            generate_business_insights, create_segment_recommendations, 
            identify_cross_sell_opportunities, create_executive_summary
        )
        print("Imported insights")
    except Exception as e:
        print(f"Error importing insights: {e}")
    
    try:
        from utils.reporting import generate_pdf_report
        print("Imported reporting")
    except Exception as e:
        print(f"Error importing reporting: {e}")
    
    try:
        from utils.performance import (
            profile_memory_usage, time_function, convert_pandas_to_polars
        )
        print("Imported performance")
    except Exception as e:
        print(f"Error importing performance: {e}")

# Call import_utils_when_needed immediately to load all modules
import_utils_when_needed()

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

# Add a loading indicator for initial page load
with st.spinner("Starting up..."):
    # Import the fast data loader early to trigger conversion
    try:
        from fast_data_loader import fast_load_sample_data, convert_excel_to_optimized_format
        print("Fast data loader imported successfully")
    except Exception as e:
        print(f"Error importing fast data loader: {e}")

# Initialize ALL session state variables at the very beginning
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
if 'filtered_rules' not in st.session_state:
    st.session_state.filtered_rules = None
if 'basket_encoded' not in st.session_state:
    st.session_state.basket_encoded = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_loader_animation' not in st.session_state:
    st.session_state.show_loader_animation = False
if 'data_loading_started' not in st.session_state:
    st.session_state.data_loading_started = False
if 'last_clicked_button' not in st.session_state:
    st.session_state.last_clicked_button = None
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None
if 'use_polars' not in st.session_state:
    st.session_state.use_polars = False
if 'use_cache' not in st.session_state:
    st.session_state.use_cache = True
if 'use_sample_subset' not in st.session_state:
    st.session_state.use_sample_subset = False
if 'sample_size' not in st.session_state:
    st.session_state.sample_size = 10
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = {}

# Add utils directory to sys.path to allow importing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import memory optimization utilities with error handling
try:
    from render_optimization import memory_optimize, optimize_dataframe, apply_streamlit_optimizations, log_memory_usage
    # Apply memory optimizations for render.com deployment
    apply_streamlit_optimizations()
    log_memory_usage("App startup")
except Exception as e:
    print(f"Error importing optimization utilities: {e}")
    # Define placeholders if imports fail
    def memory_optimize(func):
        return func
    def optimize_dataframe(df):
        return df
    def log_memory_usage(msg):
        print(msg)
    def apply_streamlit_optimizations():
        pass

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'show_loader_animation' not in st.session_state:
    st.session_state.show_loader_animation = False
if 'data_loading_started' not in st.session_state:
    st.session_state.data_loading_started = False
if 'last_clicked_button' not in st.session_state:
    st.session_state.last_clicked_button = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

# Add a custom theme and styling for better UI
st.markdown("""
<style>
    /* Custom theme colors */
    :root {
        --primary-color: #1E88E5; 
        --background-color: #fafafa;
        --secondary-background-color: #f0f2f6;
        --text-color: #262730;
        --font: "Source Sans Pro", sans-serif;
        --card-bg-color: white;
    }
    
    /* Dark mode compatibility */
    @media (prefers-color-scheme: dark) {
        :root {
            --background-color: #0e1117;
            --secondary-background-color: #262730;
            --text-color: #fafafa;
            --card-bg-color: #262730;
        }
    }
    
    /* Global styles */
    body {
        font-family: var(--font);
        color: var(--text-color);
    }
    
    /* Card styling */
    .card {
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
    
    /* Parameter section */
    .parameter-section {
        background-color: var(--card-bg-color);
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    
    /* Highlight section */
    .highlight {
        background-color: rgba(30, 136, 229, 0.1);
        border-left: 4px solid var(--primary-color);
        padding: 10px 15px;
        margin: 10px 0;
    }
    
    /* Success message */
    .success-message {
        background-color: rgba(76, 175, 80, 0.1);
        border-left: 4px solid #4CAF50;
        padding: 10px 15px;
        margin: 10px 0;
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
if 'filtered_rules' not in st.session_state:
    st.session_state.filtered_rules = None
if 'basket_encoded' not in st.session_state:
    st.session_state.basket_encoded = None
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

# Apply performance optimizations
# optimize_streamlit_performance()

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

# Check if we need to show the loading animation
if st.session_state.show_loader_animation:
    # First import the modules if needed
    if not st.session_state.data_loading_started:
        with st.spinner("Initializing modules..."):
            import_utils_when_needed()
            st.session_state.data_loading_started = True
    
    # Show a loading spinner and progress bar
    spinner_col, progress_col = st.columns([1, 3])
    with spinner_col:
        st.spinner("Loading data...")
    with progress_col:
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
    
    # Handle data processing based on which button was clicked
    if st.session_state.last_clicked_button == "Load Uploaded Data" and not st.session_state.data_loaded:
        try:
            with st.spinner("Processing uploaded data..."):
                start_time = time.time()
                
                # Use the file from session state
                if st.session_state.uploaded_file:
                    uploaded_file = st.session_state.uploaded_file
                    
                    # Use Polars if selected
                    if st.session_state.use_polars:
                        from utils.performance import convert_pandas_to_polars
                        df, basket_encoded = load_and_prep_data(file=uploaded_file)
                        df = convert_pandas_to_polars(df)
                    else:
                        df, basket_encoded = load_and_prep_data(file=uploaded_file)
                    
                    st.session_state.df = df
                    st.session_state.basket_encoded = basket_encoded
                    st.session_state.processing_time['data_loading'] = time.time() - start_time
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Successfully loaded data with {df.shape[0]} transactions")
                else:
                    st.error("No file uploaded. Please upload a file first.")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    elif st.session_state.last_clicked_button == "Load Sample Data" and not st.session_state.data_loaded:
        try:
            with st.spinner("Loading sample data..."):
                start_time = time.time()
                
                # Use the FAST data loader instead of the old method
                try:
                    # Use sample percentage if enabled
                    sample_pct = st.session_state.sample_size if st.session_state.use_sample_subset else None
                    use_cache = st.session_state.use_cache
                    
                    # Fast load with disk caching
                    df, basket_encoded = fast_load_sample_data(
                        sample_percentage=sample_pct,
                        use_cache=use_cache
                    )
                    
                    st.session_state.df = df
                    st.session_state.basket_encoded = basket_encoded
                    st.session_state.processing_time['data_loading'] = time.time() - start_time
                    st.session_state.data_loaded = True
                    
                    loading_time = time.time() - start_time
                    st.success(f"✅ Successfully loaded sample data with {df.shape[0]} transactions in {loading_time:.2f} seconds")
                    
                except ImportError:
                    st.warning("Fast data loader not available. Falling back to standard method...")
                    # Fall back to original method
                    sample_data_path = os.path.join(os.path.dirname(__file__), "Online Retail.xlsx")
                    
                    # Check if we have an optimized version of the file
                    try:
                        from utils.data_converter import get_optimized_file_path, convert_excel_to_parquet
                        
                        # Convert to parquet if not already done
                        if sample_data_path.endswith('.xlsx') and os.path.exists(sample_data_path):
                            # First check if faster version exists
                            optimized_path = get_optimized_file_path(sample_data_path)
                            
                            # If it's still the Excel file, convert it
                            if optimized_path == sample_data_path:
                                st.info("Converting Excel to Parquet for faster future loading...")
                                optimized_path = convert_excel_to_parquet(sample_data_path)
                            
                            # Use the optimized path
                            sample_data_path = optimized_path
                    except Exception as e:
                        st.warning(f"Could not optimize file format: {e}")
                    
                    if os.path.exists(sample_data_path):
                        # Use subset if requested
                        if st.session_state.use_sample_subset:
                            sample_pct = st.session_state.sample_size
                            st.info(f"Loading {sample_pct}% of the data for faster processing")
                            df, basket_encoded = load_and_prep_data(file_path=sample_data_path, sample_percentage=sample_pct)
                        else:
                            df, basket_encoded = load_and_prep_data(file_path=sample_data_path)
                        
                        st.session_state.df = df
                        st.session_state.basket_encoded = basket_encoded
                        st.session_state.processing_time['data_loading'] = time.time() - start_time
                        st.session_state.data_loaded = True
                        
                        st.success(f"✅ Successfully loaded sample data with {df.shape[0]} transactions")
                    else:
                        st.error(f"Sample data file not found at {sample_data_path}. Please upload a file instead.")
        except Exception as e:
            st.error(f"Error loading sample data: {str(e)}")
    
    # Turn off the animation flag after processing
    st.session_state.show_loader_animation = False
    st.rerun()  # Refresh UI after data load

# Upload Data tab
with data_tabs[0]:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    # File uploader
    _uploaded_file = st.file_uploader("Upload your transaction data (CSV or Excel)", 
                                     type=["csv", "xlsx", "xls", "parquet"])
    
    # If a file is uploaded, store it in session state
    if _uploaded_file is not None:
        st.session_state.uploaded_file = _uploaded_file
        
        st.info(f"File '{_uploaded_file.name}' ready to load")
        
        # Show options
        _use_cache = st.checkbox("Use cached data if available", value=True,
                                help="Reuse previously loaded data to speed up processing")
        st.session_state.use_cache = _use_cache
        
        _use_polars = st.checkbox("Use Polars instead of Pandas (faster)", value=True,
                                 help="Polars is a faster data processing library that can speed up data loading")
        st.session_state.use_polars = _use_polars
        
        # Add a button to load the data
        if st.button("Load Uploaded Data"):
            st.session_state.last_clicked_button = "Load Uploaded Data"
            st.session_state.show_loader_animation = True
            st.rerun()
    else:
        st.info("💡 Tip: Upload a CSV or Excel file containing transaction data. Each row should represent an item in a transaction.")
        
    st.markdown('</div>', unsafe_allow_html=True)

# Sample Data tab
with data_tabs[1]:
    st.markdown('<div class="parameter-section">', unsafe_allow_html=True)
    st.markdown("""
    <div class="highlight">
        <p>The sample data is the UCI Online Retail dataset, containing transactions from a UK-based online retailer 
        between December 2010 and December 2011.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show a note about speed improvements
    st.info("⚡ Fast data loading is now available! Uses caching, Parquet format, and optimized preprocessing.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample data options
        _use_cache = st.checkbox("Use cached data (fastest)", value=True,
                                help="Reuse previously processed data from disk cache for instant loading")
        st.session_state.use_cache = _use_cache
        
        _use_sample_subset = st.checkbox("Use smaller sample (faster)", value=True,
                                        help="Load only a subset of the data for faster processing")
        st.session_state.use_sample_subset = _use_sample_subset
        
        if _use_sample_subset:
            _sample_size = st.slider("Sample size (% of full data)", 1, 50, 10,
                                    help="Smaller samples load much faster but may affect analysis quality")
            st.session_state.sample_size = _sample_size
    
    with col2:
        # Show typical loading times for different methods
        st.markdown("#### Typical Loading Times:")
        
        st.markdown("""
        - **Excel (original method)**: 30-60 seconds
        - **Parquet (without cache)**: 5-10 seconds
        - **From disk cache**: < 1 second
        - **10% sample with cache**: < 0.5 seconds
        """)
        
        # Show estimated memory usage
        st.markdown("#### Estimated Memory Usage:")
        st.markdown("""
        - **Full dataset**: 200-300 MB
        - **10% sample**: 20-30 MB
        - **1% sample**: 2-3 MB
        """)
    
    # Button to load sample data
    if st.button("Load Sample Data", type="primary"):
        st.session_state.last_clicked_button = "Load Sample Data"
        st.session_state.show_loader_animation = True
        st.rerun()
        
    # Add data cache management options
    with st.expander("Advanced Options"):
        st.markdown("#### Data Cache Management")
        
        if st.button("Clear Data Cache"):
            # Add code to clear cache
            import shutil
            cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
            if os.path.exists(cache_dir):
                try:
                    shutil.rmtree(cache_dir)
                    os.makedirs(cache_dir, exist_ok=True)
                    st.success("Cache cleared successfully!")
                except Exception as e:
                    st.error(f"Error clearing cache: {e}")
        
        st.markdown("#### Force Regenerate Optimized Files")
        
        if st.button("Regenerate Parquet/CSV Files"):
            # Force regeneration
            try:
                from fast_data_loader import convert_excel_to_optimized_format
                # Delete existing optimized files first
                for ext in ["parquet", "csv"]:
                    opt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"Online Retail.{ext}")
                    if os.path.exists(opt_path):
                        os.remove(opt_path)
                # Convert again
                with st.spinner("Converting files..."):
                    convert_excel_to_optimized_format()
                st.success("Files regenerated successfully!")
            except Exception as e:
                st.error(f"Error regenerating files: {e}")
                
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