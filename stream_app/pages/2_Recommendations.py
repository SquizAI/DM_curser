import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys
from typing import List, Dict, Optional, Tuple, Union, Set
import random
from io import BytesIO
import json

# Add parent directory to path to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utility functions
from stream_app.utils.visualizations import convert_frozenset_to_str
from stream_app.utils.insights import create_recommendation_strategy

# Page configuration
st.set_page_config(
    page_title="Product Recommendations",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .page-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .section-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0D47A1;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .highlight-box {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
        margin-bottom: 15px;
    }
    .card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        text-align: center;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
        text-align: center;
    }
    .product-card {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        border-left: 4px solid #1E88E5;
    }
    .product-title {
        font-weight: 600;
        color: #0D47A1;
        margin-bottom: 5px;
    }
    .product-score {
        font-size: 0.9rem;
        color: #555;
    }
    .recommendation-chart {
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='page-title'>Product Recommendations</h1>", unsafe_allow_html=True)

# Check if rules exist in session state
if 'rules' not in st.session_state or st.session_state.rules is None or st.session_state.rules.empty:
    st.warning("No association rules found. Please go to the Home page and generate rules first.")
    st.stop()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "🔍 Product Lookup", "🛒 Smart Basket Analyzer", "📋 Recommendation Management"
])

# Get or generate recommendations
if 'recommendations' not in st.session_state or not st.session_state.recommendations:
    with st.spinner("Generating product recommendations..."):
        st.session_state.recommendations = create_recommendation_strategy(st.session_state.rules)
        st.success("Recommendations generated successfully!")

# Product Lookup tab
with tab1:
    st.markdown("<h2 class='section-title'>Find Recommendations for a Product</h2>", unsafe_allow_html=True)
    
    # Get all unique products
    all_products = sorted(list(st.session_state.recommendations.keys()))
    
    # Search box for products
    search_query = st.text_input("Search for a product", "")
    
    if search_query:
        # Filter products based on search query
        filtered_products = [p for p in all_products if search_query.lower() in p.lower()]
        
        if filtered_products:
            # Display matching products
            selected_product = st.selectbox("Select a product", filtered_products)
            
            if selected_product in st.session_state.recommendations:
                # Display recommendations
                st.markdown("<h3 class='section-title'>Top Recommendations</h3>", unsafe_allow_html=True)
                
                # Get recommendations for the selected product
                product_recs = st.session_state.recommendations[selected_product]
                
                # Create columns for layout
                cols = st.columns(3)
                
                # Display top recommendations
                for i, (rec_product, lift) in enumerate(product_recs[:6]):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.markdown(f"""
                        <div class="product-card">
                            <div class="product-title">{rec_product}</div>
                            <div class="product-score">Recommendation Score: {lift:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Create visualization of recommendation strength
                if product_recs:
                    rec_df = pd.DataFrame({
                        'Product': [rec[0] for rec in product_recs[:10]],
                        'Score': [rec[1] for rec in product_recs[:10]]
                    })
                    
                    fig = px.bar(
                        rec_df,
                        x='Product',
                        y='Score',
                        title=f"Recommendation Scores for {selected_product}",
                        color='Score',
                        color_continuous_scale='Blues',
                        labels={
                            'Product': 'Recommended Product',
                            'Score': 'Recommendation Score (Lift)'
                        }
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Find rules supporting this recommendation
                    st.markdown("<h3 class='section-title'>Supporting Rules</h3>", unsafe_allow_html=True)
                    
                    # Get rules where selected product is in antecedents
                    supporting_rules = st.session_state.rules[
                        st.session_state.rules['antecedents'].apply(lambda x: selected_product in x)
                    ]
                    
                    if not supporting_rules.empty:
                        # Convert to display format
                        display_rules = convert_frozenset_to_str(supporting_rules)
                        
                        # Round metrics
                        for col in ['support', 'confidence', 'lift']:
                            if col in display_rules.columns:
                                display_rules[col] = display_rules[col].round(3)
                        
                        # Display rules
                        st.dataframe(
                            display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                            use_container_width=True
                        )
                    else:
                        st.info("No direct supporting rules found. Recommendations may be derived from multiple patterns.")
            else:
                st.warning("No recommendations available for this product.")
        else:
            st.warning("No products found matching your search query.")
    else:
        # If no search query, show random products
        st.markdown("<h3 class='section-title'>Browse Products</h3>", unsafe_allow_html=True)
        
        # Show a few random products to browse
        sample_products = random.sample(all_products, min(6, len(all_products)))
        
        # Create columns
        cols = st.columns(3)
        
        for i, product in enumerate(sample_products):
            col_idx = i % 3
            with cols[col_idx]:
                st.button(
                    product,
                    key=f"product_button_{i}",
                    on_click=lambda p=product: st.session_state.update({"selected_product": p})
                )
        
        # If a product is selected via button, show recommendations
        if 'selected_product' in st.session_state and st.session_state.selected_product in st.session_state.recommendations:
            selected_product = st.session_state.selected_product
            
            # Display recommendations
            st.markdown(f"<h3 class='section-title'>Recommendations for {selected_product}</h3>", unsafe_allow_html=True)
            
            # Get recommendations
            product_recs = st.session_state.recommendations[selected_product]
            
            # Create columns for layout
            cols = st.columns(3)
            
            # Display top recommendations
            for i, (rec_product, lift) in enumerate(product_recs[:6]):
                col_idx = i % 3
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="product-title">{rec_product}</div>
                        <div class="product-score">Recommendation Score: {lift:.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)

# Smart Basket Analyzer tab
with tab2:
    st.markdown("<h2 class='section-title'>Smart Basket Analyzer</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="highlight-box">
        Add products to your basket and get personalized recommendations based on your current selection.
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize basket if not exists
    if 'basket' not in st.session_state:
        st.session_state.basket = []
    
    # Get unique products for selection
    all_products = sorted(list(st.session_state.recommendations.keys()))
    
    # Create a form for adding products
    with st.form("basket_form"):
        product_to_add = st.selectbox("Select a product to add", [""] + all_products)
        submit_button = st.form_submit_button("Add to Basket")
    
    # Process form submission
    if submit_button and product_to_add:
        if product_to_add not in st.session_state.basket:
            st.session_state.basket.append(product_to_add)
            st.success(f"Added {product_to_add} to your basket!")
    
    # Display current basket
    st.markdown("<h3 class='section-title'>Your Current Basket</h3>", unsafe_allow_html=True)
    
    if st.session_state.basket:
        # Display items with remove buttons
        for i, item in enumerate(st.session_state.basket):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(f"<div class='product-card'>{item}</div>", unsafe_allow_html=True)
            with col2:
                if st.button("Remove", key=f"remove_{i}"):
                    st.session_state.basket.remove(item)
                    st.rerun()
        
        # Button to clear basket
        if st.button("Clear Basket"):
            st.session_state.basket = []
            st.rerun()
        
        # Generate recommendations based on basket
        st.markdown("<h3 class='section-title'>Recommended Products</h3>", unsafe_allow_html=True)
        
        # Get recommendations for each product in basket
        all_recs = []
        for product in st.session_state.basket:
            if product in st.session_state.recommendations:
                # Get recommendations for this product
                product_recs = st.session_state.recommendations[product]
                
                # Add to all recommendations
                for rec_product, lift in product_recs:
                    if rec_product not in st.session_state.basket:  # Don't recommend items already in basket
                        all_recs.append((rec_product, lift, product))
        
        if all_recs:
            # Group by recommended product and sum scores
            rec_dict = {}
            supporting_products = {}
            
            for rec_product, lift, source_product in all_recs:
                if rec_product in rec_dict:
                    rec_dict[rec_product] += lift
                    supporting_products[rec_product].append(source_product)
                else:
                    rec_dict[rec_product] = lift
                    supporting_products[rec_product] = [source_product]
            
            # Convert to dataframe
            rec_df = pd.DataFrame({
                'Product': list(rec_dict.keys()),
                'Score': list(rec_dict.values())
            }).sort_values('Score', ascending=False)
            
            # Display top recommendations
            cols = st.columns(3)
            
            for i, (_, row) in enumerate(rec_df.head(6).iterrows()):
                col_idx = i % 3
                with cols[col_idx]:
                    source_products_str = ", ".join(supporting_products[row['Product']])
                    st.markdown(f"""
                    <div class="product-card">
                        <div class="product-title">{row['Product']}</div>
                        <div class="product-score">Recommendation Score: {row['Score']:.2f}</div>
                        <div class="product-score">Based on: {source_products_str}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Visualize recommendations
            fig = px.bar(
                rec_df.head(10),
                x='Product',
                y='Score',
                title="Smart Basket Recommendations",
                color='Score',
                color_continuous_scale='Viridis',
                labels={
                    'Product': 'Recommended Product',
                    'Score': 'Combined Recommendation Score'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add a button to add top recommendation to basket
            if not rec_df.empty:
                top_rec = rec_df.iloc[0]['Product']
                if st.button(f"Add {top_rec} to Basket"):
                    st.session_state.basket.append(top_rec)
                    st.success(f"Added {top_rec} to your basket!")
                    st.rerun()
        else:
            st.info("No recommendations available for the current basket.")
    else:
        st.info("Your basket is empty. Add products to get recommendations.")

# Recommendation Management tab
with tab3:
    st.markdown("<h2 class='section-title'>Recommendation Management</h2>", unsafe_allow_html=True)
    
    # Recommendation metrics
    st.markdown("<h3 class='section-title'>Recommendation Metrics</h3>", unsafe_allow_html=True)
    
    # Calculate metrics
    total_products = len(st.session_state.recommendations)
    total_rules = len(st.session_state.rules)
    avg_recs_per_product = sum(len(recs) for recs in st.session_state.recommendations.values()) / max(1, total_products)
    
    # Create metric cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-value">{total_products}</div>
            <div class="metric-label">Products with Recommendations</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-value">{total_rules}</div>
            <div class="metric-label">Association Rules</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-value">{avg_recs_per_product:.1f}</div>
            <div class="metric-label">Avg. Recommendations per Product</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Options to tune recommendations
    st.markdown("<h3 class='section-title'>Tune Recommendation Settings</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        min_lift_threshold = st.slider(
            "Minimum Lift Threshold", 
            min_value=1.0, 
            max_value=5.0, 
            value=1.2, 
            step=0.1
        )
    
    with col2:
        max_recommendations = st.slider(
            "Maximum Recommendations per Product",
            min_value=3,
            max_value=20,
            value=10,
            step=1
        )
    
    # Button to update recommendations
    if st.button("Update Recommendation Settings"):
        with st.spinner("Updating recommendations..."):
            # Regenerate recommendations with new settings
            st.session_state.recommendations = create_recommendation_strategy(
                st.session_state.rules,
                min_lift=min_lift_threshold
            )
            
            # Limit to max recommendations
            for product in st.session_state.recommendations:
                st.session_state.recommendations[product] = st.session_state.recommendations[product][:max_recommendations]
            
            st.success("Recommendations updated successfully!")
    
    # Export recommendations
    st.markdown("<h3 class='section-title'>Export Recommendations</h3>", unsafe_allow_html=True)
    
    export_format = st.selectbox("Export Format", ["CSV", "Excel", "JSON"])
    
    if st.button("Generate Export"):
        with st.spinner("Preparing export..."):
            # Convert to a dataframe for export
            rec_rows = []
            for product, recs in st.session_state.recommendations.items():
                for rec_product, score in recs:
                    rec_rows.append({
                        'Product': product,
                        'Recommendation': rec_product,
                        'Score': score
                    })
            
            rec_export_df = pd.DataFrame(rec_rows)
            
            if export_format == "CSV":
                csv = rec_export_df.to_csv(index=False)
                
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="product_recommendations.csv",
                    mime="text/csv"
                )
                
            elif export_format == "Excel":
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    rec_export_df.to_excel(writer, sheet_name='Recommendations', index=False)
                
                output.seek(0)
                
                st.download_button(
                    label="Download Excel",
                    data=output,
                    file_name="product_recommendations.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            elif export_format == "JSON":
                # Create a nested structure for JSON
                rec_json = {}
                
                for product, recs in st.session_state.recommendations.items():
                    rec_json[product] = [{"product": rec[0], "score": float(rec[1])} for rec in recs]
                
                json_str = json.dumps({"recommendations": rec_json}, indent=2)
                
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="product_recommendations.json",
                    mime="application/json"
                ) 