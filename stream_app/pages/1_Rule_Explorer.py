import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys
from typing import List, Dict, Optional, Tuple, Union, Set
import json

# Add parent directory to path to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utility functions
from stream_app.utils.visualizations import (
    create_rule_scatterplot, create_3d_rule_visualization, create_rule_network,
    create_metrics_distribution, create_top_rules_table, convert_frozenset_to_str
)
from stream_app.utils.rule_mining import prune_redundant_rules, detect_insights
from stream_app.utils.insights import extract_actionable_insights

# Page configuration
st.set_page_config(
    page_title="Rule Explorer",
    page_icon="🔍",
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
    .filter-container {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .info-text {
        color: #555;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='page-title'>Association Rule Explorer</h1>", unsafe_allow_html=True)

# Check if rules exist in session state
if 'rules' not in st.session_state or st.session_state.rules is None or st.session_state.rules.empty:
    st.warning("No association rules found. Please go to the Home page and generate rules first.")
    st.stop()

# Rule count
st.markdown(f"<div class='highlight-box'>Currently analyzing {len(st.session_state.rules)} association rules.</div>", unsafe_allow_html=True)

# Create tabs for different exploration features
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Rule Visualization", "🔍 Rule Filtering", "🧠 Pattern Analysis", "📝 Export Results"
])

# Visualization tab
with tab1:
    st.markdown("<h2 class='section-title'>Rule Visualization</h2>", unsafe_allow_html=True)
    
    # Visualization type selector
    viz_type = st.selectbox(
        "Visualization Type",
        [
            "Scatter Plot (Support vs Confidence)", 
            "3D Visualization (Support, Confidence, Lift)",
            "Network Graph",
            "Metric Distributions"
        ]
    )
    
    # Get the rules to visualize (filtered or all)
    viz_rules = st.session_state.filtered_rules if 'filtered_rules' in st.session_state and st.session_state.filtered_rules is not None else st.session_state.rules
    
    # Create visualization based on selected type
    if viz_type == "Scatter Plot (Support vs Confidence)":
        # Options for scatter plot
        col1, col2 = st.columns(2)
        with col1:
            color_by = st.selectbox("Color by", ["lift", "confidence", "support"], index=0)
        with col2:
            size_by = st.selectbox("Size by", ["lift", "confidence", "support"], index=0)
        
        # Create and display plot
        fig = create_rule_scatterplot(viz_rules, colorby=color_by, sizeby=size_by)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class='info-text'>
            <strong>How to interpret:</strong> Each point represents an association rule. The position shows support (x-axis) 
            and confidence (y-axis), while color and size can be configured to show additional metrics. Hover over points 
            to see the specific rule details.
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_type == "3D Visualization (Support, Confidence, Lift)":
        # Create and display 3D plot
        fig = create_3d_rule_visualization(viz_rules)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class='info-text'>
            <strong>How to interpret:</strong> This 3D visualization shows rules in a space of support (x-axis), confidence (y-axis), 
            and lift (z-axis). The color intensity also represents lift. You can rotate, zoom, and pan to explore the rules from 
            different angles.
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_type == "Network Graph":
        # Options for network graph
        col1, col2 = st.columns(2)
        with col1:
            min_lift_network = st.slider(
                "Minimum Lift", 
                min_value=float(viz_rules['lift'].min()), 
                max_value=float(viz_rules['lift'].max()), 
                value=float(max(viz_rules['lift'].min(), 1.5)),
                format="%.1f"
            )
        with col2:
            max_rules_network = st.slider("Maximum Rules to Show", 5, 50, 20, step=5)
        
        # Create and display network
        fig = create_rule_network(viz_rules, min_lift=min_lift_network, max_rules=max_rules_network)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class='info-text'>
            <strong>How to interpret:</strong> The network graph shows how products are connected through association rules. 
            Each node is a product, and edges represent associations. Darker and thicker edges indicate stronger associations (higher lift).
            Node size and color represent the number of connections (degree).
        </div>
        """, unsafe_allow_html=True)
        
    elif viz_type == "Metric Distributions":
        # Options for distributions
        col1, col2 = st.columns(2)
        with col1:
            metric = st.selectbox("Metric to Analyze", ["lift", "confidence", "support"], index=0)
        with col2:
            nbins = st.slider("Number of Bins", 5, 50, 20, step=5)
        
        # Create and display histogram
        fig = create_metrics_distribution(viz_rules, metric=metric, nbins=nbins)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div class='info-text'>
            <strong>How to interpret:</strong> This histogram shows the distribution of the selected metric across all rules. 
            The vertical dashed line represents the mean value. A right-skewed distribution is common for lift values.
        </div>
        """, unsafe_allow_html=True)

# Rule filtering tab
with tab2:
    st.markdown("<h2 class='section-title'>Rule Filtering and Search</h2>", unsafe_allow_html=True)
    
    st.markdown("<div class='filter-container'>", unsafe_allow_html=True)
    
    # Create three columns for metric filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_support_filter = st.slider(
            "Minimum Support",
            min_value=float(st.session_state.rules['support'].min()),
            max_value=float(st.session_state.rules['support'].max()),
            value=float(st.session_state.rules['support'].min()),
            format="%.4f"
        )
    
    with col2:
        min_confidence_filter = st.slider(
            "Minimum Confidence",
            min_value=float(st.session_state.rules['confidence'].min()),
            max_value=float(st.session_state.rules['confidence'].max()),
            value=float(st.session_state.rules['confidence'].min()),
            format="%.2f"
        )
    
    with col3:
        min_lift_filter = st.slider(
            "Minimum Lift",
            min_value=float(st.session_state.rules['lift'].min()),
            max_value=float(st.session_state.rules['lift'].max()),
            value=float(st.session_state.rules['lift'].min()),
            format="%.2f"
        )
    
    # Additional advanced filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Filter by antecedent or consequent items
        item_search = st.text_input("Search for rules containing item(s)", "")
        
        # Search in antecedents, consequents, or both
        search_in = st.radio(
            "Search in:",
            ["Both", "Antecedents", "Consequents"],
            horizontal=True
        )
    
    with col2:
        # Filter by rule complexity (number of items)
        max_antecedent_size = st.slider(
            "Max Antecedent Size",
            1, 10, 10
        )
        
        # Option to prune redundant rules
        prune_redundant = st.checkbox("Prune Redundant Rules", value=False)
        
        # Sort order
        sort_by = st.selectbox(
            "Sort rules by",
            ["lift", "confidence", "support"],
            index=0
        )
        
        ascending = st.checkbox("Ascending order", value=False)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Apply filters button
    if st.button("Apply Filters"):
        with st.spinner("Filtering rules..."):
            # Start with all rules
            filtered = st.session_state.rules.copy()
            
            # Apply metric filters
            filtered = filtered[
                (filtered['support'] >= min_support_filter) &
                (filtered['confidence'] >= min_confidence_filter) &
                (filtered['lift'] >= min_lift_filter)
            ]
            
            # Apply complexity filter
            filtered = filtered[filtered['antecedents'].apply(lambda x: len(x) <= max_antecedent_size)]
            
            # Apply item search if specified
            if item_search:
                # Split by commas and strip whitespace
                search_items = [item.strip() for item in item_search.split(',')]
                
                if search_in == "Antecedents":
                    filtered = filtered[filtered['antecedents'].apply(
                        lambda x: any(search_item.lower() in item.lower() for search_item in search_items for item in x)
                    )]
                elif search_in == "Consequents":
                    filtered = filtered[filtered['consequents'].apply(
                        lambda x: any(search_item.lower() in item.lower() for search_item in search_items for item in x)
                    )]
                else:  # Both
                    filtered = filtered[
                        filtered['antecedents'].apply(
                            lambda x: any(search_item.lower() in item.lower() for search_item in search_items for item in x)
                        ) |
                        filtered['consequents'].apply(
                            lambda x: any(search_item.lower() in item.lower() for search_item in search_items for item in x)
                        )
                    ]
            
            # Prune redundant rules if requested
            if prune_redundant:
                filtered = prune_redundant_rules(filtered)
            
            # Sort the filtered rules
            filtered = filtered.sort_values(sort_by, ascending=ascending)
            
            # Save to session state
            st.session_state.filtered_rules = filtered
            
            # Show filter results
            st.success(f"Found {len(filtered)} rules matching your criteria")
    
    # Display filtered rules
    if 'filtered_rules' in st.session_state and st.session_state.filtered_rules is not None:
        # Show table of filtered rules
        st.markdown("<h3 class='section-title'>Filtered Rules</h3>", unsafe_allow_html=True)
        
        # Convert to display format and show
        display_rules = convert_frozenset_to_str(st.session_state.filtered_rules)
        
        # Round the metrics for better display
        for col in ['support', 'confidence', 'lift']:
            if col in display_rules.columns:
                display_rules[col] = display_rules[col].round(3)
        
        # Display as a dataframe
        st.dataframe(
            display_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']], 
            use_container_width=True
        )
    else:
        st.info("Apply filters to see the results")

# Pattern analysis tab
with tab3:
    st.markdown("<h2 class='section-title'>Rule Pattern Analysis</h2>", unsafe_allow_html=True)
    
    # Get the rules to analyze
    analyze_rules = st.session_state.filtered_rules if 'filtered_rules' in st.session_state and st.session_state.filtered_rules is not None else st.session_state.rules
    
    # Analysis type selector
    analysis_type = st.selectbox(
        "Analysis Type",
        ["Automated Insights", "Rule Clustering", "Frequent Items Analysis"]
    )
    
    if analysis_type == "Automated Insights":
        # Extract insights from the rules
        if st.button("Generate Insights"):
            with st.spinner("Analyzing rule patterns..."):
                # Extract insights
                insights = extract_actionable_insights(
                    analyze_rules,
                    min_lift=2.0,
                    min_confidence=0.5
                )
                
                # Display insights
                if insights:
                    for i, insight in enumerate(insights):
                        st.markdown(f"<div class='highlight-box'>{i+1}. {insight}</div>", unsafe_allow_html=True)
                else:
                    st.info("No significant insights found. Try adjusting the filtering criteria.")
    
    elif analysis_type == "Rule Clustering":
        st.markdown("### Rule Similarity Analysis")
        
        # Option for clustering method
        cluster_by = st.selectbox(
            "Cluster rules by",
            ["Antecedent Similarity", "Consequent Similarity", "Overall Similarity"]
        )
        
        # Number of clusters
        num_clusters = st.slider("Number of Clusters", 2, 10, 5)
        
        if st.button("Cluster Rules"):
            with st.spinner("Clustering rules..."):
                # Prepare for clustering
                from sklearn.feature_extraction.text import CountVectorizer
                from sklearn.cluster import KMeans
                
                # Convert rules to strings for vectorization
                display_rules = convert_frozenset_to_str(analyze_rules)
                
                # Feature to cluster on
                if cluster_by == "Antecedent Similarity":
                    feature = display_rules['antecedents']
                elif cluster_by == "Consequent Similarity":
                    feature = display_rules['consequents']
                else:  # Overall Similarity
                    feature = display_rules['antecedents'] + " → " + display_rules['consequents']
                
                # Vectorize the features
                vectorizer = CountVectorizer()
                X = vectorizer.fit_transform(feature)
                
                # Perform clustering
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                display_rules['cluster'] = kmeans.fit_predict(X)
                
                # Show clusters
                for i in range(num_clusters):
                    cluster_rules = display_rules[display_rules['cluster'] == i]
                    
                    with st.expander(f"Cluster {i+1} ({len(cluster_rules)} rules)"):
                        # Show top rules in cluster by lift
                        top_cluster_rules = cluster_rules.sort_values('lift', ascending=False).head(5)
                        st.dataframe(
                            top_cluster_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']],
                            use_container_width=True
                        )
                        
                        # Get common items in this cluster
                        if cluster_by == "Antecedent Similarity" or cluster_by == "Overall Similarity":
                            antecedent_words = ' '.join(cluster_rules['antecedents']).split(', ')
                            ant_word_counts = pd.Series(antecedent_words).value_counts()
                            if not ant_word_counts.empty:
                                st.markdown("**Common antecedent items in cluster:**")
                                st.write(', '.join(ant_word_counts.index[:5]))
                        
                        if cluster_by == "Consequent Similarity" or cluster_by == "Overall Similarity":
                            consequent_words = ' '.join(cluster_rules['consequents']).split(', ')
                            cons_word_counts = pd.Series(consequent_words).value_counts()
                            if not cons_word_counts.empty:
                                st.markdown("**Common consequent items in cluster:**")
                                st.write(', '.join(cons_word_counts.index[:5]))
    
    elif analysis_type == "Frequent Items Analysis":
        st.markdown("### Frequent Items Analysis")
        
        # Analyze frequency of items in rules
        if st.button("Analyze Item Frequency"):
            with st.spinner("Analyzing item frequencies..."):
                # Count item frequency
                from collections import Counter
                
                # Extract all items from antecedents and consequents
                all_items = []
                for _, rule in analyze_rules.iterrows():
                    all_items.extend(list(rule['antecedents']))
                    all_items.extend(list(rule['consequents']))
                
                # Count frequencies
                item_counts = Counter(all_items)
                
                # Convert to dataframe for display
                item_freq_df = pd.DataFrame({
                    'Item': list(item_counts.keys()),
                    'Frequency': list(item_counts.values())
                }).sort_values('Frequency', ascending=False)
                
                # Display as bar chart
                fig = px.bar(
                    item_freq_df.head(20),
                    x='Item',
                    y='Frequency',
                    title="Top 20 Most Frequent Items in Rules",
                    labels={"Item": "Product", "Frequency": "Frequency in Rules"},
                    color='Frequency',
                    color_continuous_scale="Blues"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display as table
                st.dataframe(item_freq_df, use_container_width=True)

# Export results tab
with tab4:
    st.markdown("<h2 class='section-title'>Export Analysis Results</h2>", unsafe_allow_html=True)
    
    # Get the rules to export
    export_rules = st.session_state.filtered_rules if 'filtered_rules' in st.session_state and st.session_state.filtered_rules is not None else st.session_state.rules
    
    # Convert rules to a display-friendly format
    display_export_rules = convert_frozenset_to_str(export_rules)
    
    # Export format options
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "Excel", "JSON"]
    )
    
    # Add options for what to include
    st.markdown("### Export Options")
    
    include_all_metrics = st.checkbox("Include all additional metrics", value=True)
    include_insights = st.checkbox("Include automated insights", value=True)
    
    # Export button
    if st.button("Generate Export"):
        with st.spinner("Preparing export..."):
            if export_format == "CSV":
                # Prepare CSV
                csv = display_export_rules.to_csv(index=False)
                
                # Create download button
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="association_rules.csv",
                    mime="text/csv"
                )
                
            elif export_format == "Excel":
                # Prepare Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    display_export_rules.to_excel(writer, sheet_name='Rules', index=False)
                    
                    # Add insights if requested
                    if include_insights:
                        insights = extract_actionable_insights(
                            export_rules,
                            min_lift=2.0,
                            min_confidence=0.5
                        )
                        
                        insights_df = pd.DataFrame({
                            'Insight': insights
                        })
                        
                        insights_df.to_excel(writer, sheet_name='Insights', index=False)
                
                output.seek(0)
                
                # Create download button
                st.download_button(
                    label="Download Excel",
                    data=output,
                    file_name="association_rules.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
            elif export_format == "JSON":
                # Convert to JSON serializable format
                json_rules = []
                
                for _, rule in display_export_rules.iterrows():
                    rule_dict = {
                        'antecedents': rule['antecedents'],
                        'consequents': rule['consequents'],
                        'support': float(rule['support']),
                        'confidence': float(rule['confidence']),
                        'lift': float(rule['lift'])
                    }
                    
                    # Add additional metrics if requested
                    if include_all_metrics:
                        for col in display_export_rules.columns:
                            if col not in ['antecedents', 'consequents', 'support', 'confidence', 'lift']:
                                rule_dict[col] = float(rule[col]) if isinstance(rule[col], (int, float)) else rule[col]
                    
                    json_rules.append(rule_dict)
                
                # Create JSON string
                json_str = json.dumps({'rules': json_rules}, indent=2)
                
                # Create download button
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="association_rules.json",
                    mime="application/json"
                )
    
    # Display preview
    st.markdown("### Export Preview")
    st.dataframe(display_export_rules.head(10), use_container_width=True) 