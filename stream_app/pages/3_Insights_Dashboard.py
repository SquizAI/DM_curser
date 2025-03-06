import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import sys
from typing import List, Dict, Optional, Tuple, Union, Set
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from io import BytesIO

# Add parent directory to path to allow importing from utils
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import utility functions
from stream_app.utils.visualizations import (
    convert_frozenset_to_str, create_rule_scatterplot, create_metrics_distribution,
    create_rule_network, create_top_rules_table
)
from stream_app.utils.insights import (
    extract_actionable_insights, analyze_category_relationships, identify_frequent_items
)

# Page configuration
st.set_page_config(
    page_title="Business Insights Dashboard",
    page_icon="📈",
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
    .insight-card {
        background-color: white;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 15px;
        border-left: 5px solid #1E88E5;
    }
    .insight-title {
        font-weight: 600;
        color: #0D47A1;
        margin-bottom: 5px;
    }
    .insight-category {
        font-size: 0.8rem;
        color: #1E88E5;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .insight-text {
        color: #333;
    }
    .metric-container {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 1rem;
        color: #555;
    }
    .report-section {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Page title
st.markdown("<h1 class='page-title'>Business Insights Dashboard</h1>", unsafe_allow_html=True)

# Check if rules exist in session state
if 'rules' not in st.session_state or st.session_state.rules is None or st.session_state.rules.empty:
    st.warning("No association rules found. Please go to the Home page and generate rules first.")
    st.stop()

# Create a tabbed interface for the dashboard
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Executive Summary", "📋 Key Insights", "🔍 Deep Dive", "📑 Report Generator"
])

# Get key metrics for the dashboard
rules_count = len(st.session_state.rules)
avg_lift = st.session_state.rules['lift'].mean()
strong_rules = len(st.session_state.rules[st.session_state.rules['lift'] > 2])
frequent_items = []

# Extract insights if not already done
if 'business_insights' not in st.session_state or not st.session_state.business_insights:
    with st.spinner("Generating business insights..."):
        st.session_state.business_insights = extract_actionable_insights(
            st.session_state.rules,
            min_lift=1.5,
            min_confidence=0.4
        )

# Executive Summary tab
with tab1:
    st.markdown("<h2 class='section-title'>Executive Summary</h2>", unsafe_allow_html=True)
    
    # Top level metrics
    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Association Rules</div>
        </div>
        """.format(rules_count), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:.2f}</div>
            <div class="metric-label">Average Lift</div>
        </div>
        """.format(avg_lift), unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-container">
            <div class="metric-value">{:,}</div>
            <div class="metric-label">Strong Associations</div>
        </div>
        """.format(strong_rules), unsafe_allow_html=True)
    
    with col4:
        if 'df' in st.session_state and st.session_state.df is not None:
            product_count = len(st.session_state.df['Description'].unique())
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">{:,}</div>
                <div class="metric-label">Unique Products</div>
            </div>
            """.format(product_count), unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-value">-</div>
                <div class="metric-label">Unique Products</div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Key visualization
    st.markdown("<h3 class='section-title'>Key Patterns Overview</h3>", unsafe_allow_html=True)
    
    # Show top rules by lift
    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
    
    top_rules = create_top_rules_table(
        st.session_state.rules,
        sort_by='lift',
        top_n=5
    )
    
    st.dataframe(top_rules, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Summary visualization
    st.markdown("<h3 class='section-title'>Relationship Strength Distribution</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        lift_dist_fig = create_metrics_distribution(
            st.session_state.rules,
            metric='lift',
            nbins=20
        )
        st.plotly_chart(lift_dist_fig, use_container_width=True)
    
    with col2:
        confidence_dist_fig = create_metrics_distribution(
            st.session_state.rules,
            metric='confidence',
            nbins=20
        )
        st.plotly_chart(confidence_dist_fig, use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Top insights summary
    st.markdown("<h3 class='section-title'>Executive Insights</h3>", unsafe_allow_html=True)
    
    st.markdown("<div class='report-section'>", unsafe_allow_html=True)
    
    if st.session_state.business_insights:
        for i, insight in enumerate(st.session_state.business_insights[:3]):
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-category">Key Insight #{i+1}</div>
                <div class="insight-text">{insight}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No business insights available. Generate association rules with lower thresholds to discover more patterns.")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Key Insights tab
with tab2:
    st.markdown("<h2 class='section-title'>Key Business Insights</h2>", unsafe_allow_html=True)
    
    # Categorize insights
    if st.session_state.business_insights:
        # Define insight categories
        categories = {
            "Cross-sell Opportunities": [],
            "Bundling Suggestions": [],
            "Hidden Gems": [],
            "Strong Associations": []
        }
        
        # Categorize insights based on keywords
        for insight in st.session_state.business_insights:
            if "Cross-sell" in insight:
                categories["Cross-sell Opportunities"].append(insight)
            elif "bundle" in insight.lower() or "Bundling" in insight:
                categories["Bundling Suggestions"].append(insight)
            elif "Hidden gem" in insight:
                categories["Hidden Gems"].append(insight)
            elif "Strong complementary" in insight:
                categories["Strong Associations"].append(insight)
        
        # Create tabs for different categories
        cat_tabs = st.tabs(list(categories.keys()))
        
        for i, (category, insights) in enumerate(categories.items()):
            with cat_tabs[i]:
                if insights:
                    for j, insight in enumerate(insights):
                        st.markdown(f"""
                        <div class="insight-card">
                            <div class="insight-text">{insight}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info(f"No {category.lower()} found. Try adjusting rule parameters or analyzing more data.")
        
        # Additional action recommendations
        st.markdown("<h3 class='section-title'>Recommended Actions</h3>", unsafe_allow_html=True)
        
        # Cross-sell recommendations
        if categories["Cross-sell Opportunities"]:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">Cross-selling Strategy</div>
                <div class="insight-text">
                    Implement targeted cross-selling campaigns for identified product pairs.
                    Consider using these patterns in your e-commerce recommendation system and
                    train sales staff to suggest complementary products.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Bundle recommendations
        if categories["Bundling Suggestions"]:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">Product Bundling</div>
                <div class="insight-text">
                    Create special bundle offers with discounted prices for the identified product combinations.
                    Test different price points to maximize revenue while incentivizing bundle purchases.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Hidden gems recommendations
        if categories["Hidden Gems"]:
            st.markdown("""
            <div class="insight-card">
                <div class="insight-title">Explore Niche Markets</div>
                <div class="insight-text">
                    The identified "hidden gems" represent potential niche market segments.
                    Consider targeted marketing campaigns to these specific customer segments
                    to grow these initially small but highly engaged markets.
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No business insights available. Generate association rules with lower thresholds to discover more patterns.")

# Deep Dive tab
with tab3:
    st.markdown("<h2 class='section-title'>Association Pattern Analysis</h2>", unsafe_allow_html=True)
    
    # Network visualization of associations
    st.markdown("<h3 class='section-title'>Product Association Network</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        min_lift_network = st.slider(
            "Minimum Association Strength (Lift)",
            min_value=float(st.session_state.rules['lift'].min()),
            max_value=float(min(st.session_state.rules['lift'].max(), 10.0)),
            value=float(max(2.0, st.session_state.rules['lift'].min())),
            step=0.1
        )
        
        max_rules_network = st.slider(
            "Maximum Rules to Visualize",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
    
    with col2:
        # Create and display network
        network_fig = create_rule_network(
            st.session_state.rules,
            min_lift=min_lift_network,
            max_rules=max_rules_network
        )
        st.plotly_chart(network_fig, use_container_width=True)
    
    # Frequent items analysis
    st.markdown("<h3 class='section-title'>Frequent Items Analysis</h3>", unsafe_allow_html=True)
    
    # Identify frequent items
    with st.spinner("Analyzing frequent items..."):
        frequent_items = identify_frequent_items(st.session_state.rules)
        
        if frequent_items:
            # Convert to dataframe
            freq_df = pd.DataFrame({
                'Item': [item[0] for item in frequent_items[:20]],
                'Frequency': [item[1] for item in frequent_items[:20]]
            })
            
            # Create visualization
            fig = px.bar(
                freq_df,
                x='Item',
                y='Frequency',
                title="Top 20 Most Frequent Items in Association Rules",
                color='Frequency',
                color_continuous_scale='Blues',
                labels={
                    'Item': 'Product',
                    'Frequency': 'Appearance Frequency in Rules'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add insights about frequent items
            st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
            st.markdown("<div class='insight-title'>Frequent Items Insights</div>", unsafe_allow_html=True)
            st.markdown(
                "These items appear most frequently in association rules and represent key products "
                "that drive purchasing patterns. Consider these as anchor products that can be "
                "used to generate additional sales through cross-selling and bundling strategies."
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("No frequent items data available.")
    
    # Pattern consistency analysis
    st.markdown("<h3 class='section-title'>Rule Metrics Analysis</h3>", unsafe_allow_html=True)
    
    # Create a scatter plot of support vs confidence
    scatter_fig = create_rule_scatterplot(
        st.session_state.rules,
        colorby='lift',
        sizeby='lift'
    )
    st.plotly_chart(scatter_fig, use_container_width=True)
    
    # Add interpretation
    st.markdown("<div class='insight-card'>", unsafe_allow_html=True)
    st.markdown("<div class='insight-title'>Interpreting the Scatter Plot</div>", unsafe_allow_html=True)
    st.markdown("""
    - **Support (x-axis)**: How frequently the items appear together in the dataset
    - **Confidence (y-axis)**: How likely item B is purchased when item A is purchased
    - **Color/Size**: Lift value, which measures how much more likely items are purchased together beyond random chance
    
    Rules in the upper-right quadrant have both high support and confidence, making them the most reliable patterns.
    Rules with high lift (larger, darker points) represent the strongest associations regardless of frequency.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Report Generator tab
with tab4:
    st.markdown("<h2 class='section-title'>Business Report Generator</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <div class="insight-text">
            Generate a comprehensive business report with visualizations and insights that can be shared with stakeholders.
            Customize the report content using the options below.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Report configuration options
    st.markdown("<h3 class='section-title'>Report Configuration</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input("Report Title", "Association Rule Mining Analysis Report")
        include_executive_summary = st.checkbox("Include Executive Summary", value=True)
        include_insights = st.checkbox("Include Business Insights", value=True)
        include_visualizations = st.checkbox("Include Visualizations", value=True)
        include_recommendations = st.checkbox("Include Action Recommendations", value=True)
    
    with col2:
        report_author = st.text_input("Report Author", "Business Analytics Team")
        current_date = pd.Timestamp.now().strftime("%Y-%m-%d")
        report_date = st.date_input("Report Date", pd.Timestamp.now())
        confidentiality = st.selectbox(
            "Confidentiality Level",
            ["Public", "Internal Use Only", "Confidential", "Strictly Confidential"]
        )
    
    # Generate report button
    if st.button("Generate Report"):
        with st.spinner("Generating business report..."):
            # Create HTML content for the report
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report_title}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .title {{ font-size: 24px; font-weight: bold; color: #0D47A1; }}
                    .subtitle {{ font-size: 16px; color: #555; }}
                    .section {{ margin-top: 30px; margin-bottom: 30px; }}
                    .section-title {{ font-size: 20px; font-weight: bold; color: #0D47A1; border-bottom: 1px solid #ddd; padding-bottom: 5px; }}
                    .insight {{ background-color: #f0f7ff; padding: 15px; border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #1E88E5; }}
                    .metric {{ display: inline-block; width: 22%; text-align: center; background-color: #f5f5f5; padding: 15px; margin: 10px; border-radius: 5px; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #1E88E5; }}
                    .metric-label {{ font-size: 14px; color: #555; }}
                    .recommendation {{ background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-bottom: 15px; }}
                    .footer {{ margin-top: 50px; text-align: center; font-size: 12px; color: #777; border-top: 1px solid #ddd; padding-top: 20px; }}
                    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    .confidential {{ color: red; font-weight: bold; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <div class="title">{report_title}</div>
                    <div class="subtitle">Generated on {report_date} | Prepared by: {report_author}</div>
                    <div class="confidential">{confidentiality}</div>
                </div>
            """
            
            # Executive Summary section
            if include_executive_summary:
                html_content += f"""
                <div class="section">
                    <div class="section-title">Executive Summary</div>
                    <p>
                        This report presents the findings from association rule mining analysis of transaction data.
                        A total of {rules_count} association rules were discovered, with an average lift of {avg_lift:.2f}.
                        {strong_rules} strong associations (lift > 2) were identified, representing significant opportunities 
                        for cross-selling, bundling, and targeted marketing strategies.
                    </p>
                    
                    <div class="metrics">
                        <div class="metric">
                            <div class="metric-value">{rules_count}</div>
                            <div class="metric-label">Association Rules</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{avg_lift:.2f}</div>
                            <div class="metric-label">Average Lift</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{strong_rules}</div>
                            <div class="metric-label">Strong Associations</div>
                        </div>
                """
                
                if 'df' in st.session_state and st.session_state.df is not None:
                    product_count = len(st.session_state.df['Description'].unique())
                    html_content += f"""
                        <div class="metric">
                            <div class="metric-value">{product_count}</div>
                            <div class="metric-label">Unique Products</div>
                        </div>
                    """
                
                html_content += """
                    </div>
                </div>
                """
            
            # Business Insights section
            if include_insights and st.session_state.business_insights:
                html_content += """
                <div class="section">
                    <div class="section-title">Key Business Insights</div>
                """
                
                for i, insight in enumerate(st.session_state.business_insights[:5]):
                    html_content += f"""
                    <div class="insight">
                        <strong>Insight {i+1}:</strong> {insight}
                    </div>
                    """
                
                html_content += """
                </div>
                """
            
            # Visualizations section (placeholders since we can't easily include images)
            if include_visualizations:
                html_content += """
                <div class="section">
                    <div class="section-title">Visualization & Analysis</div>
                    <p>
                        The following visualizations provide deeper insights into the discovered association patterns.
                        (Note: In a complete report, actual visualizations would be embedded here.)
                    </p>
                    
                    <h4>Top Association Rules</h4>
                    <table>
                        <tr>
                            <th>Antecedents</th>
                            <th>Consequents</th>
                            <th>Support</th>
                            <th>Confidence</th>
                            <th>Lift</th>
                        </tr>
                """
                
                # Convert top rules to display format
                top_display_rules = convert_frozenset_to_str(
                    st.session_state.rules.sort_values('lift', ascending=False).head(5)
                )
                
                for _, rule in top_display_rules.iterrows():
                    html_content += f"""
                    <tr>
                        <td>{rule['antecedents']}</td>
                        <td>{rule['consequents']}</td>
                        <td>{rule['support']:.3f}</td>
                        <td>{rule['confidence']:.3f}</td>
                        <td>{rule['lift']:.3f}</td>
                    </tr>
                    """
                
                html_content += """
                    </table>
                </div>
                """
            
            # Recommendations section
            if include_recommendations:
                html_content += """
                <div class="section">
                    <div class="section-title">Recommended Actions</div>
                """
                
                # Cross-sell recommendations
                html_content += """
                <div class="recommendation">
                    <strong>Implement Cross-selling Strategies:</strong>
                    <ul>
                        <li>Update the recommendation engine on the e-commerce platform to suggest products based on discovered associations</li>
                        <li>Train sales staff to suggest complementary products based on the strongest associations</li>
                        <li>Create targeted email campaigns featuring complementary products for recent purchasers</li>
                    </ul>
                </div>
                """
                
                # Bundling recommendations
                html_content += """
                <div class="recommendation">
                    <strong>Product Bundling Opportunities:</strong>
                    <ul>
                        <li>Create special bundle offers with discounted prices for the identified product combinations</li>
                        <li>Feature high-confidence product combinations in marketing materials and store displays</li>
                        <li>Test different bundle price points to maximize revenue while incentivizing bundle purchases</li>
                    </ul>
                </div>
                """
                
                # Marketing recommendations
                html_content += """
                <div class="recommendation">
                    <strong>Marketing Strategy Adjustments:</strong>
                    <ul>
                        <li>Redesign store layout to place frequently associated items within proximity</li>
                        <li>Develop targeted marketing campaigns for "hidden gem" product associations</li>
                        <li>Use high-lift associations in promotional materials to increase conversion rates</li>
                    </ul>
                </div>
                """
                
                html_content += """
                </div>
                """
            
            # Footer
            html_content += f"""
                <div class="footer">
                    {confidentiality} | Generated on {report_date} | {report_author}<br>
                    Association Rule Mining Analysis Report
                </div>
            </body>
            </html>
            """
            
            # Display a download button for the HTML report
            st.download_button(
                label="Download HTML Report",
                data=html_content,
                file_name=f"association_rules_report_{current_date}.html",
                mime="text/html"
            )
            
            # Display report preview
            st.markdown("<h3 class='section-title'>Report Preview</h3>", unsafe_allow_html=True)
            
            # Show in an iframe (limited preview)
            st.components.v1.html(html_content, height=600, scrolling=True) 