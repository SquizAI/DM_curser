import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import sys
import os

# Add the utils directory to the path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.visualizations import (
    create_item_frequency_chart, create_association_network, 
    create_rule_strength_scatter, create_lift_histogram,
    create_wordcloud_from_rules, create_heatmap_item_combinations,
    create_sunburst_chart
)
from utils.rule_mining import detect_insights, prune_redundant_rules

st.set_page_config(page_title="Business Insights", page_icon="💡", layout="wide")

# Add custom CSS
st.markdown("""
<style>
    .insight-card {
        background-color: rgba(30, 34, 45, 0.5);
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        border-left: 5px solid #1E88E5;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        color: rgba(250, 250, 250, 0.9);
    }
    
    .insight-title {
        color: #1E88E5;
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 10px;
    }
    
    .insight-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: rgba(250, 250, 250, 0.9);
    }
    
    .insight-description {
        color: rgba(220, 220, 220, 0.8);
        margin-top: 10px;
    }
    
    .recommendation-box {
        background-color: rgba(30, 136, 229, 0.2);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #1E88E5;
        color: rgba(250, 250, 250, 0.9);
    }
    
    .recommendation-header {
        font-weight: 600;
        color: #90CAF9;
        margin-bottom: 10px;
    }
    
    .categorical-insight {
        border-left-color: #4CAF50;
    }
    
    .sales-insight {
        border-left-color: #FF9800;
    }
    
    .customer-insight {
        border-left-color: #9C27B0;
    }
    
    .strategic-insight {
        border-left-color: #F44336;
    }
    
    .custom-tab {
        border-radius: 5px 5px 0 0;
        padding: 10px 15px;
        font-weight: 500;
    }
    
    .stat-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    
    .stat-box {
        flex: 1;
        min-width: 200px;
        background-color: rgba(30, 34, 45, 0.5);
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        text-align: center;
        color: rgba(250, 250, 250, 0.9);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
    }
    
    .stat-label {
        font-size: 1rem;
        color: rgba(200, 200, 200, 0.8);
    }
    
    /* Fix tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(50, 50, 50, 0.2);
        color: rgba(250, 250, 250, 0.9);
        border-radius: 4px 4px 0 0;
        padding: 10px 16px;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    
    /* Make sure data elements are visible */
    .stDataFrame table, .stTable table {
        color: rgba(250, 250, 250, 0.9);
    }
    
    .stDataFrame th, .stTable th {
        background-color: rgba(30, 136, 229, 0.2);
        color: rgba(250, 250, 250, 0.9);
    }
    
    /* Make sure all text is visible */
    p, h1, h2, h3, h4, h5, h6, span, label, .stMarkdown {
        color: rgba(250, 250, 250, 0.9) !important;
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.title("💡 Business Insights & Recommendations")
st.markdown("""
This page analyzes the generated association rules to extract actionable business insights and provides 
strategic recommendations to drive business value. Explore the findings across different categories to 
inform your decision-making.
""")

# Check if rules are available in session state
if 'rules' not in st.session_state or st.session_state.rules is None or st.session_state.rules.empty:
    st.warning("⚠️ No rules found! Please generate association rules from the main page first.")
    st.stop()

# Get data from session state
rules_df = st.session_state.rules
df = st.session_state.df

# Generate insights based on rules
def generate_business_insights(rules_df, transaction_df):
    """Generate business insights from rules and transaction data"""
    insights = {}
    
    # Basic rule statistics
    insights['total_rules'] = len(rules_df)
    insights['avg_lift'] = rules_df['lift'].mean()
    insights['max_lift'] = rules_df['lift'].max()
    insights['strong_associations'] = len(rules_df[rules_df['lift'] > 3])
    
    # Product insights
    antecedent_items = set()
    for items in rules_df['antecedents']:
        antecedent_items.update(items)
    insights['unique_antecedent_items'] = len(antecedent_items)
    
    consequent_items = set()
    for items in rules_df['consequents']:
        consequent_items.update(items)
    insights['unique_consequent_items'] = len(consequent_items)
    
    # Get high lift rules
    high_lift_rules = rules_df[rules_df['lift'] > 5].sort_values('lift', ascending=False)
    if not high_lift_rules.empty:
        insights['top_associations'] = []
        for _, row in high_lift_rules.head(5).iterrows():
            antecedents = ', '.join(list(row['antecedents']))
            consequents = ', '.join(list(row['consequents']))
            insights['top_associations'].append({
                'antecedents': antecedents,
                'consequents': consequents,
                'lift': row['lift'],
                'confidence': row['confidence']
            })
    
    # Key product insights
    product_frequencies = transaction_df['Description'].value_counts()
    insights['top_products'] = [
        (product, count) for product, count in 
        product_frequencies.head(10).items()
    ]
    
    # Find products that often drive other purchases (frequent in antecedents)
    antecedent_counts = {}
    for items in rules_df['antecedents']:
        for item in items:
            antecedent_counts[item] = antecedent_counts.get(item, 0) + 1
    
    if antecedent_counts:
        insights['top_drivers'] = sorted(
            antecedent_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:7]
    
    # Find products that are often purchased as a result (frequent in consequents)
    consequent_counts = {}
    for items in rules_df['consequents']:
        for item in items:
            consequent_counts[item] = consequent_counts.get(item, 0) + 1
    
    if consequent_counts:
        insights['top_influenced'] = sorted(
            consequent_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:7]
    
    # Find hidden gems - less frequent items but strong associations
    potential_gems = []
    for _, row in rules_df.iterrows():
        for item in row['consequents']:
            if item in product_frequencies and product_frequencies[item] < product_frequencies.quantile(0.25):
                if row['lift'] > 3 and row['confidence'] > 0.5:
                    antecedents = ', '.join(list(row['antecedents']))
                    potential_gems.append({
                        'item': item,
                        'antecedents': antecedents,
                        'lift': row['lift'],
                        'confidence': row['confidence']
                    })
    
    # Sort by lift and get unique hidden gems
    if potential_gems:
        unique_gems = {}
        for gem in sorted(potential_gems, key=lambda x: x['lift'], reverse=True):
            if gem['item'] not in unique_gems:
                unique_gems[gem['item']] = gem
        
        insights['hidden_gems'] = list(unique_gems.values())[:5]  # Top 5 unique gems
    
    # Use the detect_insights function from rule_mining
    rule_insights = detect_insights(rules_df)
    insights['automatic_insights'] = rule_insights
    
    return insights

# Generate insights
with st.spinner("Analyzing rules and generating insights..."):
    insights = generate_business_insights(rules_df, df)

# Display overview statistics
st.markdown("## 📊 Overview Statistics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">{:,}</div>
        <div class="stat-label">Total Rules</div>
    </div>
    """.format(insights['total_rules']), unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">{:.2f}</div>
        <div class="stat-label">Average Lift</div>
    </div>
    """.format(insights['avg_lift']), unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">{:,}</div>
        <div class="stat-label">Strong Associations</div>
    </div>
    """.format(insights['strong_associations']), unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div class="stat-box">
        <div class="stat-value">{:.2f}</div>
        <div class="stat-label">Max Lift Value</div>
    </div>
    """.format(insights['max_lift']), unsafe_allow_html=True)

# Create tabs for different insights
insight_tabs = st.tabs([
    "🔑 Key Findings", 
    "🛒 Product Associations", 
    "📈 Visual Insights",
    "💰 Revenue Opportunities",
    "🔮 Advanced Analysis"
])

with insight_tabs[0]:  # Key Findings
    st.markdown("### Key Business Insights")
    
    # Automatic insights
    if 'automatic_insights' in insights:
        for category, category_insights in insights['automatic_insights'].items():
            if category_insights:  # Only show if there are insights
                st.markdown(f"#### {category.title()}")
                
                for i, insight in enumerate(category_insights[:5]):  # Limit to top 5
                    css_class = "insight-card"
                    if "product" in category.lower():
                        css_class += " categorical-insight"
                    elif "sales" in category.lower():
                        css_class += " sales-insight"
                    elif "customer" in category.lower():
                        css_class += " customer-insight"
                    else:
                        css_class += " strategic-insight"
                        
                    st.markdown(f"""
                    <div class="{css_class}">
                        <div class="insight-title">Insight #{i+1}</div>
                        <div class="insight-description">{insight}</div>
                    </div>
                    """, unsafe_allow_html=True)
    
    # Display top associations
    if 'top_associations' in insights and insights['top_associations']:
        st.markdown("#### Strongest Product Associations")
        
        for i, assoc in enumerate(insights['top_associations']):
            st.markdown(f"""
            <div class="insight-card">
                <div class="insight-title">Association #{i+1}</div>
                <div class="insight-description">
                    <b>If customers buy:</b> {assoc['antecedents']}<br>
                    <b>They are {assoc['lift']:.2f}x more likely to buy:</b> {assoc['consequents']}<br>
                    <b>Confidence:</b> {assoc['confidence']:.2f} ({assoc['confidence']*100:.0f}% of the time)
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Strategic recommendations
    st.markdown("### 🚀 Strategic Recommendations")
    
    # Generate recommendations based on insights
    recommendations = [
        {
            "title": "Product Placement Optimization",
            "description": "Place frequently associated products closer together in store layouts or on website pages to increase cross-selling potential.",
            "action_items": [
                "Reorganize store layout based on top 5 product associations",
                "Create 'Frequently Bought Together' sections online using strong rules",
                "Train sales staff on top complementary products"
            ]
        },
        {
            "title": "Promotional Strategy Enhancement",
            "description": "Create targeted bundle offers and promotions based on products with high lift values to maximize revenue.",
            "action_items": [
                "Develop seasonal bundles using top product associations",
                "Offer discounts on consequent items when antecedent items are purchased",
                "Design email marketing campaigns featuring complementary products"
            ]
        },
        {
            "title": "Inventory Management Improvements",
            "description": "Ensure associated products are adequately stocked together to prevent missed sales opportunities.",
            "action_items": [
                "Adjust reorder points for complementary products",
                "Monitor stock levels of products within strong association rules",
                "Create alerts when inventory of associated products becomes unbalanced"
            ]
        }
    ]
    
    # Add recommendations for hidden gems if available
    if 'hidden_gems' in insights and insights['hidden_gems']:
        gems_rec = {
            "title": "Hidden Gem Product Promotion",
            "description": "Promote less popular products that have strong associations to boost sales of underperforming inventory.",
            "action_items": [
                f"Feature {insights['hidden_gems'][0]['item']} more prominently when {insights['hidden_gems'][0]['antecedents']} are purchased",
                "Create educational content about the hidden gem products to increase awareness",
                "Offer trial or sample programs for less-known but complementary products"
            ]
        }
        recommendations.append(gems_rec)
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation-box">
            <div class="recommendation-header">{rec['title']}</div>
            <p>{rec['description']}</p>
            <ul>
                {"".join(f"<li>{item}</li>" for item in rec['action_items'])}
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
with insight_tabs[1]:  # Product Associations
    st.markdown("### Product Association Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Top products frequency chart
        st.markdown("#### Top Products by Frequency")
        freq_chart = create_item_frequency_chart(df)
        st.plotly_chart(freq_chart, use_container_width=True, key="item_frequency_chart")
    
    with col2:
        # Top drivers and influenced products
        st.markdown("#### Key Product Roles")
        
        st.markdown("##### Top Purchase Drivers")
        if 'top_drivers' in insights:
            driver_df = pd.DataFrame(insights['top_drivers'], columns=['Product', 'Frequency in Rules'])
            driver_df = driver_df.head(5)  # Limit to top 5
            st.dataframe(driver_df, use_container_width=True)
        
        st.markdown("##### Most Influenced Products")
        if 'top_influenced' in insights:
            influenced_df = pd.DataFrame(insights['top_influenced'], columns=['Product', 'Frequency in Rules'])
            influenced_df = influenced_df.head(5)  # Limit to top 5
            st.dataframe(influenced_df, use_container_width=True)
    
    # Network visualization with rules
    st.markdown("#### Product Association Network")
    # Add a slider to filter by lift
    min_lift_filter = st.slider("Minimum Lift for Visualization", 1.0, 10.0, 2.0, 0.1)
    
    network_fig = create_association_network(rules_df, min_lift=min_lift_filter)
    if network_fig:
        st.plotly_chart(network_fig, use_container_width=True, key="association_network")
    
    # Co-occurrence heatmap
    st.markdown("#### Product Co-occurrence Heatmap")
    top_n = st.slider("Number of Top Products", 5, 20, 10)
    
    heatmap_fig = create_heatmap_item_combinations(df, rules_df, top_n=top_n)
    if heatmap_fig:
        st.plotly_chart(heatmap_fig, use_container_width=True, key="association_heatmap")
        
with insight_tabs[2]:  # Visual Insights
    st.markdown("### Data Visualization & Pattern Exploration")
    
    # Rule strength visualization
    st.markdown("#### Rule Strength Visualization")
    st.markdown("This scatter plot shows the relationship between support, confidence, and lift for all rules.")
    
    rule_scatter = create_rule_strength_scatter(rules_df)
    if rule_scatter:
        st.plotly_chart(rule_scatter, use_container_width=True, key="rule_scatter_plot")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Lift histogram
        lift_hist = create_lift_histogram(rules_df)
        if lift_hist:
            st.plotly_chart(lift_hist, use_container_width=True, key="lift_histogram")
    
    with col2:
        # Word cloud visualization
        st.markdown("#### Word Cloud of Products in Rules")
        wordcloud_type = st.selectbox(
            "Select item type for word cloud", 
            ["all", "antecedents", "consequents"],
            format_func=lambda x: {
                "all": "All Products", 
                "antecedents": "If Products (Antecedents)", 
                "consequents": "Then Products (Consequents)"
            }[x]
        )
        
        wordcloud_fig = create_wordcloud_from_rules(rules_df, type_filter=wordcloud_type)
        if wordcloud_fig:
            st.pyplot(wordcloud_fig)
    
    # Hierarchical visualization
    st.markdown("#### Hierarchical Product Relationships")
    sunburst_fig = create_sunburst_chart(rules_df)
    if sunburst_fig:
        st.plotly_chart(sunburst_fig, use_container_width=True, key="product_hierarchy_sunburst")
        
with insight_tabs[3]:  # Revenue Opportunities
    st.markdown("### 💰 Revenue Growth Opportunities")
    
    # Cross-selling opportunities
    st.markdown("#### Cross-Selling Opportunities")
    
    # Filter strong cross-sell rules
    cross_sell_rules = rules_df[
        (rules_df['lift'] > 2) & 
        (rules_df['confidence'] > 0.4)
    ].sort_values('lift', ascending=False)
    
    if not cross_sell_rules.empty:
        # Convert rules to readable format
        cross_sell_df = cross_sell_rules.copy()
        cross_sell_df['antecedents_str'] = cross_sell_df['antecedents'].apply(lambda x: ', '.join(list(x)))
        cross_sell_df['consequents_str'] = cross_sell_df['consequents'].apply(lambda x: ', '.join(list(x)))
        cross_sell_df['rule'] = cross_sell_df.apply(
            lambda row: f"{row['antecedents_str']} → {row['consequents_str']}", axis=1
        )
        
        # Display cross-sell opportunities
        for i, (_, row) in enumerate(cross_sell_df.head(5).iterrows()):
            st.markdown(f"""
            <div class="insight-card sales-insight">
                <div class="insight-title">Cross-Sell Opportunity #{i+1}</div>
                <div class="insight-description">
                    <b>When customers purchase:</b> {row['antecedents_str']}<br>
                    <b>Recommend:</b> {row['consequents_str']}<br>
                    <b>Lift:</b> {row['lift']:.2f}x &nbsp; | &nbsp; <b>Confidence:</b> {row['confidence']:.2f} ({row['confidence']*100:.0f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        # Show more details in an expander
        with st.expander("View more cross-selling opportunities"):
            st.dataframe(
                cross_sell_df[['rule', 'lift', 'confidence', 'support']].rename(
                    columns={'rule': 'Cross-Sell Rule', 'lift': 'Lift', 'confidence': 'Confidence', 'support': 'Support'}
                ).head(20),
                use_container_width=True
            )
    else:
        st.info("No strong cross-selling opportunities found. Try adjusting the minimum support or confidence values.")
    
    # Hidden gem products
    st.markdown("#### Hidden Gem Products")
    st.markdown("""
    Hidden gems are less popular products that have strong associations with other items. 
    These represent opportunities to boost sales of underperforming inventory.
    """)
    
    if 'hidden_gems' in insights and insights['hidden_gems']:
        for i, gem in enumerate(insights['hidden_gems']):
            st.markdown(f"""
            <div class="insight-card sales-insight">
                <div class="insight-title">Hidden Gem #{i+1}: {gem['item']}</div>
                <div class="insight-description">
                    <b>This product is strongly associated with:</b> {gem['antecedents']}<br>
                    <b>Lift:</b> {gem['lift']:.2f}x &nbsp; | &nbsp; <b>Confidence:</b> {gem['confidence']:.2f} ({gem['confidence']*100:.0f}%)<br>
                    <b>Recommendation:</b> Promote this product to customers who purchase {gem['antecedents']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No hidden gem products identified. Try adjusting the rule parameters.")
    
    # Bundle suggestions
    st.markdown("#### Product Bundle Suggestions")
    
    # Find rules with multiple antecedents and high confidence
    bundle_rules = rules_df[
        (rules_df['confidence'] > 0.6) & 
        (rules_df['lift'] > 2)
    ].sort_values(['confidence', 'lift'], ascending=False)
    
    if not bundle_rules.empty:
        # Convert to readable format for display
        bundle_df = bundle_rules.copy()
        bundle_df['items'] = bundle_df.apply(
            lambda row: list(row['antecedents']) + list(row['consequents']), axis=1
        )
        bundle_df['bundle'] = bundle_df['items'].apply(lambda x: ', '.join(x))
        bundle_df['num_items'] = bundle_df['items'].apply(len)
        
        # Filter to bundles with at least 2 items
        bundle_df = bundle_df[bundle_df['num_items'] >= 2]
        
        # Display bundle suggestions
        if not bundle_df.empty:
            for i, (_, row) in enumerate(bundle_df.head(5).iterrows()):
                st.markdown(f"""
                <div class="insight-card sales-insight">
                    <div class="insight-title">Bundle Suggestion #{i+1}</div>
                    <div class="insight-description">
                        <b>Suggested Bundle:</b> {row['bundle']}<br>
                        <b>Number of Items:</b> {row['num_items']}<br>
                        <b>Confidence:</b> {row['confidence']:.2f} &nbsp; | &nbsp; <b>Lift:</b> {row['lift']:.2f}x
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            # Show more details in an expander
            with st.expander("View more bundle suggestions"):
                st.dataframe(
                    bundle_df[['bundle', 'num_items', 'confidence', 'lift']].rename(
                        columns={'bundle': 'Suggested Bundle', 'num_items': 'Item Count', 'confidence': 'Confidence', 'lift': 'Lift'}
                    ).head(20),
                    use_container_width=True
                )
        else:
            st.info("No suitable bundle suggestions found. Try adjusting the rule parameters.")
    else:
        st.info("No suitable bundle suggestions found. Try adjusting the rule parameters.")
        
with insight_tabs[4]:  # Advanced Analysis
    st.markdown("### 🔮 Advanced Analysis")
    
    # Customer segment analysis
    if 'customer_segments' in st.session_state:
        st.markdown("#### Analysis by Customer Segment")
        segments = st.session_state.customer_segments
        
        # Create segment summary
        segment_summary = {segment: len(df) for segment, df in segments.items()}
        segment_df = pd.DataFrame(list(segment_summary.items()), columns=['Segment', 'Count'])
        segment_df['Percentage'] = segment_df['Count'] / segment_df['Count'].sum() * 100
        
        # Display segment pie chart
        fig = px.pie(
            segment_df, 
            values='Count', 
            names='Segment',
            title='Customer Segments Breakdown',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True, key="insight_visualization")
        
        # Display segment-specific insights
        if 'segment_rules' in st.session_state:
            segment_rules = st.session_state.segment_rules
            
            # Select a segment to analyze
            selected_segment = st.selectbox(
                "Select a customer segment to analyze",
                list(segment_rules.keys())
            )
            
            if selected_segment and selected_segment in segment_rules:
                # Get rules for the selected segment
                seg_rules_df = segment_rules[selected_segment]
                
                if not seg_rules_df.empty:
                    # Display segment-specific metrics
                    st.markdown(f"##### Insights for {selected_segment} Segment")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rules Count", len(seg_rules_df))
                    with col2:
                        st.metric("Average Lift", f"{seg_rules_df['lift'].mean():.2f}")
                    with col3:
                        st.metric("Strong Associations", len(seg_rules_df[seg_rules_df['lift'] > 3]))
                    
                    # Show top rules for this segment
                    st.markdown(f"##### Top Rules for {selected_segment}")
                    
                    # Format rules for display
                    display_rules = seg_rules_df.sort_values('lift', ascending=False).head(10).copy()
                    display_rules['antecedents_str'] = display_rules['antecedents'].apply(lambda x: ', '.join(list(x)))
                    display_rules['consequents_str'] = display_rules['consequents'].apply(lambda x: ', '.join(list(x)))
                    display_rules['rule'] = display_rules.apply(
                        lambda row: f"{row['antecedents_str']} → {row['consequents_str']}", axis=1
                    )
                    
                    st.dataframe(
                        display_rules[['rule', 'support', 'confidence', 'lift']].rename(
                            columns={'rule': 'Rule', 'support': 'Support', 'confidence': 'Confidence', 'lift': 'Lift'}
                        ),
                        use_container_width=True
                    )
                    
                    # Visualization for segment
                    st.markdown(f"##### Visualization for {selected_segment}")
                    network_fig = create_association_network(seg_rules_df, min_lift=1.5)
                    if network_fig:
                        st.plotly_chart(network_fig, use_container_width=True, key="segment_network_viz")
                else:
                    st.info(f"No rules found for the {selected_segment} segment.")
    else:
        st.info("Customer segment analysis is not available. Enable customer segmentation when generating rules.")
    
    # Time-based analysis
    if 'time_rules' in st.session_state:
        st.markdown("#### Time-Based Analysis")
        time_rules = st.session_state.time_rules
        
        # Display time period metrics
        st.markdown("##### Association Rule Trends Over Time")
        
        # Create a selector for the metric to visualize
        metric = st.selectbox(
            "Select metric to visualize", 
            ["count", "avg_lift", "avg_confidence", "avg_support"],
            format_func=lambda x: {
                "count": "Number of Rules", 
                "avg_lift": "Average Lift", 
                "avg_confidence": "Average Confidence",
                "avg_support": "Average Support"
            }[x]
        )
        
        # Import visualization function for time trends
        from utils.visualizations import plot_rules_over_time
        
        # Plot the trend
        trend_fig = plot_rules_over_time(time_rules, metric)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True, key="time_trend_analysis")
        
        # Allow selection of a time period for detailed analysis
        selected_period = st.selectbox(
            "Select a time period for detailed analysis",
            list(time_rules.keys())
        )
        
        if selected_period in time_rules:
            # Get rules for the selected time period
            period_rules = time_rules[selected_period]
            
            if not period_rules.empty:
                # Display time period metrics
                st.markdown(f"##### Insights for {selected_period}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Rules Count", len(period_rules))
                with col2:
                    st.metric("Average Lift", f"{period_rules['lift'].mean():.2f}")
                with col3:
                    st.metric("Strong Associations", len(period_rules[period_rules['lift'] > 3]))
                
                # Show unique rules for this time period
                other_periods = [p for p in time_rules.keys() if p != selected_period]
                
                unique_rules = []
                for _, row in period_rules.iterrows():
                    is_unique = True
                    antecedents = row['antecedents']
                    consequents = row['consequents']
                    
                    for other_period in other_periods:
                        other_rules = time_rules[other_period]
                        for _, other_row in other_rules.iterrows():
                            if antecedents == other_row['antecedents'] and consequents == other_row['consequents']:
                                is_unique = False
                                break
                        if not is_unique:
                            break
                    
                    if is_unique:
                        unique_rules.append({
                            'antecedents': ', '.join(list(antecedents)),
                            'consequents': ', '.join(list(consequents)),
                            'lift': row['lift'],
                            'confidence': row['confidence']
                        })
                
                # Display unique rules
                if unique_rules:
                    st.markdown(f"##### Unique Rules for {selected_period}")
                    unique_df = pd.DataFrame(unique_rules)
                    st.dataframe(
                        unique_df.rename(
                            columns={'antecedents': 'Antecedents', 'consequents': 'Consequents', 
                                     'lift': 'Lift', 'confidence': 'Confidence'}
                        ),
                        use_container_width=True
                    )
                else:
                    st.info(f"No unique rules found for {selected_period}.")
                
                # Visualization for time period
                st.markdown(f"##### Visualization for {selected_period}")
                network_fig = create_association_network(period_rules, min_lift=1.5)
                if network_fig:
                    st.plotly_chart(network_fig, use_container_width=True, key="product_network_viz")
    else:
        st.info("Time-based analysis is not available. Enable time analysis when generating rules.")

# Performance statistics at the bottom
if 'processing_time' in st.session_state:
    with st.expander("⚡ Performance Statistics"):
        perf_times = st.session_state.processing_time
        
        col1, col2, col3 = st.columns(3)
        with col1:
            data_time = perf_times.get('data_loading', 0)
            st.metric("Data Loading Time", f"{data_time:.2f}s")
        
        with col2:
            mining_time = perf_times.get('rule_mining', 0)
            st.metric("Rule Mining Time", f"{mining_time:.2f}s")
        
        with col3:
            total_time = sum(perf_times.values())
            st.metric("Total Processing Time", f"{total_time:.2f}s") 