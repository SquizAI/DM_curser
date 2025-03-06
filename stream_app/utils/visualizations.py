import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import List, Dict, Tuple, Optional, Union
import matplotlib.cm as cm
from wordcloud import WordCloud
import polars as pl
import time

def convert_frozenset_to_str(rule_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert frozenset objects to strings for visualization
    
    Args:
        rule_df: Rules DataFrame with frozenset objects
        
    Returns:
        DataFrame with string representations
    """
    if rule_df.empty:
        return rule_df
        
    df = rule_df.copy()
    df['antecedents'] = df['antecedents'].apply(lambda x: ', '.join(list(x)))
    df['consequents'] = df['consequents'].apply(lambda x: ', '.join(list(x)))
    return df

def create_rule_scatterplot(rules: pd.DataFrame, 
                          colorby: str = 'lift', 
                          sizeby: str = 'lift') -> go.Figure:
    """
    Create an interactive scatter plot of rules
    
    Args:
        rules: Association rules DataFrame
        colorby: Metric to use for point color
        sizeby: Metric to use for point size
        
    Returns:
        Plotly Figure object
    """
    if rules.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No rules found with current parameters",
            xaxis_title="Support",
            yaxis_title="Confidence"
        )
        return fig
    
    # Convert frozensets to strings
    display_rules = convert_frozenset_to_str(rules)
    
    # Create scatter plot
    fig = px.scatter(
        display_rules, 
        x="support", 
        y="confidence", 
        size=sizeby, 
        color=colorby,
        hover_data=["antecedents", "consequents", "lift", "support", "confidence"],
        title="Support vs Confidence (colored by {})".format(colorby),
        labels={
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
            "antecedents": "Antecedents",
            "consequents": "Consequents"
        }
    )
    
    # Enhance the layout
    fig.update_layout(
        plot_bgcolor='white',
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        ),
        margin=dict(l=20, r=20, t=40, b=20),
        coloraxis_colorbar=dict(
            title=colorby.capitalize()
        )
    )
    
    return fig

def create_3d_rule_visualization(rules: pd.DataFrame) -> go.Figure:
    """
    Create a 3D visualization of rules showing support, confidence, and lift
    
    Args:
        rules: Association rules DataFrame
        
    Returns:
        Plotly Figure object
    """
    if rules.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No rules found with current parameters",
        )
        return fig
    
    display_rules = convert_frozenset_to_str(rules)
    
    fig = go.Figure(data=[go.Scatter3d(
        x=display_rules['support'],
        y=display_rules['confidence'],
        z=display_rules['lift'],
        text=display_rules['antecedents'] + ' → ' + display_rules['consequents'],
        mode='markers',
        marker=dict(
            size=display_rules['lift']*5,
            color=display_rules['lift'],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Lift")
        ),
        hoverinfo='text'
    )])
    
    fig.update_layout(
        title="3D Visualization of Association Rules",
        scene=dict(
            xaxis_title='Support',
            yaxis_title='Confidence',
            zaxis_title='Lift'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    return fig

def create_rule_network(rules: pd.DataFrame, 
                       min_lift: float = 1.5, 
                       max_rules: int = 20) -> go.Figure:
    """
    Create a network graph visualization of association rules
    
    Args:
        rules: Association rules DataFrame
        min_lift: Minimum lift for rules to include
        max_rules: Maximum number of rules to show
        
    Returns:
        Plotly Figure object
    """
    if rules.empty:
        fig = go.Figure()
        fig.update_layout(
            title="No rules found with current parameters",
        )
        return fig
    
    # Filter rules
    filtered_rules = rules[rules['lift'] >= min_lift].sort_values('lift', ascending=False).head(max_rules)
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes and edges
    for _, row in filtered_rules.iterrows():
        # Add antecedent items
        for item in row['antecedents']:
            if item not in G:
                G.add_node(item)
        
        # Add consequent items
        for item in row['consequents']:
            if item not in G:
                G.add_node(item)
            
        # Add edges from each antecedent to each consequent
        for a_item in row['antecedents']:
            for c_item in row['consequents']:
                G.add_edge(a_item, c_item, weight=row['lift'])
    
    # Use spring layout for node positions
    pos = nx.spring_layout(G, seed=42)
    
    # Extract node positions
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
    
    # Create edges
    edge_x = []
    edge_y = []
    edge_text = []
    edge_weights = []
    
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        edge_text.append(f"{edge[0]} → {edge[1]}<br>Lift: {edge[2]['weight']:.2f}")
        edge_weights.append(edge[2]['weight'])
    
    # Create edge trace
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        text=edge_text,
        mode='lines')
    
    # Create node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))
    
    # Color nodes by number of connections
    node_adjacencies = []
    for node in G.nodes():
        node_adjacencies.append(len(list(G.neighbors(node))))
    
    node_trace.marker.color = node_adjacencies
    
    # Create figure
    fig = go.Figure(data=[edge_trace, node_trace],
                  layout=go.Layout(
                      title="Network Graph of Association Rules",
                      titlefont_size=16,
                      showlegend=False,
                      hovermode='closest',
                      margin=dict(b=20,l=5,r=5,t=40),
                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                  )
    
    return fig

def create_metrics_distribution(rules: pd.DataFrame, 
                              metric: str = 'lift',
                              nbins: int = 20) -> go.Figure:
    """
    Create a histogram showing distribution of a metric
    
    Args:
        rules: Association rules DataFrame
        metric: Metric to visualize
        nbins: Number of histogram bins
        
    Returns:
        Plotly Figure object
    """
    if rules.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No rules found to analyze {metric} distribution",
        )
        return fig
    
    # Create histogram
    fig = px.histogram(
        rules, 
        x=metric, 
        nbins=nbins,
        title=f"Distribution of {metric.capitalize()} Values",
        labels={metric: metric.capitalize()},
        color_discrete_sequence=['#636EFA']
    )
    
    # Add a vertical line for mean
    mean_value = rules[metric].mean()
    fig.add_vline(x=mean_value, line_dash="dash", line_color="red",
                annotation_text=f"Mean: {mean_value:.2f}", 
                annotation_position="top right")
    
    # Enhance layout
    fig.update_layout(
        plot_bgcolor='white',
        bargap=0.1,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def create_top_rules_table(rules: pd.DataFrame, 
                          sort_by: str = 'lift', 
                          ascending: bool = False,
                          top_n: int = 10) -> pd.DataFrame:
    """
    Create a formatted table of top rules
    
    Args:
        rules: Association rules DataFrame
        sort_by: Metric to sort by
        ascending: Sort direction
        top_n: Number of rules to include
        
    Returns:
        Formatted DataFrame
    """
    if rules.empty:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 'confidence', 'lift'])
    
    # Convert frozensets to strings
    display_rules = convert_frozenset_to_str(rules)
    
    # Sort and select top N
    top_rules = display_rules.sort_values(sort_by, ascending=ascending).head(top_n)
    
    # Round metrics to improve readability
    for col in ['support', 'confidence', 'lift']:
        if col in top_rules.columns:
            top_rules[col] = top_rules[col].round(3)
    
    return top_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

def create_temporal_analysis_chart(time_rules: Dict[str, pd.DataFrame]) -> go.Figure:
    """
    Create a chart showing rule metrics over time
    
    Args:
        time_rules: Dictionary of time period -> rules DataFrame
        
    Returns:
        Plotly Figure object
    """
    if not time_rules:
        fig = go.Figure()
        fig.update_layout(
            title="No temporal data available",
        )
        return fig
    
    # Extract metrics over time
    time_periods = []
    rule_counts = []
    avg_supports = []
    avg_confidences = []
    avg_lifts = []
    
    for period, rules in sorted(time_rules.items()):
        time_periods.append(period)
        rule_counts.append(len(rules))
        avg_supports.append(rules['support'].mean())
        avg_confidences.append(rules['confidence'].mean())
        avg_lifts.append(rules['lift'].mean())
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Bar(x=time_periods, y=rule_counts, name="Rule Count", marker_color='#636EFA')
    )
    
    fig.add_trace(
        go.Scatter(x=time_periods, y=avg_supports, name="Avg Support", 
                  mode='lines+markers', marker_color='red', yaxis="y2")
    )
    
    fig.add_trace(
        go.Scatter(x=time_periods, y=avg_confidences, name="Avg Confidence", 
                  mode='lines+markers', marker_color='green', yaxis="y2")
    )
    
    fig.add_trace(
        go.Scatter(x=time_periods, y=avg_lifts, name="Avg Lift", 
                  mode='lines+markers', marker_color='orange', yaxis="y2")
    )
    
    # Create layout with secondary y-axis
    fig.update_layout(
        title="Association Rule Metrics Over Time",
        xaxis=dict(title="Time Period"),
        yaxis=dict(title="Rule Count", side="left"),
        yaxis2=dict(title="Average Metric Value", side="right", overlaying="y", rangemode="tozero"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor='white'
    )
    
    return fig 

@st.cache_data
def create_item_frequency_chart(transaction_df: pd.DataFrame, top_n: int = 20):
    """
    Create an interactive bar chart showing the most frequently purchased items.
    
    Args:
        transaction_df: DataFrame containing transaction data with 'Description' column
        top_n: Number of top items to display
        
    Returns:
        Plotly figure object
    """
    start_time = time.time()
    
    # Use Polars for faster processing
    df_pl = pl.from_pandas(transaction_df)
    
    # Get counts of each product
    item_counts = df_pl.group_by('Description').agg(
        pl.count().alias('Count')
    ).sort('Count', descending=True).head(top_n)
    
    # Convert back to pandas for plotting
    item_counts_pd = item_counts.to_pandas()
    
    # Create the plot
    fig = px.bar(
        item_counts_pd,
        x='Count',
        y='Description',
        orientation='h',
        title=f'Top {top_n} Most Frequently Purchased Products',
        labels={'Count': 'Purchase Frequency', 'Description': 'Product'},
        color='Count',
        color_continuous_scale='Viridis',
    )
    
    # Update layout
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        height=600,
        xaxis_title="Purchase Frequency",
        yaxis_title="Product",
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    # Add hover data
    fig.update_traces(
        hovertemplate='<b>%{y}</b><br>Frequency: %{x:,}<extra></extra>'
    )
    
    print(f"Item frequency chart created in {time.time() - start_time:.2f} seconds")
    return fig

def create_metric_distribution_plots(rules_df: pd.DataFrame) -> Dict:
    """
    Create histograms of confidence, support, and lift distributions.
    
    Args:
        rules_df: DataFrame containing association rules
        
    Returns:
        Dictionary of Plotly figures
    """
    if rules_df.empty:
        # Return empty figures
        empty_fig = px.histogram(title="No rules to display")
        return {
            'confidence': empty_fig,
            'support': empty_fig,
            'lift': empty_fig
        }
    
    # Create histograms
    confidence_fig = px.histogram(
        rules_df, x='confidence',
        title='Distribution of Confidence Values',
        labels={'confidence': 'Confidence'},
        color_discrete_sequence=['#1E88E5'],
        nbins=20
    )
    
    support_fig = px.histogram(
        rules_df, x='support',
        title='Distribution of Support Values',
        labels={'support': 'Support'},
        color_discrete_sequence=['#FFC107'],
        nbins=20
    )
    
    lift_fig = px.histogram(
        rules_df, x='lift',
        title='Distribution of Lift Values',
        labels={'lift': 'Lift'},
        color_discrete_sequence=['#4CAF50'],
        nbins=20
    )
    
    # Customize layouts
    for fig in [confidence_fig, support_fig, lift_fig]:
        fig.update_layout(
            bargap=0.1,
            height=400,
            width=700
        )
    
    return {
        'confidence': confidence_fig,
        'support': support_fig,
        'lift': lift_fig
    }

def visualize_rules_over_time(rules_by_time: Dict[str, pd.DataFrame], 
                              metric: str = 'count',
                              time_unit: str = 'month') -> go.Figure:
    """
    Visualize how rules change over time.
    
    Args:
        rules_by_time: Dictionary mapping time periods to rule DataFrames
        metric: Metric to visualize ('count', 'avg_confidence', 'avg_lift', 'avg_support')
        time_unit: Time unit for x-axis ('day', 'week', 'month', 'quarter')
        
    Returns:
        Plotly figure
    """
    if not rules_by_time:
        return px.line(title="No time-based data available")
    
    # Prepare data for visualization
    periods = []
    metrics = []
    
    for period, rules in rules_by_time.items():
        periods.append(period)
        
        if metric == 'count':
            metrics.append(len(rules))
        elif metric == 'avg_confidence':
            metrics.append(rules['confidence'].mean() if not rules.empty else 0)
        elif metric == 'avg_lift':
            metrics.append(rules['lift'].mean() if not rules.empty else 0)
        elif metric == 'avg_support':
            metrics.append(rules['support'].mean() if not rules.empty else 0)
    
    # Create DataFrame for visualization
    time_data = pd.DataFrame({
        'Period': periods,
        'Value': metrics
    })
    
    # Create line chart
    metric_labels = {
        'count': 'Number of Rules',
        'avg_confidence': 'Average Confidence',
        'avg_lift': 'Average Lift',
        'avg_support': 'Average Support'
    }
    
    fig = px.line(
        time_data, 
        x='Period', 
        y='Value',
        title=f'{metric_labels[metric]} by {time_unit.capitalize()}',
        labels={'Period': time_unit.capitalize(), 'Value': metric_labels[metric]},
        markers=True
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title=time_unit.capitalize(),
        yaxis_title=metric_labels[metric],
        height=500,
        width=800
    )
    
    return fig 

@st.cache_data
def create_association_network(rules_df: pd.DataFrame, min_lift: float = 1.5, max_nodes: int = 50) -> go.Figure:
    """
    Create an interactive network visualization of association rules.
    
    Args:
        rules_df: DataFrame containing association rules
        min_lift: Minimum lift value for rules to include in visualization
        max_nodes: Maximum number of nodes to show to avoid overcrowding
        
    Returns:
        Plotly figure with network graph
    """
    if rules_df.empty:
        st.warning("No rules to visualize")
        return None
    
    # Filter rules by lift value
    filtered_rules = rules_df[rules_df['lift'] >= min_lift].sort_values('lift', ascending=False)
    
    # Limit rules if there are too many
    if len(filtered_rules) > 100:
        filtered_rules = filtered_rules.head(100)
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add edges for each rule
    all_items = set()
    for _, row in filtered_rules.iterrows():
        for item_a in row['antecedents']:
            for item_c in row['consequents']:
                if len(all_items) < max_nodes or item_a in all_items or item_c in all_items:
                    G.add_edge(item_a, item_c, weight=row['lift'], confidence=row['confidence'])
                    all_items.add(item_a)
                    all_items.add(item_c)
    
    # Limit nodes if there are too many
    if len(G.nodes) > max_nodes:
        # Keep top nodes by degree centrality
        centrality = nx.degree_centrality(G)
        top_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node[0] for node in top_nodes]
        G = G.subgraph(top_node_names)
    
    # Use a spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Extract node positions
    node_x = []
    node_y = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    
    # Calculate node sizes based on degree
    node_degrees = dict(G.degree())
    node_sizes = [node_degrees[node] * 10 + 10 for node in G.nodes()]
    
    # Node trace
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_sizes,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line=dict(width=2)
        ),
        text=[node for node in G.nodes()],
        textposition="top center",
        textfont=dict(size=10, color='black')
    )
    
    # Color nodes by their degree
    node_adjacencies = []
    node_hover_texts = []
    
    for node in G.nodes():
        adjacencies = len(list(G.neighbors(node)))
        node_adjacencies.append(adjacencies)
        
        connected_nodes = list(G.neighbors(node))
        hover_text = f"<b>{node}</b><br># of connections: {adjacencies}"
        if adjacencies > 0:
            hover_text += "<br><br>Connected to:<br>"
            for conn in connected_nodes[:10]:  # Limit to 10 connections in hover
                hover_text += f"- {conn}<br>"
        
        node_hover_texts.append(hover_text)
    
    node_trace.marker.color = node_adjacencies
    node_trace.hovertext = node_hover_texts
    
    # Create edge traces
    edge_traces = []
    
    # Set up colors based on lift
    lift_values = [G.edges[edge]['weight'] for edge in G.edges()]
    min_lift_val = min(lift_values) if lift_values else 1
    max_lift_val = max(lift_values) if lift_values else 5
    
    # Create a colormap
    colormap = cm.get_cmap('plasma')
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        
        lift = G.edges[edge]['weight']
        confidence = G.edges[edge].get('confidence', 0)
        
        # Normalize lift for color intensity (0 to 1 range)
        norm_lift = (lift - min_lift_val) / (max_lift_val - min_lift_val) if max_lift_val > min_lift_val else 0.5
        
        # Get RGB values from colormap
        rgba = colormap(norm_lift)
        rgb = f'rgb({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)})'
        
        # Adjust width based on confidence
        width = 1 + confidence * 3
        
        edge_trace = go.Scatter(
            x=[x0, x1, None], 
            y=[y0, y1, None],
            line=dict(width=width, color=rgb),
            hoverinfo='text',
            mode='lines',
            text=f"{edge[0]} → {edge[1]}<br>Lift: {lift:.2f}<br>Confidence: {confidence:.2f}",
            opacity=min(0.8, max(0.3, norm_lift))
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(
        data=edge_traces + [node_trace],
        layout=go.Layout(
            title='Association Rules Network',
            titlefont=dict(size=16),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Network of associated products<br>Node size: Importance of item<br>Edge color: Rule strength",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='rgba(248,248,248,0.9)',
            paper_bgcolor='rgba(248,248,248,0.9)',
        )
    )
    
    return fig

@st.cache_data
def create_rule_strength_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of rules based on support, confidence and lift.
    
    Args:
        rules_df: DataFrame containing association rules
        
    Returns:
        Plotly scatter plot figure
    """
    if rules_df.empty:
        st.warning("No rules to visualize")
        return None
    
    # Convert antecedents and consequents to strings for better visualization
    rules_df = rules_df.copy()
    rules_df['antecedent_str'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    rules_df['consequent_str'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Create rule text for display
    rules_df['rule'] = rules_df.apply(
        lambda row: f"{row['antecedent_str']} → {row['consequent_str']}", axis=1
    )
    
    # Create the scatter plot
    fig = px.scatter(
        rules_df,
        x='support',
        y='confidence',
        size='lift',
        color='lift',
        hover_name='rule',
        size_max=50,
        color_continuous_scale='Viridis',
        title='Rule Strength Visualization',
        labels={
            'support': 'Support (frequency of occurrence)',
            'confidence': 'Confidence (reliability)',
            'lift': 'Lift (strength of association)'
        }
    )
    
    # Add hover data
    fig.update_traces(
        hovertemplate='<b>%{hovertext}</b><br>Support: %{x:.4f}<br>Confidence: %{y:.4f}<br>Lift: %{marker.color:.2f}<extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        coloraxis_colorbar=dict(
            title="Lift",
            thicknessmode="pixels", thickness=15,
            lenmode="pixels", len=300,
            yanchor="top", y=1,
            ticks="outside", tickprefix="",
            ticksuffix=""
        ),
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

@st.cache_data
def create_lift_histogram(rules_df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram showing the distribution of lift values among rules.
    
    Args:
        rules_df: DataFrame containing association rules
        
    Returns:
        Plotly histogram figure
    """
    if rules_df.empty:
        st.warning("No rules to visualize")
        return None
    
    # Create histogram
    fig = px.histogram(
        rules_df, 
        x='lift',
        nbins=30,
        title='Distribution of Lift Values',
        labels={'lift': 'Lift Value', 'count': 'Number of Rules'},
        color_discrete_sequence=['#1E88E5'],
    )
    
    # Add a vertical line at lift = 1
    fig.add_vline(
        x=1, 
        line_dash="dash", 
        line_color="red",
        annotation_text="Lift = 1 (No Association)",
        annotation_position="top right"
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        bargap=0.1,
        xaxis_title="Lift Value",
        yaxis_title="Number of Rules",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

def create_wordcloud_from_rules(rules_df: pd.DataFrame, type_filter: str = 'all') -> plt.Figure:
    """
    Create a word cloud visualization from association rules.
    
    Args:
        rules_df: DataFrame containing association rules
        type_filter: Type of items to include: 'all', 'antecedents', or 'consequents'
        
    Returns:
        Matplotlib figure with wordcloud
    """
    if rules_df.empty:
        st.warning("No rules to visualize")
        return None
    
    # Create a dictionary to count word frequencies
    word_counts = {}
    
    # Extract words from rules based on filter
    for _, row in rules_df.iterrows():
        # Get words from antecedents
        if type_filter in ['all', 'antecedents']:
            for item in row['antecedents']:
                if item in word_counts:
                    word_counts[item] += row['lift']  # Weight by lift
                else:
                    word_counts[item] = row['lift']
        
        # Get words from consequents
        if type_filter in ['all', 'consequents']:
            for item in row['consequents']:
                if item in word_counts:
                    word_counts[item] += row['lift'] * 1.5  # Weight consequents a bit more
                else:
                    word_counts[item] = row['lift'] * 1.5
    
    # Create wordcloud
    wordcloud = WordCloud(
        width=800, 
        height=400,
        background_color='white',
        colormap='viridis',
        max_words=100,
        contour_width=1,
        contour_color='steelblue',
        collocations=False
    ).generate_from_frequencies(word_counts)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    
    # Set title based on filter
    if type_filter == 'all':
        title = 'Word Cloud of All Items in Rules'
    elif type_filter == 'antecedents':
        title = 'Word Cloud of Items in Antecedents (If part)'
    else:
        title = 'Word Cloud of Items in Consequents (Then part)'
    
    plt.title(title, fontsize=14)
    plt.tight_layout()
    
    return fig

@st.cache_data
def plot_rules_over_time(time_rules: Dict[str, pd.DataFrame], metric: str = 'count') -> go.Figure:
    """
    Create a line chart showing how rules change over time periods.
    
    Args:
        time_rules: Dictionary with time periods as keys and rule DataFrames as values
        metric: Metric to visualize ('count', 'avg_lift', 'avg_confidence', 'avg_support')
        
    Returns:
        Plotly line chart figure
    """
    # Extract metrics for each time period
    time_periods = []
    metrics = []
    
    for period, rules in time_rules.items():
        time_periods.append(period)
        
        if metric == 'count':
            metrics.append(len(rules))
        elif metric == 'avg_lift':
            metrics.append(rules['lift'].mean() if not rules.empty else 0)
        elif metric == 'avg_confidence':
            metrics.append(rules['confidence'].mean() if not rules.empty else 0)
        elif metric == 'avg_support':
            metrics.append(rules['support'].mean() if not rules.empty else 0)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Time Period': time_periods,
        'Value': metrics
    })
    
    # Sort by time period
    plot_df['Time Period'] = pd.Categorical(plot_df['Time Period'], categories=time_periods, ordered=True)
    plot_df = plot_df.sort_values('Time Period')
    
    # Create metric labels
    metric_labels = {
        'count': 'Number of Rules',
        'avg_lift': 'Average Lift',
        'avg_confidence': 'Average Confidence',
        'avg_support': 'Average Support'
    }
    
    # Create the line chart
    fig = px.line(
        plot_df, 
        x='Time Period', 
        y='Value',
        markers=True,
        title=f'{metric_labels[metric]} Over Time',
        labels={'Value': metric_labels[metric], 'Time Period': 'Time Period'},
        color_discrete_sequence=['#1E88E5']
    )
    
    # Add data points with values
    fig.update_traces(
        textposition="top center",
        texttemplate='%{y:.2f}' if metric != 'count' else '%{y:,.0f}',
        hovertemplate='<b>%{x}</b><br>%{y:,.2f}' if metric != 'count' else '<b>%{x}</b><br>%{y:,.0f}<extra></extra>'
    )
    
    # Update layout
    fig.update_layout(
        height=400,
        xaxis_title="Time Period",
        yaxis_title=metric_labels[metric],
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

@st.cache_data
def create_segment_comparison_chart(segment_rules: Dict[str, pd.DataFrame], metric: str = 'count') -> go.Figure:
    """
    Create a bar chart comparing rule metrics across different customer segments.
    
    Args:
        segment_rules: Dictionary with segment names as keys and rule DataFrames as values
        metric: Metric to visualize ('count', 'avg_lift', 'avg_confidence', 'avg_support')
        
    Returns:
        Plotly bar chart figure
    """
    # Extract metrics for each segment
    segments = []
    metrics = []
    
    for segment, rules in segment_rules.items():
        segments.append(segment)
        
        if metric == 'count':
            metrics.append(len(rules))
        elif metric == 'avg_lift':
            metrics.append(rules['lift'].mean() if not rules.empty else 0)
        elif metric == 'avg_confidence':
            metrics.append(rules['confidence'].mean() if not rules.empty else 0)
        elif metric == 'avg_support':
            metrics.append(rules['support'].mean() if not rules.empty else 0)
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Customer Segment': segments,
        'Value': metrics
    })
    
    # Sort by metric value for better visualization
    plot_df = plot_df.sort_values('Value', ascending=False)
    
    # Create metric labels
    metric_labels = {
        'count': 'Number of Rules',
        'avg_lift': 'Average Lift',
        'avg_confidence': 'Average Confidence',
        'avg_support': 'Average Support'
    }
    
    # Create the bar chart
    fig = px.bar(
        plot_df, 
        x='Customer Segment', 
        y='Value',
        title=f'{metric_labels[metric]} by Customer Segment',
        labels={'Value': metric_labels[metric], 'Customer Segment': 'Customer Segment'},
        color='Value',
        color_continuous_scale='Viridis',
        text_auto='.2f' if metric != 'count' else True
    )
    
    # Update layout
    fig.update_layout(
        height=500,
        xaxis_title="Customer Segment",
        yaxis_title=metric_labels[metric],
        coloraxis_showscale=False,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

@st.cache_data
def create_rule_browser(rules_df: pd.DataFrame, search_term: Optional[str] = None) -> pd.DataFrame:
    """
    Create a searchable, sortable table of rules.
    
    Args:
        rules_df: DataFrame containing association rules
        search_term: Optional term to filter rules by
        
    Returns:
        Formatted DataFrame for display
    """
    if rules_df.empty:
        return pd.DataFrame()
    
    # Create a copy of the rules DataFrame
    display_df = rules_df.copy()
    
    # Convert sets to strings for display
    display_df['antecedents_str'] = display_df['antecedents'].apply(lambda x: ', '.join(list(x)))
    display_df['consequents_str'] = display_df['consequents'].apply(lambda x: ', '.join(list(x)))
    
    # Create rule text columns
    display_df['rule'] = display_df.apply(
        lambda row: f"{row['antecedents_str']} → {row['consequents_str']}", axis=1
    )
    
    # Filter by search term if provided
    if search_term and search_term.strip():
        term = search_term.lower().strip()
        mask = (
            display_df['antecedents_str'].str.lower().str.contains(term) | 
            display_df['consequents_str'].str.lower().str.contains(term)
        )
        display_df = display_df[mask]
    
    # Format metrics for display
    display_df['support'] = display_df['support'].apply(lambda x: f"{x:.4f}")
    display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.4f}")
    display_df['lift'] = display_df['lift'].apply(lambda x: f"{x:.2f}")
    
    # Select and rename columns for display
    result_df = display_df[['rule', 'support', 'confidence', 'lift']].rename(
        columns={
            'rule': 'Association Rule',
            'support': 'Support', 
            'confidence': 'Confidence',
            'lift': 'Lift'
        }
    )
    
    return result_df

def create_heatmap_item_combinations(transaction_df: pd.DataFrame, rules_df: pd.DataFrame, 
                                     top_n: int = 10) -> go.Figure:
    """
    Create a heatmap showing the most frequent item combinations.
    
    Args:
        transaction_df: DataFrame containing transaction data
        rules_df: DataFrame containing association rules
        top_n: Number of top items to include in the heatmap
        
    Returns:
        Plotly heatmap figure
    """
    # Extract top items from rules
    all_items = set()
    for _, row in rules_df.iterrows():
        all_items.update(row['antecedents'])
        all_items.update(row['consequents'])
    
    # Get top items by frequency
    item_counts = transaction_df['Description'].value_counts().head(top_n)
    top_items = item_counts.index.tolist()
    
    # Create a co-occurrence matrix
    cooccurrence = np.zeros((len(top_items), len(top_items)))
    
    # Group transactions by invoice
    invoice_groups = transaction_df.groupby('InvoiceNo')
    
    # Fill the co-occurrence matrix
    for _, group in invoice_groups:
        items_in_invoice = set(group['Description'].unique())
        
        # Check each pair of top items
        for i, item1 in enumerate(top_items):
            for j, item2 in enumerate(top_items):
                if item1 in items_in_invoice and item2 in items_in_invoice:
                    cooccurrence[i, j] += 1
    
    # Create heatmap
    fig = px.imshow(
        cooccurrence,
        x=top_items,
        y=top_items,
        labels=dict(x="Product", y="Product", color="Co-occurrence"),
        title=f"Co-occurrence Heatmap of Top {top_n} Products",
        color_continuous_scale="Viridis"
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45,
        margin=dict(l=10, r=10, t=50, b=100),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig

@st.cache_data
def create_sunburst_chart(rules_df: pd.DataFrame, min_lift: float = 2.0) -> go.Figure:
    """
    Create a sunburst chart to visualize hierarchical rule relationships.
    
    Args:
        rules_df: DataFrame containing association rules
        min_lift: Minimum lift value for rules to include
        
    Returns:
        Plotly sunburst figure
    """
    if rules_df.empty:
        st.warning("No rules to visualize")
        return None
    
    # Filter rules by lift
    filtered_rules = rules_df[rules_df['lift'] >= min_lift].sort_values('lift', ascending=False)
    
    # Limit to top rules if there are too many
    if len(filtered_rules) > 50:
        filtered_rules = filtered_rules.head(50)
    
    # Prepare data for sunburst chart
    labels = ['All Products']  # Start with root node
    parents = ['']  # Root has no parent
    values = [100]  # Arbitrary value for root
    colors = [0]  # Color value for root
    
    # Process each rule
    for _, row in filtered_rules.iterrows():
        # Add antecedents
        for item in row['antecedents']:
            if item not in labels:
                labels.append(item)
                parents.append('All Products')  # All items are children of root
                values.append(row['support'] * 1000)  # Scale support for better visualization
                colors.append(row['lift'])  # Color by lift
        
        # Add consequents and their connections
        for item in row['consequents']:
            if item not in labels:
                labels.append(item)
                parents.append('All Products')
                values.append(row['support'] * 1000)
                colors.append(row['lift'])
            
            # Add connections from antecedents to consequents
            for ant in row['antecedents']:
                connection_name = f"{ant} → {item}"
                if connection_name not in labels:
                    labels.append(connection_name)
                    parents.append(ant)  # Connection is child of antecedent
                    values.append(row['confidence'] * 1000)  # Scale confidence for visualization
                    colors.append(row['lift'])  # Color by lift
    
    # Create sunburst chart
    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(
            colors=colors,
            colorscale='Viridis',
            cmin=min_lift,
            cmax=max(filtered_rules['lift']) if not filtered_rules.empty else min_lift + 5,
            colorbar=dict(
                title="Lift",
                thickness=15,
                xpad=0
            ),
        ),
        hovertemplate='<b>%{label}</b><br>Value: %{value:.2f}<extra></extra>',
        maxdepth=2  # Limit depth for clarity
    ))
    
    # Update layout
    fig.update_layout(
        title="Product Relationships Hierarchy",
        height=700,
        margin=dict(t=50, l=10, r=10, b=10),
        font=dict(family="Arial, sans-serif", size=12)
    )
    
    return fig 