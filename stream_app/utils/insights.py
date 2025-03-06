import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Union
import re
import streamlit as st
from collections import Counter

def extract_actionable_insights(rules: pd.DataFrame, 
                               min_lift: float = 2.0, 
                               min_confidence: float = 0.5) -> List[str]:
    """
    Extract actionable business insights from association rules
    
    Args:
        rules: DataFrame of association rules
        min_lift: Minimum lift value for significant rules
        min_confidence: Minimum confidence for reliable rules
        
    Returns:
        List of insight strings
    """
    if rules.empty:
        return ["No rules available to extract insights."]
    
    insights = []
    
    # Find strong complementary products
    strong_rules = rules[(rules['lift'] > min_lift) & (rules['confidence'] > min_confidence)]
    if not strong_rules.empty:
        top_rules = strong_rules.sort_values('lift', ascending=False).head(5)
        for _, rule in top_rules.iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insights.append(
                f"Strong complementary relationship: Customers who purchase {antecedents} "
                f"are {rule['lift']:.2f}x more likely to also purchase {consequents} "
                f"(confidence: {rule['confidence']:.2f})."
            )
    
    # Find potential cross-sell opportunities
    cross_sell = rules[(rules['lift'] > 1.5) & (rules['confidence'] > 0.4) & 
                       (rules['support'] > rules['support'].mean())]
    if not cross_sell.empty:
        for _, rule in cross_sell.sort_values('confidence', ascending=False).head(3).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insights.append(
                f"Cross-sell opportunity: Consider bundling {antecedents} with {consequents} "
                f"or suggesting {consequents} to customers who buy {antecedents}."
            )
    
    # Find hidden gems (high lift but low support)
    hidden_gems = rules[(rules['lift'] > min_lift * 1.5) & 
                        (rules['support'] < rules['support'].quantile(0.25))]
    if not hidden_gems.empty:
        for _, rule in hidden_gems.sort_values('lift', ascending=False).head(3).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insights.append(
                f"Hidden gem: While not common, the combination of {antecedents} and {consequents} "
                f"has an unusually strong association (lift: {rule['lift']:.2f}). This may represent "
                f"an untapped market segment."
            )
    
    # Find product cannibalization (negative associations)
    if 'conviction' in rules.columns:
        potential_cannibalization = rules[rules['conviction'] < 0.8]
        if not potential_cannibalization.empty:
            for _, rule in potential_cannibalization.head(3).iterrows():
                antecedents = ', '.join(list(rule['antecedents']))
                consequents = ', '.join(list(rule['consequents']))
                insights.append(
                    f"Potential cannibalization: Customers who purchase {antecedents} are less "
                    f"likely to purchase {consequents}, suggesting these products may be substitutes."
                )
    
    # Find frequent itemsets that could be bundled
    high_confidence = rules[rules['confidence'] > 0.7].sort_values('support', ascending=False)
    if not high_confidence.empty:
        for _, rule in high_confidence.head(3).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insights.append(
                f"Bundling opportunity: {antecedents} and {consequents} are purchased together "
                f"with high confidence ({rule['confidence']:.2f}). Consider creating a bundle."
            )
    
    return insights

def analyze_category_relationships(rules: pd.DataFrame,
                                  item_categories: Dict[str, str]) -> Dict[str, List[Tuple[str, float]]]:
    """
    Analyze relationships between product categories
    
    Args:
        rules: DataFrame of association rules
        item_categories: Mapping of items to their categories
        
    Returns:
        Dictionary of category relationships
    """
    if rules.empty or not item_categories:
        return {}
    
    category_relationships = {}
    
    # Map items to categories
    def get_categories(items_set):
        return {item_categories.get(item, "Unknown") for item in items_set}
    
    # Analyze category pairs
    for _, rule in rules.iterrows():
        antecedent_categories = get_categories(rule['antecedents'])
        consequent_categories = get_categories(rule['consequents'])
        
        # Create pairs of categories
        for ant_cat in antecedent_categories:
            if ant_cat == "Unknown":
                continue
                
            if ant_cat not in category_relationships:
                category_relationships[ant_cat] = []
                
            for cons_cat in consequent_categories:
                if cons_cat == "Unknown" or cons_cat == ant_cat:
                    continue
                
                # Add relationship with lift
                category_relationships[ant_cat].append((cons_cat, rule['lift']))
    
    # Aggregate and average lift values by category pairs
    for category, relationships in category_relationships.items():
        if relationships:
            cat_lift_map = {}
            for rel_cat, lift in relationships:
                if rel_cat in cat_lift_map:
                    cat_lift_map[rel_cat].append(lift)
                else:
                    cat_lift_map[rel_cat] = [lift]
            
            # Replace with average lift
            category_relationships[category] = [(cat, np.mean(lifts)) for cat, lifts in cat_lift_map.items()]
            # Sort by average lift
            category_relationships[category] = sorted(category_relationships[category], 
                                                    key=lambda x: x[1], reverse=True)
    
    return category_relationships

def identify_frequent_items(rules: pd.DataFrame) -> List[Tuple[frozenset, int]]:
    """
    Identify most frequent items across all rules
    
    Args:
        rules: DataFrame of association rules
        
    Returns:
        List of tuples (item, frequency)
    """
    if rules.empty:
        return []
    
    # Collect all items from both antecedents and consequents
    all_items = []
    for _, rule in rules.iterrows():
        all_items.extend(list(rule['antecedents']))
        all_items.extend(list(rule['consequents']))
    
    # Count frequencies
    item_counts = Counter(all_items)
    
    # Sort by frequency
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_items

def compare_time_periods(current_rules: pd.DataFrame, 
                        previous_rules: pd.DataFrame) -> List[str]:
    """
    Compare rules between two time periods to identify trends
    
    Args:
        current_rules: Current period rules
        previous_rules: Previous period rules
        
    Returns:
        List of insight strings about trends
    """
    if current_rules.empty or previous_rules.empty:
        return ["Insufficient data to compare time periods."]
    
    insights = []
    
    # Compare number of rules
    curr_count = len(current_rules)
    prev_count = len(previous_rules)
    pct_change = ((curr_count - prev_count) / prev_count) * 100 if prev_count > 0 else float('inf')
    
    if abs(pct_change) > 10:
        insights.append(
            f"The number of significant associations {'increased' if pct_change > 0 else 'decreased'} "
            f"by {abs(pct_change):.1f}% compared to the previous period "
            f"({curr_count} vs {prev_count})."
        )
    
    # Compare average metrics
    metrics = ['support', 'confidence', 'lift']
    for metric in metrics:
        if metric in current_rules.columns and metric in previous_rules.columns:
            curr_avg = current_rules[metric].mean()
            prev_avg = previous_rules[metric].mean()
            pct_change = ((curr_avg - prev_avg) / prev_avg) * 100 if prev_avg > 0 else float('inf')
            
            if abs(pct_change) > 5:
                insights.append(
                    f"Average {metric} {'increased' if pct_change > 0 else 'decreased'} "
                    f"by {abs(pct_change):.1f}% compared to the previous period "
                    f"({curr_avg:.3f} vs {prev_avg:.3f})."
                )
    
    # Find new rules
    current_rule_pairs = {
        (frozenset(row['antecedents']), frozenset(row['consequents']))
        for _, row in current_rules.iterrows()
    }
    
    previous_rule_pairs = {
        (frozenset(row['antecedents']), frozenset(row['consequents']))
        for _, row in previous_rules.iterrows()
    }
    
    new_rules = current_rule_pairs - previous_rule_pairs
    if new_rules:
        # Get top new rules by lift
        top_new_rules = []
        for ant, cons in new_rules:
            matching_rows = current_rules[
                (current_rules['antecedents'].apply(frozenset) == ant) & 
                (current_rules['consequents'].apply(frozenset) == cons)
            ]
            if not matching_rows.empty:
                row = matching_rows.iloc[0]
                top_new_rules.append((
                    ', '.join(list(ant)), 
                    ', '.join(list(cons)), 
                    row['lift']
                ))
        
        top_new_rules.sort(key=lambda x: x[2], reverse=True)
        for ant, cons, lift in top_new_rules[:3]:
            insights.append(
                f"New pattern emerged: {ant} → {cons} with lift {lift:.2f}."
            )
    
    # Find disappeared rules
    disappeared_rules = previous_rule_pairs - current_rule_pairs
    if disappeared_rules:
        insights.append(
            f"{len(disappeared_rules)} association patterns that were present in the previous "
            f"period are no longer significant."
        )
    
    return insights

def segment_based_insights(rules_by_segment: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """
    Generate insights for different customer segments
    
    Args:
        rules_by_segment: Dictionary mapping segment names to rule DataFrames
        
    Returns:
        Dictionary of segment-specific insights
    """
    segment_insights = {}
    
    if not rules_by_segment:
        return {"error": ["No segment data available."]}
    
    # For each segment, extract top insights
    for segment, rules in rules_by_segment.items():
        if rules.empty:
            segment_insights[segment] = [f"No significant patterns found for {segment} segment."]
            continue
        
        insights = []
        
        # Get average lift and compare to overall
        avg_lift = rules['lift'].mean()
        insights.append(f"Average lift for {segment} segment is {avg_lift:.2f}.")
        
        # Top rules by lift
        top_rules = rules.sort_values('lift', ascending=False).head(3)
        for _, rule in top_rules.iterrows():
            ant = ', '.join(list(rule['antecedents']))
            cons = ', '.join(list(rule['consequents']))
            insights.append(
                f"Strong association: {ant} → {cons} (lift: {rule['lift']:.2f}, "
                f"confidence: {rule['confidence']:.2f})."
            )
        
        # Unique high-confidence rules for this segment
        high_conf = rules[rules['confidence'] > 0.6].sort_values('confidence', ascending=False).head(2)
        for _, rule in high_conf.iterrows():
            ant = ', '.join(list(rule['antecedents']))
            cons = ', '.join(list(rule['consequents']))
            insights.append(
                f"Reliable pattern: {ant} → {cons} occurs with {rule['confidence']:.2f} confidence."
            )
        
        segment_insights[segment] = insights
    
    return segment_insights

def create_recommendation_strategy(rules: pd.DataFrame, 
                                 target_item: str = None) -> Dict[str, List[Tuple[str, float]]]:
    """
    Create product recommendation strategy based on rules
    
    Args:
        rules: DataFrame of association rules
        target_item: Optional specific item to build recommendations for
        
    Returns:
        Dictionary of recommendations
    """
    if rules.empty:
        return {}
    
    recommendations = {}
    
    # If target item is specified, find specific recommendations
    if target_item:
        # Find rules where target_item is in antecedents
        target_rules = rules[rules['antecedents'].apply(lambda x: target_item in x)]
        
        if not target_rules.empty:
            recommendations[target_item] = []
            for _, rule in target_rules.sort_values('lift', ascending=False).iterrows():
                for item in rule['consequents']:
                    recommendations[target_item].append((item, rule['lift']))
            
            # Remove duplicates keeping highest lift
            item_to_lift = {}
            for item, lift in recommendations[target_item]:
                if item not in item_to_lift or lift > item_to_lift[item]:
                    item_to_lift[item] = lift
            
            recommendations[target_item] = [(item, lift) for item, lift in item_to_lift.items()]
            recommendations[target_item].sort(key=lambda x: x[1], reverse=True)
    else:
        # Build recommendation map for all items
        all_items = set()
        for _, rule in rules.iterrows():
            all_items.update(rule['antecedents'])
        
        for item in all_items:
            # Find rules where this item is in antecedents
            item_rules = rules[rules['antecedents'].apply(lambda x: item in x)]
            
            if not item_rules.empty:
                recommendations[item] = []
                for _, rule in item_rules.sort_values('lift', ascending=False).head(5).iterrows():
                    for rec_item in rule['consequents']:
                        recommendations[item].append((rec_item, rule['lift']))
                
                # Remove duplicates keeping highest lift
                item_to_lift = {}
                for rec_item, lift in recommendations[item]:
                    if rec_item not in item_to_lift or lift > item_to_lift[rec_item]:
                        item_to_lift[rec_item] = lift
                
                recommendations[item] = [(rec_item, lift) for rec_item, lift in item_to_lift.items()]
                recommendations[item].sort(key=lambda x: x[1], reverse=True)
    
    return recommendations

def generate_business_insights(rules_df: pd.DataFrame, 
                              transaction_data: pd.DataFrame,
                              top_n: int = 10) -> List[Dict]:
    """
    Generate actionable business insights from association rules.
    
    Args:
        rules_df: DataFrame containing association rules
        transaction_data: Original transaction data
        top_n: Number of top insights to generate
        
    Returns:
        List of dictionaries with insights and supporting data
    """
    insights = []
    
    if rules_df.empty:
        return [{"type": "warning", "text": "No association rules found to generate insights."}]
    
    # High-lift product combinations
    high_lift_rules = rules_df.sort_values('lift', ascending=False).head(top_n)
    for idx, rule in high_lift_rules.iterrows():
        antecedents = list(rule['antecedents']) if hasattr(rule['antecedents'], '__iter__') else [rule['antecedents']]
        consequents = list(rule['consequents']) if hasattr(rule['consequents'], '__iter__') else [rule['consequents']]
        insights.append({
            "type": "opportunity",
            "text": f"Products {', '.join(antecedents)} and {', '.join(consequents)} have a strong association (lift: {rule['lift']:.2f}).",
            "metric": "lift",
            "value": rule['lift'],
            "products": antecedents + consequents
        })
    
    # High-confidence rules for reliable recommendations
    high_conf_rules = rules_df.sort_values('confidence', ascending=False).head(top_n)
    for idx, rule in high_conf_rules.iloc[:3].iterrows():  # Top 3 only to avoid repetition
        antecedents = list(rule['antecedents']) if hasattr(rule['antecedents'], '__iter__') else [rule['antecedents']]
        consequents = list(rule['consequents']) if hasattr(rule['consequents'], '__iter__') else [rule['consequents']]
        insights.append({
            "type": "recommendation",
            "text": f"When customers buy {', '.join(antecedents)}, they are {rule['confidence']*100:.1f}% likely to also buy {', '.join(consequents)}.",
            "metric": "confidence",
            "value": rule['confidence'],
            "products": antecedents + consequents
        })
    
    # Surprising associations (high lift, lower support)
    surprising_rules = rules_df[rules_df['support'] < rules_df['support'].median()]
    surprising_rules = surprising_rules.sort_values('lift', ascending=False).head(3)
    for idx, rule in surprising_rules.iterrows():
        antecedents = list(rule['antecedents']) if hasattr(rule['antecedents'], '__iter__') else [rule['antecedents']]
        consequents = list(rule['consequents']) if hasattr(rule['consequents'], '__iter__') else [rule['consequents']]
        insights.append({
            "type": "discovery",
            "text": f"Surprising association: {', '.join(antecedents)} and {', '.join(consequents)} (lift: {rule['lift']:.2f}, support: {rule['support']:.3f}).",
            "metric": "surprise_factor",
            "value": rule['lift'] / (rule['support'] + 0.001),  # Avoid division by zero
            "products": antecedents + consequents
        })
    
    # Cross-category insights if we have category information
    if 'StockCode' in transaction_data.columns:
        insights.append({
            "type": "strategic",
            "text": "Consider analyzing cross-category purchasing patterns for store layout optimization.",
            "metric": "cross_category",
            "value": None
        })
    
    # Seasonality insights if we have time data
    if 'InvoiceDate' in transaction_data.columns:
        insights.append({
            "type": "temporal",
            "text": "Consider analyzing how purchase patterns change over time for inventory planning.",
            "metric": "seasonality",
            "value": None
        })
    
    return insights

def create_segment_recommendations(
    rules_df: pd.DataFrame,
    customer_segments: Dict[str, pd.DataFrame],
    n_recommendations: int = 5
) -> Dict[str, List[Dict]]:
    """
    Create targeted recommendations for different customer segments.
    
    Args:
        rules_df: DataFrame containing association rules
        customer_segments: Dictionary mapping segment names to customer DataFrames
        n_recommendations: Number of recommendations per segment
        
    Returns:
        Dictionary mapping segment names to lists of recommendation dictionaries
    """
    segment_recommendations = {}
    
    for segment_name, segment_df in customer_segments.items():
        recommendations = []
        
        if not rules_df.empty:
            # Sort rules by relevance to this segment
            # For illustration, use lift, but could be customized per segment
            sorted_rules = rules_df.sort_values('lift', ascending=False)
            
            # Generate top recommendations
            for idx, rule in sorted_rules.head(n_recommendations).iterrows():
                antecedents = list(rule['antecedents']) if hasattr(rule['antecedents'], '__iter__') else [rule['antecedents']]
                consequents = list(rule['consequents']) if hasattr(rule['consequents'], '__iter__') else [rule['consequents']]
                
                recommendations.append({
                    "products": consequents,
                    "trigger_products": antecedents,
                    "confidence": rule['confidence'],
                    "lift": rule['lift'],
                    "recommendation_text": f"Customers who buy {', '.join(antecedents)} may also be interested in {', '.join(consequents)}"
                })
        
        # Add segment-specific recommendations
        segment_recommendations[segment_name] = recommendations
        
    return segment_recommendations

def identify_cross_sell_opportunities(
    rules_df: pd.DataFrame,
    revenue_data: Optional[pd.DataFrame] = None,
    min_confidence: float = 0.3,
    min_lift: float = 1.5
) -> List[Dict]:
    """
    Identify cross-selling opportunities from association rules.
    
    Args:
        rules_df: DataFrame containing association rules
        revenue_data: Optional DataFrame with product revenue data
        min_confidence: Minimum confidence to consider
        min_lift: Minimum lift to consider
        
    Returns:
        List of dictionaries with cross-sell opportunities
    """
    opportunities = []
    
    if rules_df.empty:
        return opportunities
    
    # Filter rules by confidence and lift
    qualified_rules = rules_df[(rules_df['confidence'] >= min_confidence) & 
                              (rules_df['lift'] >= min_lift)]
    
    # Sort by most promising first
    qualified_rules = qualified_rules.sort_values(['lift', 'confidence'], ascending=False)
    
    for idx, rule in qualified_rules.head(10).iterrows():
        antecedents = list(rule['antecedents']) if hasattr(rule['antecedents'], '__iter__') else [rule['antecedents']]
        consequents = list(rule['consequents']) if hasattr(rule['consequents'], '__iter__') else [rule['consequents']]
        
        opportunity = {
            "trigger_products": antecedents,
            "recommended_products": consequents,
            "confidence": rule['confidence'],
            "lift": rule['lift'],
            "description": f"Cross-sell {', '.join(consequents)} to customers purchasing {', '.join(antecedents)}",
            "implementation": "Add as recommended products or bundle offer"
        }
        
        # Add revenue impact if data available
        if revenue_data is not None:
            # Simplified calculation - would need actual price data
            opportunity["potential_revenue_impact"] = "Medium"
            
        opportunities.append(opportunity)
    
    return opportunities

def create_executive_summary(
    rules_df: pd.DataFrame,
    transaction_data: pd.DataFrame,
    insights: List[Dict]
) -> Dict:
    """
    Create an executive summary of association rule analysis.
    
    Args:
        rules_df: DataFrame containing association rules
        transaction_data: Original transaction data
        insights: List of generated insights
        
    Returns:
        Dictionary with executive summary components
    """
    summary = {
        "total_transactions": len(transaction_data['InvoiceNo'].unique()) if 'InvoiceNo' in transaction_data.columns else "N/A",
        "total_products": len(transaction_data['StockCode'].unique()) if 'StockCode' in transaction_data.columns else "N/A",
        "total_rules_found": len(rules_df),
        "average_confidence": rules_df['confidence'].mean() if not rules_df.empty else 0,
        "average_lift": rules_df['lift'].mean() if not rules_df.empty else 0,
        "key_insights": [insight["text"] for insight in insights[:3]],
        "top_opportunities": [],
        "recommendation_summary": "Based on the association rules analysis, there are several actionable opportunities to improve sales through targeted cross-selling, product bundling, and store layout optimization."
    }
    
    # Extract top opportunities
    if not rules_df.empty:
        opportunity_rules = rules_df.sort_values('lift', ascending=False).head(3)
        for idx, rule in opportunity_rules.iterrows():
            antecedents = list(rule['antecedents']) if hasattr(rule['antecedents'], '__iter__') else [rule['antecedents']]
            consequents = list(rule['consequents']) if hasattr(rule['consequents'], '__iter__') else [rule['consequents']]
            summary["top_opportunities"].append(
                f"Promote {', '.join(consequents)} to customers who buy {', '.join(antecedents)}"
            )
    
    return summary 