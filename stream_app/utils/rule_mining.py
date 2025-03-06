import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from typing import Dict, List, Tuple, Optional, Union, Set
import streamlit as st
import concurrent.futures
from functools import partial
from mlxtend.preprocessing import TransactionEncoder

@st.cache_data
def get_rules(basket_encoded: Union[pd.DataFrame, Dict[str, Set]], 
              min_support: float = 0.01, 
              min_confidence: float = 0.3, 
              min_lift: float = 1.0,
              algorithm: str = 'apriori',
              max_len: Optional[int] = None) -> pd.DataFrame:
    """
    Generate association rules using specified algorithm with enhanced performance.
    
    Args:
        basket_encoded: Binary encoded transaction data or dictionary of transaction sets
        min_support: Minimum support threshold
        min_confidence: Minimum confidence threshold
        min_lift: Minimum lift threshold
        algorithm: Algorithm to use ('apriori' or 'fpgrowth')
        max_len: Maximum length of itemsets
        
    Returns:
        DataFrame of association rules
    """
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
    else:
        # Already a DataFrame
        basket_df = basket_encoded
    
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

@st.cache_data
def process_rules_batch(_df: pd.DataFrame, batch_size: int = 1000, 
                        min_support: float = 0.01, 
                        min_confidence: float = 0.3,
                        min_lift: float = 1.0,
                        algorithm: str = 'apriori') -> pd.DataFrame:
    """
    Process larger datasets in batches to avoid memory issues
    
    Args:
        _df: Original transaction dataframe
        batch_size: Size of each batch
        min_support, min_confidence, min_lift: Thresholds
        algorithm: 'apriori' or 'fpgrowth'
        
    Returns:
        Combined rules DataFrame
    """
    # Get unique invoice numbers
    invoices = _df['InvoiceNo'].unique()
    
    # Split into batches
    batches = [invoices[i:i + batch_size] for i in range(0, len(invoices), batch_size)]
    
    all_rules = []
    
    for batch in batches:
        # Filter data for this batch
        batch_df = _df[_df['InvoiceNo'].isin(batch)]
        
        # Convert to basket format
        basket = batch_df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0)
        basket_sets = (basket.drop('InvoiceNo', axis=1) > 0).astype(bool)
        
        # Get rules
        rules = get_rules(basket_sets, min_support, min_confidence, min_lift, algorithm)
        
        if not rules.empty:
            all_rules.append(rules)
    
    # Combine all rules
    if all_rules:
        combined_rules = pd.concat(all_rules)
        # Remove duplicates
        combined_rules = combined_rules.drop_duplicates()
        return combined_rules
    else:
        return pd.DataFrame(columns=['antecedents', 'consequents', 'support', 
                                   'confidence', 'lift', 'leverage', 'conviction'])

@st.cache_data
def analyze_rules_over_time(time_datasets: Dict[str, pd.DataFrame], 
                          min_support: float = 0.01,
                          min_confidence: float = 0.3,
                          min_lift: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Analyze how rules change over time periods
    
    Args:
        time_datasets: Dictionary of time-based dataframes
        min_support, min_confidence, min_lift: Thresholds
        
    Returns:
        Dictionary of time-based rules
    """
    time_rules = {}
    
    # Process each time period
    for time_key, df in time_datasets.items():
        # Skip if not enough transactions
        if len(df['InvoiceNo'].unique()) < 10:
            continue
            
        # Convert to basket format
        basket = df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0)
        basket_sets = (basket.drop('InvoiceNo', axis=1) > 0).astype(bool)
        
        # Get rules
        rules = get_rules(basket_sets, min_support, min_confidence, min_lift)
        
        if not rules.empty:
            time_rules[time_key] = rules
    
    return time_rules

def prune_redundant_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """
    Remove redundant or less valuable rules
    
    Args:
        rules: Association rules DataFrame
        
    Returns:
        Pruned rules DataFrame
    """
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
        antecedent = tuple(rule['antecedents'])
        
        # Simple pruning: if we've seen this exact consequent before with higher lift, skip
        if consequent in seen_consequents:
            continue
            
        pruned_rules.append(rule)
        seen_consequents.add(consequent)
    
    return pd.DataFrame(pruned_rules)

def detect_insights(rules: pd.DataFrame) -> List[Dict[str, Union[str, float]]]:
    """
    Automatically detect interesting insights from rules
    
    Args:
        rules: Association rules DataFrame
        
    Returns:
        List of insight dictionaries
    """
    insights = []
    
    if rules.empty:
        return insights
    
    # Find rules with high lift (strong associations)
    high_lift_rules = rules[rules['lift'] > 3]
    if not high_lift_rules.empty:
        for _, rule in high_lift_rules.head(5).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insight = {
                'type': 'strong_association',
                'description': f"Strong association between {antecedents} and {consequents}",
                'metric': f"Lift: {rule['lift']:.2f}",
                'business_value': "Consider cross-merchandising these products"
            }
            insights.append(insight)
    
    # Find hidden gems (high lift, low support)
    hidden_gems = rules[(rules['lift'] > 3) & (rules['support'] < 0.05)]
    if not hidden_gems.empty:
        for _, rule in hidden_gems.head(3).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insight = {
                'type': 'hidden_gem',
                'description': f"Hidden gem: {antecedents} → {consequents}",
                'metric': f"Lift: {rule['lift']:.2f}, Support: {rule['support']:.3f}",
                'business_value': "Potential for targeted marketing to niche customer segments"
            }
            insights.append(insight)
    
    # Find highly confident rules
    confident_rules = rules[rules['confidence'] > 0.8]
    if not confident_rules.empty:
        for _, rule in confident_rules.head(3).iterrows():
            antecedents = ', '.join(list(rule['antecedents']))
            consequents = ', '.join(list(rule['consequents']))
            insight = {
                'type': 'high_confidence',
                'description': f"Highly confident: {antecedents} → {consequents}",
                'metric': f"Confidence: {rule['confidence']:.2f}",
                'business_value': "Strong recommendation candidates"
            }
            insights.append(insight)
    
    return insights 

@st.cache_data
def mine_rules_by_segment(df: pd.DataFrame,
                         segments_df: pd.DataFrame,
                         min_support: float = 0.01,
                         min_confidence: float = 0.3,
                         min_lift: float = 1.0) -> Dict[str, pd.DataFrame]:
    """
    Mine association rules for each customer segment
    
    Args:
        df: Original transaction DataFrame
        segments_df: DataFrame with customer segments
        min_support, min_confidence, min_lift: Thresholds for rule mining
        
    Returns:
        Dictionary mapping segment names to rule DataFrames
    """
    segment_rules = {}
    
    # Get segment column name - different depending on segmentation method
    if 'Segment' in segments_df.columns:
        segment_col = 'Segment'
    elif 'SpendSegment' in segments_df.columns:
        segment_col = 'SpendSegment'
    elif 'FrequencySegment' in segments_df.columns:
        segment_col = 'FrequencySegment'
    else:
        # No valid segment column found
        return {}
    
    # Get unique segments
    unique_segments = segments_df[segment_col].unique()
    
    # For each segment, filter transactions and mine rules
    for segment in unique_segments:
        # Get customer IDs for this segment
        segment_customers = segments_df[segments_df[segment_col] == segment]['CustomerID'].unique()
        
        # Filter transactions for these customers
        segment_df = df[df['CustomerID'].isin(segment_customers)]
        
        # Skip if not enough transactions
        if len(segment_df['InvoiceNo'].unique()) < 10:
            segment_rules[segment] = pd.DataFrame(columns=['antecedents', 'consequents', 'support', 
                                                          'confidence', 'lift', 'leverage', 'conviction'])
            continue
        
        # Convert to basket format
        basket = segment_df.groupby(['InvoiceNo', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0)
        basket_sets = (basket.drop('InvoiceNo', axis=1) > 0).astype(bool)
        
        # Get rules
        rules = get_rules(basket_sets, min_support, min_confidence, min_lift)
        
        # Store rules for this segment
        segment_rules[segment] = rules
    
    return segment_rules 