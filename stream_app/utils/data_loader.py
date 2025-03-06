import pandas as pd
import polars as pl
import numpy as np
from typing import Tuple, Optional, Union, List, Set
import streamlit as st
from datetime import datetime

@st.cache_data
def load_and_prep_data(file_path: Optional[str] = None, 
                       file = None,
                       dataframe: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, Set]:
    """
    Load and preprocess transaction data from a file or uploaded file
    
    Args:
        file_path: Path to the file
        file: Uploaded file object
        dataframe: Existing DataFrame
        
    Returns:
        Tuple of (processed DataFrame, encoded basket)
    """
    try:
        # Load data from one of the sources
        if dataframe is not None:
            df = dataframe.copy()
        elif file is not None:
            try:
                # Try to determine file type
                if hasattr(file, 'name'):
                    if file.name.endswith('.csv'):
                        df = pd.read_csv(file)
                    elif file.name.endswith(('.xlsx', '.xls')):
                        # Use explicit dtypes to avoid 'int' object decoding error
                        df = pd.read_excel(
                            file, 
                            dtype={
                                'InvoiceNo': str,
                                'StockCode': str,
                                'CustomerID': str
                            },
                            engine='openpyxl'
                        )
                    else:
                        st.error(f"Unsupported file format: {file.name}")
                        return pd.DataFrame(), set()
                else:
                    # If file type can't be determined, try Excel first, then CSV
                    try:
                        df = pd.read_excel(
                            file, 
                            dtype={
                                'InvoiceNo': str,
                                'StockCode': str,
                                'CustomerID': str
                            },
                            engine='openpyxl'
                        )
                    except Exception as e:
                        st.warning(f"Could not read as Excel, trying CSV: {str(e)}")
                        df = pd.read_csv(file)
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                return pd.DataFrame(), set()
        elif file_path is not None:
            try:
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(('.xlsx', '.xls')):
                    # Use explicit dtypes to avoid 'int' object decoding error
                    df = pd.read_excel(
                        file_path,
                        dtype={
                            'InvoiceNo': str,
                            'StockCode': str,
                            'CustomerID': str
                        },
                        engine='openpyxl'
                    )
                else:
                    st.error(f"Unsupported file format: {file_path}")
                    return pd.DataFrame(), set()
            except Exception as e:
                st.error(f"Error reading file from path: {str(e)}")
                return pd.DataFrame(), set()
        else:
            st.error("No data source provided")
            return pd.DataFrame(), set()
        
        # Check if the required columns exist
        required_columns = ['InvoiceNo', 'Description']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            return pd.DataFrame(), set()
        
        # Basic preprocessing
        # Ensure InvoiceNo and StockCode are treated as strings
        if 'InvoiceNo' in df.columns:
            df['InvoiceNo'] = df['InvoiceNo'].astype(str)
        if 'StockCode' in df.columns:
            df['StockCode'] = df['StockCode'].astype(str)
            
        # Drop records with missing InvoiceNo or Description
        df = df.dropna(subset=['InvoiceNo', 'Description'])
        
        # Filter out returns (invoices starting with 'C')
        df = df[~df['InvoiceNo'].str.startswith('C')]
        
        # Create the basket (encoded transaction data)
        basket_df = df.groupby(['InvoiceNo', 'Description']).size().reset_index(name='Count')
        basket_encoded = basket_df.groupby('InvoiceNo')['Description'].apply(set).to_dict()
        
        return df, basket_encoded
        
    except Exception as e:
        st.error(f"Error in data loading and preparation: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        return pd.DataFrame(), set()

@st.cache_data
def create_time_based_datasets(df: pd.DataFrame, time_field: str = 'InvoiceDate', 
                              _time_granularity: str = 'month') -> dict:
    """
    Create time-based subsets for temporal analysis
    
    Args:
        df: Transaction DataFrame
        time_field: Column containing datetime information
        _time_granularity: Time granularity ('day', 'week', 'month', 'quarter')
        
    Returns:
        Dictionary of time-based dataframes
    """
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(df[time_field]):
        df[time_field] = pd.to_datetime(df[time_field])
    
    # Extract time components
    df['Year'] = df[time_field].dt.year
    df['Month'] = df[time_field].dt.month
    df['Week'] = df[time_field].dt.isocalendar().week
    df['Day'] = df[time_field].dt.day
    df['Quarter'] = df[time_field].dt.quarter
    
    # Create time-based datasets
    time_datasets = {}
    
    if _time_granularity == 'month':
        groups = df.groupby(['Year', 'Month'])
    elif _time_granularity == 'quarter':
        groups = df.groupby(['Year', 'Quarter'])
    elif _time_granularity == 'week':
        groups = df.groupby(['Year', 'Week'])
    else:  # day
        groups = df.groupby(['Year', 'Month', 'Day'])
    
    for name, group in groups:
        time_key = '-'.join([str(x) for x in name])
        time_datasets[time_key] = group
    
    return time_datasets

@st.cache_data
def segment_customers(df: pd.DataFrame, method: str = 'rfm') -> pd.DataFrame:
    """
    Segment customers based on different methods
    
    Args:
        df: Transaction DataFrame
        method: Segmentation method ('rfm', 'spend', 'frequency')
        
    Returns:
        DataFrame with customer segments
    """
    if 'CustomerID' not in df.columns:
        return None
    
    # Remove missing CustomerID
    customer_df = df.dropna(subset=['CustomerID'])
    
    if method == 'rfm':
        # RFM Segmentation
        # Recency
        max_date = customer_df['InvoiceDate'].max()
        customer_df['Recency'] = customer_df.groupby('CustomerID')['InvoiceDate'].transform(
            lambda x: (max_date - x.max()).days)
        
        # Frequency
        customer_df['Frequency'] = customer_df.groupby('CustomerID')['InvoiceNo'].transform('nunique')
        
        # Monetary
        customer_df['Monetary'] = customer_df.groupby('CustomerID').apply(
            lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(
            name='Monetary')['Monetary']
        
        # Get unique customers with RFM values
        rfm_df = customer_df.groupby('CustomerID').agg({
            'Recency': 'min',
            'Frequency': 'max',
            'Monetary': 'max'
        }).reset_index()
        
        # Create segments
        rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'], q=3, labels=[3, 2, 1])
        rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'], q=3, labels=[1, 2, 3])
        rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'], q=3, labels=[1, 2, 3])
        
        # Combine scores
        rfm_df['RFM_Score'] = rfm_df['R_Score'].astype(str) + rfm_df['F_Score'].astype(str) + rfm_df['M_Score'].astype(str)
        
        # Define segments
        segment_map = {
            '111': 'Lost Customers',
            '112': 'Lost Customers',
            '121': 'Lost Customers',
            '122': 'Lost Customers',
            '123': 'Lost Customers',
            '132': 'Lost Customers',
            '113': 'Lost Customers',
            '131': 'Lost Customers',
            '133': 'Lost Customers',
            '211': 'At Risk',
            '212': 'At Risk',
            '213': 'At Risk',
            '221': 'At Risk',
            '222': 'Needs Attention',
            '223': 'Promising',
            '231': 'At Risk',
            '232': 'Promising',
            '233': 'Loyal Customers',
            '311': 'New Customers',
            '312': 'New Customers',
            '313': 'New Customers',
            '321': 'New Customers',
            '322': 'Promising',
            '323': 'Loyal Customers',
            '331': 'Promising',
            '332': 'Loyal Customers',
            '333': 'Champions'
        }
        
        rfm_df['Segment'] = rfm_df['RFM_Score'].map(segment_map)
        
        return rfm_df
    
    elif method == 'spend':
        # Simple spending based segmentation
        spending = customer_df.groupby('CustomerID').apply(
            lambda x: (x['Quantity'] * x['UnitPrice']).sum()).reset_index(
            name='TotalSpend')
        
        spending['SpendSegment'] = pd.qcut(spending['TotalSpend'], q=3, 
                                          labels=['Low Spenders', 'Medium Spenders', 'High Spenders'])
        
        return spending
    
    elif method == 'frequency':
        # Frequency based segmentation
        frequency = customer_df.groupby('CustomerID')['InvoiceNo'].nunique().reset_index(
            name='PurchaseFrequency')
        
        frequency['FrequencySegment'] = pd.qcut(frequency['PurchaseFrequency'], q=3,
                                              labels=['Infrequent', 'Regular', 'Frequent'])
        
        return frequency 