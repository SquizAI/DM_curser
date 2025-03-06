import pandas as pd
import polars as pl
import os
import time
import streamlit as st
import numpy as np
from typing import Tuple, Dict, List, Optional, Set, Union
import gc
import pickle
import hashlib
from pathlib import Path

# Global cache directory
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

def generate_cache_key(file_path: str, sample_pct: Optional[int] = None) -> str:
    """Generate a unique cache key based on file path and options"""
    file_name = os.path.basename(file_path)
    file_mtime = os.path.getmtime(file_path)
    key_components = f"{file_name}_{file_mtime}"
    
    if sample_pct is not None:
        key_components += f"_sample_{sample_pct}"
    
    # Generate MD5 hash of the components
    return hashlib.md5(key_components.encode()).hexdigest()

def save_to_cache(df: pd.DataFrame, basket_encoded: Set, cache_key: str) -> None:
    """Save processed data to cache for future fast loading"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    # Prepare data for serialization
    cache_data = {
        'df': df,
        'basket_encoded': list(basket_encoded) if basket_encoded else []
    }
    
    # Save to pickle file
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f)
    
    print(f"Data saved to cache: {cache_path}")

def load_from_cache(cache_key: str) -> Tuple[Optional[pd.DataFrame], Optional[Set]]:
    """Load processed data from cache if available"""
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    if os.path.exists(cache_path):
        try:
            print(f"Loading data from cache: {cache_path}")
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            df = cache_data['df']
            basket_encoded = set(cache_data['basket_encoded']) if cache_data['basket_encoded'] else set()
            
            return df, basket_encoded
        except Exception as e:
            print(f"Error loading from cache: {e}")
    
    return None, None

def preprocess_retail_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Set]:
    """Preprocess the retail dataset specifically for optimal performance"""
    start_time = time.time()
    print(f"Preprocessing data with shape {df.shape}")
    
    # Force garbage collection before processing
    gc.collect()
    
    # Convert columns to best dtypes
    if 'InvoiceNo' in df.columns:
        df['InvoiceNo'] = df['InvoiceNo'].astype(str)
    if 'StockCode' in df.columns:
        df['StockCode'] = df['StockCode'].astype(str)
    if 'CustomerID' in df.columns:
        df['CustomerID'] = df['CustomerID'].astype(str)
    
    # Convert date column
    if 'InvoiceDate' in df.columns:
        if not pd.api.types.is_datetime64_dtype(df['InvoiceDate']):
            df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    
    # Create basket encodings
    basket_encoded = set()
    if 'Description' in df.columns and 'InvoiceNo' in df.columns:
        # Group items by invoice
        invoice_items = df.groupby('InvoiceNo')['Description'].apply(set).reset_index()
        
        # Extract unique items for each invoice
        for _, row in invoice_items.iterrows():
            invoice_items = frozenset(item for item in row['Description'] if isinstance(item, str))
            if invoice_items:
                basket_encoded.add(invoice_items)
    
    # Optimize memory usage
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Convert object columns to categories if they have few unique values
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    print(f"Preprocessing completed in {time.time() - start_time:.2f} seconds")
    return df, basket_encoded

def fast_load_sample_data(sample_percentage: Optional[int] = None, use_cache: bool = True) -> Tuple[pd.DataFrame, Set]:
    """
    Ultra-fast loading of sample data with smart caching and pre-processing
    
    Args:
        sample_percentage: Percentage of data to sample (1-100)
        use_cache: Whether to use cached data if available
        
    Returns:
        Tuple of (processed DataFrame, encoded basket)
    """
    start_time = time.time()
    
    # Path to the sample data (prioritize Parquet > CSV > Excel)
    data_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(data_dir, "Online Retail.parquet")
    csv_path = os.path.join(data_dir, "Online Retail.csv")
    excel_path = os.path.join(data_dir, "Online Retail.xlsx")
    
    # Determine the best available file format
    if os.path.exists(parquet_path):
        file_path = parquet_path
        file_type = "parquet"
    elif os.path.exists(csv_path):
        file_path = csv_path
        file_type = "csv"
    elif os.path.exists(excel_path):
        file_path = excel_path
        file_type = "excel"
    else:
        raise FileNotFoundError(f"Sample data not found in any format in {data_dir}")
    
    # Generate cache key
    cache_key = generate_cache_key(file_path, sample_percentage)
    
    # Try to load from cache first
    if use_cache:
        df, basket_encoded = load_from_cache(cache_key)
        if df is not None:
            print(f"Loaded from cache in {time.time() - start_time:.2f} seconds")
            return df, basket_encoded
    
    # If not in cache, load from file using the most efficient method
    print(f"Loading data from {file_type} file: {file_path}")
    
    try:
        if file_type == "parquet":
            df = pd.read_parquet(file_path)
        elif file_type == "csv":
            df = pd.read_csv(file_path, dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str})
        else:  # Excel
            # For Excel, use Polars since it's faster than pandas for Excel
            try:
                import polars as pl
                df = pl.read_excel(file_path).to_pandas()
            except:
                # Fall back to pandas if polars fails
                df = pd.read_excel(file_path, dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str})
        
        # Sample if requested
        if sample_percentage is not None and 1 <= sample_percentage < 100:
            sample_size = int(len(df) * (sample_percentage / 100))
            if len(df) > sample_size:
                df = df.sample(n=sample_size, random_state=42)
        
        # Preprocess the data
        df, basket_encoded = preprocess_retail_data(df)
        
        # Save to cache for future use
        save_to_cache(df, basket_encoded, cache_key)
        
        print(f"Loaded and processed in {time.time() - start_time:.2f} seconds")
        return df, basket_encoded
        
    except Exception as e:
        raise Exception(f"Error loading sample data: {str(e)}")

def convert_excel_to_optimized_format():
    """Convert the Excel file to Parquet and CSV for faster future loading"""
    data_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(data_dir, "Online Retail.xlsx")
    parquet_path = os.path.join(data_dir, "Online Retail.parquet")
    csv_path = os.path.join(data_dir, "Online Retail.csv")
    
    if not os.path.exists(excel_path):
        print(f"Excel file not found at {excel_path}")
        return
    
    # If parquet already exists and is newer than Excel, skip conversion
    if os.path.exists(parquet_path) and os.path.getmtime(parquet_path) > os.path.getmtime(excel_path):
        print(f"Parquet file already up to date: {parquet_path}")
        return
    
    print(f"Converting Excel to optimized formats...")
    start_time = time.time()
    
    try:
        # Read Excel file
        df = pd.read_excel(excel_path, dtype={'InvoiceNo': str, 'StockCode': str, 'CustomerID': str})
        
        # Save as Parquet (fastest for loading)
        df.to_parquet(parquet_path, index=False)
        print(f"Saved Parquet file: {parquet_path}")
        
        # Also save as CSV (as backup)
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV file: {csv_path}")
        
        print(f"Conversion completed in {time.time() - start_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error during conversion: {e}")

# Run conversion when this module is imported
try:
    convert_excel_to_optimized_format()
except Exception as e:
    print(f"Could not pre-convert data: {e}") 