import pandas as pd
import numpy as np
import polars as pl
from typing import List, Dict, Tuple, Set, Optional, Union, Callable, Any
import time
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial, wraps
import psutil
import os
import sys
import gc
import platform
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('performance_monitor')

def profile_memory_usage(df: Union[pd.DataFrame, pl.DataFrame] = None) -> Dict[str, Any]:
    """
    Profile memory usage of a DataFrame or the current process.
    
    Args:
        df: Optional DataFrame to profile
        
    Returns:
        Dictionary with memory usage statistics
    """
    # If a DataFrame is provided, profile it
    if df is not None:
        if isinstance(df, pd.DataFrame):
            memory_usage = df.memory_usage(deep=True)
            total_memory = memory_usage.sum()
            
            # Calculate memory per column
            column_memory = {}
            for col in df.columns:
                column_memory[col] = {
                    'memory': memory_usage[col],
                    'memory_percent': memory_usage[col] / total_memory * 100,
                    'dtype': df[col].dtype
                }
            
            return {
                'total_memory_mb': total_memory / 1e6,
                'column_memory': column_memory,
                'num_rows': len(df),
                'num_columns': len(df.columns),
                'memory_per_row_kb': total_memory / len(df) / 1e3
            }
        
        elif isinstance(df, pl.DataFrame):
            # Basic memory info for Polars
            estimated_size = sys.getsizeof(df)
            return {
                'total_memory_mb': estimated_size / 1e6,
                'num_rows': df.height,
                'num_columns': df.width,
                'memory_per_row_kb': estimated_size / df.height / 1e3 if df.height > 0 else 0,
                'schema': str(df.schema)
            }
    
    # Otherwise profile the current process
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        'rss': mem_info.rss / (1024 * 1024),  # RSS in MB
        'vms': mem_info.vms / (1024 * 1024),  # VMS in MB
        'percent': process.memory_percent()    # Percentage of system memory
    }

def time_function(func):
    """
    Decorator to time a function's execution
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function
    """
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time:.4f} seconds to run")
        return result
    
    return wrapper

def convert_pandas_to_polars(df: pd.DataFrame) -> pl.DataFrame:
    """
    Convert pandas DataFrame to polars DataFrame.
    
    Args:
        df: pandas DataFrame
        
    Returns:
        polars DataFrame
    """
    return pl.from_pandas(df)

def convert_polars_to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    """
    Convert polars DataFrame to pandas DataFrame.
    
    Args:
        df: polars DataFrame
        
    Returns:
        pandas DataFrame
    """
    return df.to_pandas()

def optimize_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize pandas DataFrame memory usage by converting dtypes
    
    Args:
        df: pandas DataFrame to optimize
        
    Returns:
        Optimized pandas DataFrame
    """
    optimized = df.copy()
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        # Downcast integers if possible
        optimized[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        # Downcast floats if possible
        optimized[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize object columns to categorical when beneficial
    for col in df.select_dtypes(include=['object']).columns:
        num_unique = df[col].nunique()
        num_total = len(df)
        
        # If less than 50% of values are unique, convert to categorical
        if num_unique / num_total < 0.5:
            optimized[col] = df[col].astype('category')
    
    return optimized

@st.cache_data
def parallel_process_data(df: pd.DataFrame, func, n_jobs: int = -1, partition_method: str = 'equal') -> List:
    """
    Process data in parallel using multiple cores
    
    Args:
        df: DataFrame to process
        func: Function to apply to each partition
        n_jobs: Number of jobs (-1 for all cores)
        partition_method: Method to partition data ('equal' or 'interleave')
        
    Returns:
        List of processed results
    """
    if n_jobs == -1:
        n_jobs = os.cpu_count()
    
    # Split dataframe
    if partition_method == 'equal':
        # Split into equal chunks
        chunks = np.array_split(df, n_jobs)
    else:
        # Interleaved partitioning
        chunks = [df.iloc[i::n_jobs] for i in range(n_jobs)]
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(func, chunks))
    
    return results

def optimize_polars_pipeline(data_processing_steps: List[callable], input_df: Union[pd.DataFrame, pl.DataFrame]) -> pl.DataFrame:
    """
    Optimize a data processing pipeline using Polars
    
    Args:
        data_processing_steps: List of functions to apply
        input_df: Input DataFrame (pandas or polars)
        
    Returns:
        Processed polars DataFrame
    """
    # Convert to polars if needed
    if isinstance(input_df, pd.DataFrame):
        df = pl.from_pandas(input_df)
    else:
        df = input_df
    
    # Start lazy execution
    lazy_df = df.lazy()
    
    # Apply each step
    for step in data_processing_steps:
        lazy_df = step(lazy_df)
    
    # Execute the pipeline
    return lazy_df.collect()

def scan_csv_lazy(file_path: str, columns: Optional[List[str]] = None) -> pl.LazyFrame:
    """
    Lazily scan a CSV file with Polars for memory-efficient processing
    
    Args:
        file_path: Path to CSV file
        columns: Optional list of columns to load
        
    Returns:
        LazyFrame for further processing
    """
    return pl.scan_csv(file_path, columns=columns)

def process_large_dataset(file_path: str, 
                         processing_func: callable, 
                         batch_size: int = 100000) -> pl.DataFrame:
    """
    Process a large dataset in batches to avoid memory issues
    
    Args:
        file_path: Path to data file
        processing_func: Function to apply to each batch
        batch_size: Number of rows per batch
        
    Returns:
        Processed DataFrame
    """
    # Initialize reader
    if file_path.endswith('.csv'):
        # For CSV, use streaming reader with batches
        reader = pl.read_csv_batched(file_path, batch_size=batch_size)
        results = []
        
        for batch in reader:
            # Process each batch
            processed = processing_func(batch)
            results.append(processed)
        
        # Combine results
        if results:
            return pl.concat(results)
        return pl.DataFrame()
        
    elif file_path.endswith(('.xls', '.xlsx')):
        # For Excel, we need to use pandas then convert
        # This is less memory efficient but Excel isn't ideal for large data anyway
        df = pd.read_excel(file_path)
        pl_df = pl.from_pandas(df)
        return processing_func(pl_df)
    
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def aggregate_rules_efficiently(transactions: pl.DataFrame, 
                             item_col: str, 
                             invoice_col: str) -> Dict[str, Set[str]]:
    """
    Efficiently aggregate transactions into invoice-item format
    
    Args:
        transactions: Transaction DataFrame
        item_col: Column containing item identifiers
        invoice_col: Column containing invoice/basket identifiers
        
    Returns:
        Dictionary mapping invoice IDs to sets of items
    """
    # Use polars to group items by invoice
    grouped = transactions.group_by(invoice_col).agg(
        pl.col(item_col).alias('items')
    )
    
    # Convert to dictionary format
    baskets = {}
    for row in grouped.iter_rows(named=True):
        invoice_id = row[invoice_col]
        items = set(row['items'])
        baskets[invoice_id] = items
    
    return baskets

def create_polars_transaction_encoder(transactions: pl.DataFrame, 
                                    item_col: str, 
                                    invoice_col: str) -> Tuple[np.ndarray, List[str]]:
    """
    Create binary encoded transaction array using Polars for improved performance
    
    Args:
        transactions: Transaction DataFrame
        item_col: Column containing item identifiers
        invoice_col: Column containing invoice identifiers
        
    Returns:
        Tuple of (encoded_array, unique_items)
    """
    # Get unique invoices and items
    unique_invoices = transactions[invoice_col].unique().to_list()
    unique_items = transactions[item_col].unique().to_list()
    
    # Create mapping dictionaries for faster lookups
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    invoice_to_idx = {invoice: idx for idx, invoice in enumerate(unique_invoices)}
    
    # Initialize binary array
    encoded_array = np.zeros((len(unique_invoices), len(unique_items)), dtype=bool)
    
    # Group by invoice and fill the array
    grouped = transactions.group_by(invoice_col).agg(
        pl.col(item_col).alias('items')
    )
    
    for row in grouped.iter_rows(named=True):
        invoice_id = row[invoice_col]
        items = row['items']
        
        invoice_idx = invoice_to_idx[invoice_id]
        for item in items:
            item_idx = item_to_idx[item]
            encoded_array[invoice_idx, item_idx] = True
    
    return encoded_array, unique_items

def benchmark_function(func: Callable) -> Callable:
    """
    Decorator to benchmark a function's runtime.
    
    Args:
        func: Function to benchmark
        
    Returns:
        Wrapped function with benchmarking
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if hasattr(st, 'session_state') and 'benchmarks' in st.session_state:
            if not isinstance(st.session_state.benchmarks, dict):
                st.session_state.benchmarks = {}
            func_name = func.__name__
            if func_name not in st.session_state.benchmarks:
                st.session_state.benchmarks[func_name] = []
            st.session_state.benchmarks[func_name].append(execution_time)
        return result
    return wrapper

def optimize_streamlit_performance():
    """Apply various optimizations to improve Streamlit app performance"""
    # 1. Configure page caching to avoid redundant computation
    # Note: st.set_page_config moved to app.py as it must be the first Streamlit command
    
    # 2. Check and log system information
    system_info = get_system_info()
    
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {
            'system_info': system_info,
            'page_load_time': time.time(),
            'function_timings': {},
            'memory_usage': []
        }
    
    # 3. Apply caching decorators to expensive functions
    st.cache_data.clear()
    
    # 4. Return the optimization state
    return {
        'system_info': system_info,
        'optimizations_applied': True
    }

# Create a decorator to measure function execution time
def measure_time(func):
    """Decorator to measure the execution time of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Log the execution time
        logger.info(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        
        # Store in session state for monitoring
        if 'performance_metrics' not in st.session_state:
            st.session_state.performance_metrics = {}
        
        if 'function_times' not in st.session_state.performance_metrics:
            st.session_state.performance_metrics['function_times'] = {}
        
        st.session_state.performance_metrics['function_times'][func.__name__] = execution_time
        
        return result
    return wrapper

def get_system_info() -> Dict[str, Any]:
    """Get system information for performance context"""
    system_info = {
        'platform': platform.system(),
        'platform_version': platform.version(),
        'processor': platform.processor(),
        'python_version': platform.python_version(),
        'cpu_count': psutil.cpu_count(logical=True),
        'physical_cpu_count': psutil.cpu_count(logical=False),
        'total_memory': f"{psutil.virtual_memory().total / (1024**3):.2f} GB"
    }
    return system_info

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage information"""
    memory = psutil.virtual_memory()
    return {
        'total_memory_mb': memory.total / (1024**2),
        'available_memory_mb': memory.available / (1024**2),
        'used_memory_mb': memory.used / (1024**2),
        'memory_percent': memory.percent
    }

@measure_time
def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize a pandas DataFrame to reduce memory usage.
    
    Args:
        df: Input DataFrame to optimize
        
    Returns:
        Optimized DataFrame with appropriate data types
    """
    start_mem = df.memory_usage().sum() / (1024**2)
    logger.info(f"DataFrame memory usage before optimization: {start_mem:.2f} MB")
    
    # List of numeric columns
    int_cols = df.select_dtypes(include=['int']).columns
    float_cols = df.select_dtypes(include=['float']).columns
    
    # Optimize integer columns
    for col in int_cols:
        col_min = df[col].min()
        col_max = df[col].max()
        
        # Convert to the smallest possible integer type
        if col_min >= 0:
            if col_max < 255:
                df[col] = df[col].astype(np.uint8)
            elif col_max < 65535:
                df[col] = df[col].astype(np.uint16)
            elif col_max < 4294967295:
                df[col] = df[col].astype(np.uint32)
            else:
                df[col] = df[col].astype(np.uint64)
        else:
            if col_min > -128 and col_max < 127:
                df[col] = df[col].astype(np.int8)
            elif col_min > -32768 and col_max < 32767:
                df[col] = df[col].astype(np.int16)
            elif col_min > -2147483648 and col_max < 2147483647:
                df[col] = df[col].astype(np.int32)
            else:
                df[col] = df[col].astype(np.int64)
    
    # Optimize float columns
    for col in float_cols:
        # Check if float32 is sufficient precision
        df[col] = df[col].astype(np.float32)
    
    # Convert object columns to categories if beneficial
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        # Only convert if the number of unique values is less than 50% of total values
        if df[col].nunique() / len(df) < 0.5:
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / (1024**2)
    logger.info(f"DataFrame memory usage after optimization: {end_mem:.2f} MB")
    logger.info(f"Memory usage reduced by {100 * (start_mem - end_mem) / start_mem:.2f}%")
    
    return df

@measure_time
def monitor_performance() -> Dict[str, Any]:
    """
    Collect and return performance metrics.
    
    Returns:
        Dictionary of performance metrics
    """
    # Get current memory usage
    memory_usage = get_memory_usage()
    
    # Get CPU usage
    cpu_percent = psutil.cpu_percent(interval=0.1)
    
    # Calculate uptime
    uptime = time.time() - st.session_state.monitoring_start_time
    
    metrics = {
        'memory_usage': memory_usage,
        'cpu_percent': cpu_percent,
        'uptime_seconds': uptime
    }
    
    # Store metrics in session state
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    
    st.session_state.performance_metrics['current'] = metrics
    
    if 'history' not in st.session_state.performance_metrics:
        st.session_state.performance_metrics['history'] = []
    
    # Store history (limit to 100 data points)
    if len(st.session_state.performance_metrics['history']) >= 100:
        st.session_state.performance_metrics['history'].pop(0)
    
    st.session_state.performance_metrics['history'].append({
        'timestamp': time.time(),
        'memory_percent': memory_usage['memory_percent'],
        'cpu_percent': cpu_percent
    })
    
    return metrics

def display_performance_metrics():
    """Display performance metrics in the Streamlit app"""
    if 'performance_metrics' not in st.session_state:
        st.warning("Performance metrics not available")
        return
    
    metrics = st.session_state.performance_metrics
    
    # Get current metrics
    if 'current' in metrics:
        current = metrics['current']
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Memory Usage", 
                f"{current['memory_usage']['used_memory_mb']:.1f} MB", 
                f"{current['memory_usage']['memory_percent']:.1f}%"
            )
        
        with col2:
            st.metric("CPU Usage", f"{current['cpu_percent']:.1f}%")
        
        with col3:
            st.metric(
                "Uptime", 
                f"{int(current['uptime_seconds'] // 60)} min {int(current['uptime_seconds'] % 60)} sec"
            )
    
    # Show function execution times
    if 'function_times' in metrics and metrics['function_times']:
        st.subheader("Function Execution Times")
        
        # Convert to DataFrame for better display
        times_df = pd.DataFrame([
            {"Function": func_name, "Execution Time (sec)": exec_time}
            for func_name, exec_time in metrics['function_times'].items()
        ])
        
        # Sort by execution time (descending)
        times_df = times_df.sort_values("Execution Time (sec)", ascending=False)
        
        # Display as a dataframe
        st.dataframe(times_df, use_container_width=True)
    
    # Show system information
    if 'system_info' in metrics:
        with st.expander("System Information"):
            for key, value in metrics['system_info'].items():
                st.text(f"{key}: {value}")

@measure_time
def optimize_batch_processing(func: Callable, data: List[Any], batch_size: int = 1000, **kwargs) -> List[Any]:
    """
    Process large datasets in batches to avoid memory issues.
    
    Args:
        func: Function to apply to each batch
        data: List of data items to process
        batch_size: Number of items to process in each batch
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        List of results from processing all batches
    """
    results = []
    total_items = len(data)
    
    # Process in batches
    for i in range(0, total_items, batch_size):
        batch = data[i:min(i + batch_size, total_items)]
        batch_results = func(batch, **kwargs)
        results.extend(batch_results)
        
        # Log progress
        progress = min(i + batch_size, total_items) / total_items * 100
        logger.info(f"Batch processing progress: {progress:.1f}% ({min(i + batch_size, total_items)}/{total_items})")
        
        # Collect garbage after each batch to free memory
        gc.collect()
    
    return results

def log_error(func_name: str, error: Exception):
    """Log errors with context information"""
    logger.error(f"Error in function '{func_name}': {str(error)}")
    
    # Store in session state
    if 'performance_metrics' not in st.session_state:
        st.session_state.performance_metrics = {}
    
    if 'errors' not in st.session_state.performance_metrics:
        st.session_state.performance_metrics['errors'] = []
    
    error_info = {
        'timestamp': time.time(),
        'function': func_name,
        'error': str(error)
    }
    
    st.session_state.performance_metrics['errors'].append(error_info)

def clear_cache():
    """Clear Streamlit cache to free up memory"""
    try:
        # Clear Streamlit cache
        st.cache_data.clear()
        st.cache_resource.clear()
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Cache cleared successfully")
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False

def get_performance_summary() -> Dict[str, Any]:
    """
    Generate a summary of performance metrics.
    
    Returns:
        Dictionary with performance summary
    """
    if 'performance_metrics' not in st.session_state:
        return {"status": "No performance data available"}
    
    metrics = st.session_state.performance_metrics
    
    # Calculate average CPU and memory usage if history exists
    avg_cpu = None
    avg_memory = None
    peak_memory = None
    
    if 'history' in metrics and metrics['history']:
        history = metrics['history']
        avg_cpu = sum(item['cpu_percent'] for item in history) / len(history)
        avg_memory = sum(item['memory_percent'] for item in history) / len(history)
        peak_memory = max(item['memory_percent'] for item in history)
    
    # Calculate average function execution times
    avg_func_times = {}
    if 'function_times' in metrics and metrics['function_times']:
        avg_func_times = metrics['function_times']
    
    # Count errors if any
    error_count = 0
    if 'errors' in metrics and metrics['errors']:
        error_count = len(metrics['errors'])
    
    # Create summary
    summary = {
        "avg_cpu_usage": avg_cpu,
        "avg_memory_usage": avg_memory,
        "peak_memory_usage": peak_memory,
        "function_times": avg_func_times,
        "error_count": error_count,
        "uptime": time.time() - st.session_state.get('monitoring_start_time', time.time())
    }
    
    return summary 