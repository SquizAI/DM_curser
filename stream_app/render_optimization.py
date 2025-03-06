import streamlit as st
import pandas as pd
import gc
import psutil
import os
import time
import threading
from typing import Callable, Any

# Memory threshold in MB (80% of Render free tier limit)
MEMORY_THRESHOLD = 450  # Render free tier has ~512MB limit

def get_memory_usage():
    """Return the current memory usage in MB."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss / 1024 / 1024  # Convert to MB

def log_memory_usage(message: str = ""):
    """Log current memory usage."""
    mem_usage = get_memory_usage()
    print(f"Memory usage {message}: {mem_usage:.2f} MB")
    return mem_usage

def memory_optimize(func: Callable) -> Callable:
    """Decorator to optimize memory usage around function calls."""
    def wrapper(*args, **kwargs) -> Any:
        # Force garbage collection before function call
        gc.collect()
        start_mem = log_memory_usage(f"before {func.__name__}")
        start_time = time.time()
        
        # Check if we're already near the memory limit
        if start_mem > MEMORY_THRESHOLD:
            print(f"WARNING: Memory usage too high ({start_mem:.2f} MB) before executing {func.__name__}")
            st.warning(f"Memory usage is high ({start_mem:.2f} MB). This may affect performance.")
            # Force aggressive garbage collection
            for _ in range(3):
                gc.collect()
        
        result = func(*args, **kwargs)
        
        end_time = time.time()
        end_mem = log_memory_usage(f"after {func.__name__}")
        
        print(f"Function {func.__name__} took {end_time - start_time:.2f} seconds")
        print(f"Memory change: {end_mem - start_mem:.2f} MB")
        
        # Force garbage collection after function call
        gc.collect()
        return result
    
    return wrapper

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize memory usage of a pandas DataFrame."""
    start_mem = df.memory_usage().sum() / 1024 / 1024
    print(f"Initial DataFrame memory usage: {start_mem:.2f} MB")
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    # Optimize string columns
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # If column has fewer unique values
            df[col] = df[col].astype('category')
    
    end_mem = df.memory_usage().sum() / 1024 / 1024
    print(f"Optimized DataFrame memory usage: {end_mem:.2f} MB")
    print(f"Memory reduction: {100 * (start_mem - end_mem) / start_mem:.2f}%")
    
    return df

def monitor_memory_usage(interval=30):
    """Start a thread to periodically monitor memory usage."""
    def _monitor():
        while True:
            mem_usage = get_memory_usage()
            print(f"Periodic memory check: {mem_usage:.2f} MB")
            
            if mem_usage > MEMORY_THRESHOLD:
                print(f"WARNING: High memory usage detected: {mem_usage:.2f} MB")
                # Clear Streamlit cache
                st.cache_data.clear()
                # Force garbage collection
                for _ in range(3):
                    gc.collect()
                
                # Free large objects in session state
                for key in list(st.session_state.keys()):
                    if isinstance(st.session_state[key], (pd.DataFrame, list, dict)) and key != 'important_state':
                        print(f"Removing large object from session state: {key}")
                        del st.session_state[key]
            
            time.sleep(interval)
    
    # Start the monitoring thread
    thread = threading.Thread(target=_monitor, daemon=True)
    thread.start()
    return thread

# Set Streamlit config options for memory optimization
def apply_streamlit_optimizations():
    """Apply memory optimizations for Streamlit on Render."""
    # Reduce memory used by session state
    if 'large_data' in st.session_state:
        del st.session_state['large_data']
        gc.collect()
    
    # Optimize cache usage
    st.cache_data.clear()
    
    # Set environment variables for better performance
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    
    # Start memory monitoring in background
    monitor_memory_usage(interval=30)
    
    # Print initial memory usage
    print(f"Initial memory usage: {get_memory_usage():.2f} MB")
    print(f"Memory threshold: {MEMORY_THRESHOLD} MB") 