import pandas as pd
import polars as pl
import os
import time
import streamlit as st
from typing import Tuple, Optional, Union
import gc

def convert_excel_to_parquet(excel_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convert Excel file to Parquet format for faster loading
    
    Args:
        excel_path: Path to Excel file
        output_dir: Directory to save the Parquet file (defaults to same directory)
        
    Returns:
        Path to the created Parquet file
    """
    start_time = time.time()
    print(f"Converting {excel_path} to Parquet...")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(excel_path)
    
    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.parquet")
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"Parquet file already exists at {output_path}")
        return output_path
    
    # Read Excel in chunks to avoid memory issues
    excel_file = pd.ExcelFile(excel_path, engine='openpyxl')
    sheet_name = excel_file.sheet_names[0]
    
    # Get sheet dimensions
    xls = excel_file.book.worksheets[0]
    row_count = xls.max_row
    
    # Process in chunks
    chunk_size = 5000
    dfs = []
    
    for i in range(0, row_count, chunk_size):
        end_row = min(i + chunk_size, row_count)
        print(f"Reading Excel rows {i} to {end_row} of {row_count}")
        
        chunk_df = pd.read_excel(
            excel_file, 
            sheet_name=sheet_name,
            skiprows=i if i > 0 else None,
            nrows=chunk_size
        )
        
        dfs.append(chunk_df)
        gc.collect()  # Free memory
    
    # Combine chunks
    df = pd.concat(dfs, ignore_index=True)
    
    # Write to Parquet
    df.to_parquet(output_path, index=False)
    
    # Close Excel file and free memory
    excel_file.close()
    del dfs, df
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"Conversion completed in {elapsed:.2f} seconds. Parquet file saved to {output_path}")
    
    return output_path

def convert_excel_to_csv(excel_path: str, output_dir: Optional[str] = None) -> str:
    """
    Convert Excel file to CSV format for faster loading
    
    Args:
        excel_path: Path to Excel file
        output_dir: Directory to save the CSV file (defaults to same directory)
        
    Returns:
        Path to the created CSV file
    """
    start_time = time.time()
    print(f"Converting {excel_path} to CSV...")
    
    # Determine output path
    if output_dir is None:
        output_dir = os.path.dirname(excel_path)
    
    base_name = os.path.splitext(os.path.basename(excel_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.csv")
    
    # Check if file already exists
    if os.path.exists(output_path):
        print(f"CSV file already exists at {output_path}")
        return output_path
    
    # Read Excel in chunks to avoid memory issues
    excel_file = pd.ExcelFile(excel_path, engine='openpyxl')
    sheet_name = excel_file.sheet_names[0]
    
    # Get sheet dimensions
    xls = excel_file.book.worksheets[0]
    row_count = xls.max_row
    
    # Process in chunks
    chunk_size = 5000
    
    # Open CSV file for writing
    with open(output_path, 'w', encoding='utf-8') as f:
        # Write header
        header_df = pd.read_excel(excel_file, sheet_name=sheet_name, nrows=0)
        f.write(','.join(header_df.columns) + '\n')
        
        # Write data chunks
        for i in range(0, row_count, chunk_size):
            end_row = min(i + chunk_size, row_count)
            print(f"Reading Excel rows {i} to {end_row} of {row_count}")
            
            chunk_df = pd.read_excel(
                excel_file, 
                sheet_name=sheet_name,
                skiprows=i if i > 0 else None,
                nrows=chunk_size
            )
            
            # Skip header for all but first chunk
            if i > 0:
                chunk_df.to_csv(f, header=False, index=False, mode='a')
            else:
                chunk_df.to_csv(f, index=False, mode='a')
            
            # Free memory
            del chunk_df
            gc.collect()
    
    # Close Excel file and free memory
    excel_file.close()
    gc.collect()
    
    elapsed = time.time() - start_time
    print(f"Conversion completed in {elapsed:.2f} seconds. CSV file saved to {output_path}")
    
    return output_path

def get_optimized_file_path(file_path: str) -> str:
    """
    Get the path to the most optimized version of a file
    
    Args:
        file_path: Original file path
        
    Returns:
        Path to optimized file (parquet > csv > original)
    """
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    dir_name = os.path.dirname(file_path)
    
    # Check for Parquet version
    parquet_path = os.path.join(dir_name, f"{base_name}.parquet")
    if os.path.exists(parquet_path):
        return parquet_path
    
    # Check for CSV version
    csv_path = os.path.join(dir_name, f"{base_name}.csv")
    if os.path.exists(csv_path):
        return csv_path
    
    # Return original path if no optimized version exists
    return file_path 