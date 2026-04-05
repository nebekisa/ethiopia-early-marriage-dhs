"""
data_loader.py
Reusable module for loading DHS .DTA files
Author: Your Name
Date: 2024
"""

import pandas as pd

try:
    import pyreadstat
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing required package 'pyreadstat'. Install it with:\n"
        "    python -m pip install pyreadstat\n"
        "or:\n"
        "    python -m pip install -r requirements.txt"
    ) from exc

from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dhs_file(file_path, year=None):
    """
    Load a DHS .DTA file and return a pandas DataFrame.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to the .DTA file
    year : int, optional
        Survey year (for logging and reference)
    
    Returns:
    --------
    df : pandas DataFrame
        Loaded data
    metadata : dict
        Dictionary with file metadata
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logging.info(f"Loading {file_path.name}...")
    
    # Load .DTA file with pyreadstat to preserve value labels
    df, meta = pyreadstat.read_dta(file_path)
    
    logging.info(f"Successfully loaded {len(df):,} rows and {len(df.columns)} columns")
    
    metadata = {
        'file_name': file_path.name,
        'year': year,
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'column_names': list(df.columns),
        'value_labels': meta.variable_value_labels
    }
    
    return df, metadata

def load_all_ethiopia_dhs(raw_data_dir):
    """
    Load all Ethiopia DHS IR files from 2000-2016.
    
    Parameters:
    -----------
    raw_data_dir : Path
        Path to raw data directory
    
    Returns:
    --------
    dict : Dictionary with year as key and (df, metadata) as value
    """
    file_mapping = {
        2000: "ETIR41FL.DTA",
        2005: "ETIR51FL.DTA", 
        2011: "ETIR61FL.DTA",
        2016: "ETIR71FL.DTA"
    }
    
    data_dict = {}
    
    for year, filename in file_mapping.items():
        file_path = raw_data_dir / filename
        if file_path.exists():
            df, metadata = load_dhs_file(file_path, year)
            data_dict[year] = {
                'df': df,
                'metadata': metadata
            }
        else:
            logging.warning(f"File not found: {file_path}")
    
    return data_dict

def get_variable_info(df, variable_name):
    """
    Get information about a specific DHS variable.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Loaded DHS data
    variable_name : str
        DHS variable name (e.g., 'v511')
    
    Returns:
    --------
    dict : Variable information including unique values and description
    """
    if variable_name not in df.columns:
        return {"error": f"Variable {variable_name} not found in DataFrame"}
    
    info = {
        'name': variable_name,
        'dtype': str(df[variable_name].dtype),
        'n_missing': df[variable_name].isna().sum(),
        'n_unique': df[variable_name].nunique(),
        'unique_values': df[variable_name].value_counts().head(10).to_dict()
    }
    
    return info

if __name__ == "__main__":
    # Test the loader when run directly
    from src.config import RAW_DATA_DIR
    print(f"Raw data directory: {RAW_DATA_DIR}")
    
    # Load one file to test
    test_file = RAW_DATA_DIR / "ETIR41FL.DTA"
    if test_file.exists():
        df, meta = load_dhs_file(test_file, 2000)
        print(f"\nFirst 5 columns: {list(df.columns[:10])}")
        print(f"\nDataset shape: {df.shape}")
    else:
        print(f"Test file not found. Please place DHS files in {RAW_DATA_DIR}")