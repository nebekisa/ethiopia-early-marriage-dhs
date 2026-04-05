"""
wealth_proxy.py
Create consistent wealth index using PCA on household assets
Author: Bereket Andualem
Date: 2026
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_assets_for_pca(df, asset_cols):
    """
    Prepare asset columns for PCA
    Ensure all are numeric and handle missing values
    """
    X = df[asset_cols].copy()
    
    # Convert to numeric if needed
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Fill missing with median (conservative approach)
    for col in X.columns:
        if X[col].isna().any():
            median_val = X[col].median()
            X[col].fillna(median_val, inplace=True)
            logging.info(f"Filled {X[col].isna().sum()} missing in {col} with median={median_val}")
    
    return X

def create_wealth_pca(df, asset_cols, year):
    """
    Create wealth index using PCA on household assets
    Returns df with wealth_proxy_score and wealth_quintile
    """
    logging.info(f"Creating wealth proxy for {year} using assets: {asset_cols}")
    
    # Prepare assets
    X = prepare_assets_for_pca(df, asset_cols)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA - first component
    pca = PCA(n_components=1)
    wealth_score = pca.fit_transform(X_scaled)
    
    # Add to dataframe
    df['wealth_proxy_score'] = wealth_score.flatten()
    
    # Create quintiles
    try:
        df['wealth_quintile'] = pd.qcut(df['wealth_proxy_score'], q=5, 
                                         labels=['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'])
        df['wealth_quintile_num'] = pd.qcut(df['wealth_proxy_score'], q=5, labels=False) + 1
    except ValueError:
        # Handle case where too many identical values
        df['wealth_quintile'] = pd.cut(df['wealth_proxy_score'], bins=5, 
                                        labels=['Poorest', 'Poorer', 'Middle', 'Richer', 'Richest'])
        df['wealth_quintile_num'] = pd.cut(df['wealth_proxy_score'], bins=5, labels=False) + 1
    
    # Log variance explained
    logging.info(f"PCA first component explains {pca.explained_variance_ratio_[0]*100:.1f}% of variance")
    
    # Show component loadings
    loadings = pd.Series(pca.components_[0], index=asset_cols)
    logging.info(f"Component loadings:\n{loadings}")
    
    return df

def validate_wealth_proxy(df_2005, df_2011, df_2016):
    """
    Validate PCA proxy against DHS official wealth index (v190)
    for years where both exist
    """
    validation_results = {}
    
    for year, df in [(2005, df_2005), (2011, df_2011), (2016, df_2016)]:
        if 'v190' in df.columns and 'wealth_proxy_score' in df.columns:
            # Clean v190
            v190_clean = pd.to_numeric(df['v190'], errors='coerce')
            
            # Calculate correlation
            valid_mask = v190_clean.notna() & df['wealth_proxy_score'].notna()
            corr = df.loc[valid_mask, 'wealth_proxy_score'].corr(v190_clean[valid_mask])
            
            validation_results[year] = {
                'correlation': corr,
                'n_valid': valid_mask.sum(),
                'interpretation': 'Good' if corr > 0.7 else 'Moderate' if corr > 0.5 else 'Poor'
            }
            
            logging.info(f"{year}: Correlation between PCA proxy and DHS wealth: {corr:.3f} ({validation_results[year]['interpretation']})")
    
    return validation_results

def create_consistent_wealth_for_all_years(cleaned_data):
    """
    Create consistent wealth proxy across all years
    Uses same asset columns for all years
    """
    asset_cols = ['has_electricity', 'water_improved', 'toilet_improved', 'floor_finished']
    
    for year, df in cleaned_data.items():
        # Check if all asset columns exist
        available_assets = [col for col in asset_cols if col in df.columns]
        
        if len(available_assets) >= 3:  # Need at least 3 assets
            df = create_wealth_pca(df, available_assets, year)
            cleaned_data[year] = df
        else:
            logging.warning(f"{year}: Only {len(available_assets)} assets available - skipping wealth proxy")
    
    # Validate for years with DHS wealth index
    if 2005 in cleaned_data and 2011 in cleaned_data and 2016 in cleaned_data:
        validate_wealth_proxy(
            cleaned_data.get(2005), 
            cleaned_data.get(2011), 
            cleaned_data.get(2016)
        )
    
    return cleaned_data

if __name__ == "__main__":
    print("Wealth proxy module loaded successfully")