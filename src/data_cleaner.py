"""
data_cleaner.py
Professional data cleaning functions for DHS Ethiopia data
Author: Bereket Andualem
Date: 2026
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def clean_marital_status(df, year):
    """
    Clean marital status variable (v501)
    DHS codes: 0=never married, 1=married, 2=living together, 3=widowed, 4=divorced, 5=not living together
    """
    if 'v501' not in df.columns:
        logging.warning(f"v501 not found in {year} data")
        df['marital_status'] = np.nan
        df['ever_married'] = np.nan
        return df
    
    # Create standardized marital status
    marital_map = {
        0: 'never_married',
        1: 'married',
        2: 'living_together',
        3: 'widowed',
        4: 'divorced',
        5: 'separated'
    }
    
    df['marital_status'] = df['v501'].map(marital_map)
    df['ever_married'] = df['v501'].isin([1, 2, 3, 4, 5]).astype(int)
    
    logging.info(f"{year}: Cleaned marital status - {df['ever_married'].sum():,} ever married women")
    return df

def clean_age_first_marriage(df, year, min_age=10, max_age=49):
    """
    Clean age at first marriage variable (v511)
    - Convert to numeric
    - Set unrealistic ages to missing
    - DHS uses 0-6 for 'married before age 7', 99 for missing
    """
    if 'v511' not in df.columns:
        logging.warning(f"v511 not found in {year} data")
        df['age_first_marriage'] = np.nan
        df['early_marriage'] = np.nan
        return df
    
    # Convert to numeric
    df['age_first_marriage'] = pd.to_numeric(df['v511'], errors='coerce')
    
    # DHS missing codes: 99, 98, 96
    df.loc[df['age_first_marriage'] >= 95, 'age_first_marriage'] = np.nan
    
    # Set unrealistic ages to missing
    df.loc[df['age_first_marriage'] < min_age, 'age_first_marriage'] = np.nan
    df.loc[df['age_first_marriage'] > max_age, 'age_first_marriage'] = np.nan
    
    # Create early marriage indicator (only for ever-married women)
    df['early_marriage'] = np.where(
        (df['ever_married'] == 1) & (df['age_first_marriage'] < 18),
        1,
        np.where(df['ever_married'] == 1, 0, np.nan)
    )
    
    n_early = df['early_marriage'].sum()
    n_ever_married = df[df['ever_married'] == 1].shape[0]
    
    if n_ever_married > 0:
        rate = (n_early / n_ever_married) * 100
        logging.info(f"{year}: Early marriage rate among ever-married: {rate:.1f}% (n={n_early:,}/{n_ever_married:,})")
    
    return df

def clean_education(df, year):
    """
    Clean education variables (v106, v107, v155)
    """
    # Education level (v106)
    if 'v106' in df.columns:
        edu_map = {
            0: 'no_education',
            1: 'primary',
            2: 'secondary',
            3: 'higher'
        }
        df['education_level'] = df['v106'].map(edu_map)
        df['education_level_num'] = df['v106']
    else:
        df['education_level'] = np.nan
        df['education_level_num'] = np.nan
    
    # Years of education (v107)
    if 'v107' in df.columns:
        df['education_years'] = pd.to_numeric(df['v107'], errors='coerce')
        df.loc[df['education_years'] > 25, 'education_years'] = np.nan
    else:
        df['education_years'] = np.nan
    
    # Literacy (v155)
    if 'v155' in df.columns:
        # DHS: 0=cannot read, 1=can read, 2=blind/visually impaired
        df['literate'] = df['v155'].map({0: 0, 1: 1, 2: np.nan})
    else:
        df['literate'] = np.nan
    
    logging.info(f"{year}: Education - {df['education_level'].value_counts().to_dict()}")
    return df

def clean_demographics(df, year):
    """
    Clean demographic variables (age, residence, region)
    """
    # Current age (v012)
    if 'v012' in df.columns:
        df['current_age'] = pd.to_numeric(df['v012'], errors='coerce')
        df.loc[df['current_age'] > 95, 'current_age'] = np.nan
    else:
        df['current_age'] = np.nan
    
    # Residence (v025)
    if 'v025' in df.columns:
        df['residence'] = df['v025'].map({1: 'urban', 2: 'rural'})
        df['residence_urban'] = (df['v025'] == 1).astype(int)
    else:
        df['residence'] = np.nan
        df['residence_urban'] = np.nan
    
    # Region (v024)
    if 'v024' in df.columns:
        region_map = {
            1: 'Tigray', 2: 'Afar', 3: 'Amhara', 4: 'Oromia',
            5: 'Somali', 6: 'Benishangul-Gumuz', 7: 'SNNPR',
            8: 'Gambela', 9: 'Harari', 10: 'Addis_Ababa', 12: 'Dire_Dawa'
        }
        df['region'] = df['v024'].map(region_map)
        df['region_code'] = df['v024']
    else:
        df['region'] = np.nan
        df['region_code'] = np.nan
    
    # Religion (v130)
    if 'v130' in df.columns:
        religion_map = {
            1: 'Orthodox', 2: 'Muslim', 3: 'Catholic', 
            4: 'Protestant', 5: 'Traditional', 6: 'Other'
        }
        df['religion'] = df['v130'].map(religion_map)
    else:
        df['religion'] = np.nan
    
    # Ethnicity (v131)
    if 'v131' in df.columns:
        df['ethnicity'] = df['v131'].astype(str)
    else:
        df['ethnicity'] = np.nan
    
    logging.info(f"{year}: Demographics - Mean age: {df['current_age'].mean():.1f}, Rural: {df['residence_urban'].mean()*100:.1f}%")
    return df

def clean_fertility(df, year):
    """
    Clean fertility variables (children ever born, surviving)
    """
    # Children ever born (v201)
    if 'v201' in df.columns:
        df['children_ever_born'] = pd.to_numeric(df['v201'], errors='coerce')
    else:
        df['children_ever_born'] = np.nan
    
    # Children surviving (v202)
    if 'v202' in df.columns:
        df['children_surviving'] = pd.to_numeric(df['v202'], errors='coerce')
    else:
        df['children_surviving'] = np.nan
    
    # Children deceased
    df['children_deceased'] = df['children_ever_born'] - df['children_surviving']
    df['children_deceased'] = df['children_deceased'].clip(lower=0)
    
    # Ideal number of children (v613)
    if 'v613' in df.columns:
        df['ideal_children'] = pd.to_numeric(df['v613'], errors='coerce')
        df.loc[df['ideal_children'] > 20, 'ideal_children'] = np.nan
    else:
        df['ideal_children'] = np.nan
    
    logging.info(f"{year}: Fertility - Mean children: {df['children_ever_born'].mean():.2f}")
    return df

def clean_work(df, year):
    """
    Clean work/employment variable
    """
    if 'v714' in df.columns:
        # DHS: 0=not working, 1=working
        df['currently_working'] = pd.to_numeric(df['v714'], errors='coerce')
        df['currently_working'] = df['currently_working'].map({0: 0, 1: 1})
    else:
        df['currently_working'] = np.nan
    
    logging.info(f"{year}: Working women: {df['currently_working'].mean()*100:.1f}%")
    return df

def clean_household_assets(df, year):
    """
    Clean household asset variables for wealth proxy
    """
    # Electricity (v119)
    if 'v119' in df.columns:
        df['has_electricity'] = pd.to_numeric(df['v119'], errors='coerce')
        df['has_electricity'] = df['has_electricity'].map({0: 0, 1: 1})
    else:
        df['has_electricity'] = np.nan
    
    # Water source (v113)
    # Improved water: piped, borehole, protected well (codes 11-19, 21, 31, 41, 51)
    if 'v113' in df.columns:
        df['water_improved'] = df['v113'].isin(range(11, 20)).astype(int)
    else:
        df['water_improved'] = np.nan
    
    # Toilet facility (v116)
    # Improved toilet: flush, VIP latrine, pit latrine with slab (codes 11-15, 21, 22)
    if 'v116' in df.columns:
        df['toilet_improved'] = df['v116'].isin([11, 12, 13, 14, 15, 21, 22]).astype(int)
    else:
        df['toilet_improved'] = np.nan
    
    # Floor material (v127)
    # Finished floor: cement, tiles, wood (codes 30-39)
    if 'v127' in df.columns:
        df['floor_finished'] = df['v127'].between(30, 39).astype(int)
    else:
        df['floor_finished'] = np.nan
    
    # Create simple asset index (sum of assets)
    asset_cols = ['has_electricity', 'water_improved', 'toilet_improved', 'floor_finished']
    df['asset_count'] = df[asset_cols].sum(axis=1)
    
    logging.info(f"{year}: Asset index - Mean: {df['asset_count'].mean():.2f}, Max: {df['asset_count'].max()}")
    return df

def apply_survey_weights(df, year):
    """
    Apply DHS survey weights (v005)
    v005 is the woman's sample weight divided by 1,000,000
    """
    if 'v005' in df.columns:
        df['sample_weight'] = pd.to_numeric(df['v005'], errors='coerce') / 1000000
        df['sample_weight'].fillna(1, inplace=True)
        logging.info(f"{year}: Survey weights applied - Range: {df['sample_weight'].min():.3f} to {df['sample_weight'].max():.3f}")
    else:
        df['sample_weight'] = 1.0
        logging.warning(f"{year}: No survey weights found, using weight=1")
    
    return df

def clean_all_years(all_data):
    """
    Apply all cleaning functions to each year's data
    """
    cleaned_data = {}
    
    for year, data_dict in all_data.items():
        logging.info(f"\n{'='*60}")
        logging.info(f"Cleaning {year} data")
        logging.info(f"{'='*60}")
        
        df = data_dict['df'].copy()
        
        # Apply cleaning functions in order
        df = clean_marital_status(df, year)
        df = clean_age_first_marriage(df, year)
        df = clean_education(df, year)
        df = clean_demographics(df, year)
        df = clean_fertility(df, year)
        df = clean_work(df, year)
        df = clean_household_assets(df, year)
        df = apply_survey_weights(df, year)
        
        # Add survey year
        df['survey_year'] = year
        
        cleaned_data[year] = df
        
        # Summary
        logging.info(f"Cleaned {year}: {len(df):,} rows, {len(df.columns)} columns")
    
    return cleaned_data

def select_final_columns(df, year):
    """
    Select only the columns we need for analysis
    """
    final_cols = [
        'survey_year',
        'sample_weight',
        'age_first_marriage',
        'early_marriage',
        'ever_married',
        'marital_status',
        'current_age',
        'education_level',
        'education_level_num',
        'education_years',
        'literate',
        'children_ever_born',
        'children_surviving',
        'children_deceased',
        'ideal_children',
        'currently_working',
        'residence',
        'residence_urban',
        'region',
        'region_code',
        'religion',
        'ethnicity',
        'has_electricity',
        'water_improved',
        'toilet_improved',
        'floor_finished',
        'asset_count',
        'wealth_proxy_score',    # ADD THIS
        'wealth_quintile',       # ADD THIS
        'wealth_quintile_num'    # ADD THIS
    ]
    
    # Only keep columns that exist
    existing_cols = [col for col in final_cols if col in df.columns]
    return df[existing_cols]
def handle_missing_values(df, impute_strategy='balanced'):
    """
    Professional missing data handling for DHS data.
    
    Parameters:
    -----------
    df : pandas DataFrame
        Cleaned but raw DataFrame with missing values
    impute_strategy : str
        'balanced' - impute as described below
        'conservative' - minimal imputation
        'aggressive' - drop missing
    
    Returns:
    --------
    df_clean : pandas DataFrame
        DataFrame with handled missing values
    """
    
    if impute_strategy == 'aggressive':
        # Drop any row with missing in key variables
        key_vars = ['age_first_marriage', 'education_level', 'residence', 'region']
        df_clean = df.dropna(subset=key_vars)
        print(f"Aggressive: Dropped {len(df) - len(df_clean):,} rows with missing values")
        return df_clean
    
    df_clean = df.copy()
    
    # 1. Variables that should remain NaN (never married women)
    # age_first_marriage and early_marriage stay as-is
    # These are valid missing - women who never married
    
    # 2. Handle region - Create 'Unknown' category
    if 'region' in df_clean.columns:
        missing_region_count = df_clean['region'].isna().sum()
        df_clean['region'] = df_clean['region'].fillna('Unknown')
        df_clean['region_code'] = df_clean['region_code'].fillna(99)
        print(f"Region: Filled {missing_region_count:,} missing values as 'Unknown'")
    
    # 3. Recalculate education_years based on education_level (for ALL rows)
    if 'education_years' in df_clean.columns and 'education_level' in df_clean.columns:
        # First, check what education_level values exist
        print(f"  Education levels found: {df_clean['education_level'].unique()}")
        
        # Standard Ethiopian education system
        # Primary: Grades 1-8 (8 years historically, but DHS often uses 6)
        # Secondary: Grades 9-12 (4 years)
        # Higher: 4+ years
        
        # More accurate mapping for Ethiopia
        def map_education_to_years(level):
            if pd.isna(level):
                return np.nan
            elif level == 'no_education':
                return 0
            elif level == 'primary':
                return 6  # Typical primary completion
            elif level == 'secondary':
                return 10  # Primary (6) + Secondary (4)
            elif level == 'higher':
                return 14  # Through university (10 + 4)
            else:
                return np.nan
        
        # Apply mapping to ALL rows (overwrite existing values)
        df_clean['education_years'] = df_clean['education_level'].apply(map_education_to_years)
        print(f"  Education years: Recalculated for all {len(df_clean):,} rows based on education level")
    
    # 4. Impute literate based on education_years (after recalculating)
    if 'literate' in df_clean.columns and 'education_years' in df_clean.columns:
        missing_before = df_clean['literate'].isna().sum()
        
        # If has any education years > 0, they are literate
        mask = (df_clean['education_years'] > 0) & (df_clean['literate'].isna())
        df_clean.loc[mask, 'literate'] = 1
        
        # If no education (0 years), they are likely not literate
        mask = (df_clean['education_years'] == 0) & (df_clean['literate'].isna())
        df_clean.loc[mask, 'literate'] = 0
        
        # For any remaining missing, use mode by region
        mask = df_clean['literate'].isna()
        if mask.any():
            for region in df_clean['region'].unique():
                if pd.notna(region):
                    region_mask = (df_clean['region'] == region) & (df_clean['literate'].notna())
                    if region_mask.any():
                        mode_val = df_clean.loc[region_mask, 'literate'].mode()
                        if len(mode_val) > 0:
                            fill_mask = mask & (df_clean['region'] == region)
                            df_clean.loc[fill_mask, 'literate'] = mode_val.iloc[0]
        
        # Global fallback
        if df_clean['literate'].isna().any():
            global_mode = df_clean['literate'].mode()[0]
            df_clean['literate'] = df_clean['literate'].fillna(global_mode)
        
        missing_after = df_clean['literate'].isna().sum()
        print(f"  Literacy: Imputed {missing_before - missing_after:,} values, {missing_after} remain")
    
    # 5. Impute ideal_children using median by education_level and residence
    if 'ideal_children' in df_clean.columns:
        missing_before = df_clean['ideal_children'].isna().sum()
        
        # Group by education and residence for more accurate imputation
        group_median = df_clean.groupby(['education_level', 'residence'])['ideal_children'].transform('median')
        df_clean['ideal_children'] = df_clean['ideal_children'].fillna(group_median)
        
        # Remaining missing use global median
        if df_clean['ideal_children'].isna().any():
            global_median = df_clean['ideal_children'].median()
            df_clean['ideal_children'] = df_clean['ideal_children'].fillna(global_median)
        
        missing_after = df_clean['ideal_children'].isna().sum()
        print(f"Ideal children: Imputed {missing_before - missing_after:,} values")
    
    # 6. Impute has_electricity with mode by region and residence
    if 'has_electricity' in df_clean.columns:
        missing_before = df_clean['has_electricity'].isna().sum()
        
        # Group by region and residence
        group_mode = df_clean.groupby(['region', 'residence'])['has_electricity'].transform(
            lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else 0)
        )
        df_clean['has_electricity'] = df_clean['has_electricity'].fillna(group_mode)
        
        # Global fallback
        if df_clean['has_electricity'].isna().any():
            global_mode = df_clean['has_electricity'].mode()[0]
            df_clean['has_electricity'] = df_clean['has_electricity'].fillna(global_mode)
        
        missing_after = df_clean['has_electricity'].isna().sum()
        print(f"Electricity: Imputed {missing_before - missing_after:,} values")
    
    # 7. Impute religion using mode by region
    if 'religion' in df_clean.columns:
        missing_before = df_clean['religion'].isna().sum()
        
        # Fill with mode by region
        for region in df_clean['region'].unique():
            if pd.notna(region):
                region_mask = (df_clean['region'] == region) & (df_clean['religion'].notna())
                if region_mask.any():
                    mode_val = df_clean.loc[region_mask, 'religion'].mode()
                    if len(mode_val) > 0:
                        fill_mask = (df_clean['region'] == region) & (df_clean['religion'].isna())
                        df_clean.loc[fill_mask, 'religion'] = mode_val.iloc[0]
        
        # Global fallback
        if df_clean['religion'].isna().any():
            global_mode = df_clean['religion'].mode()[0]
            df_clean['religion'] = df_clean['religion'].fillna(global_mode)
        
        missing_after = df_clean['religion'].isna().sum()
        print(f"Religion: Imputed {missing_before - missing_after:,} values")
    
    # 8. Impute currently_working with mode by education and region
    if 'currently_working' in df_clean.columns:
        missing_before = df_clean['currently_working'].isna().sum()
        
        group_mode = df_clean.groupby(['education_level', 'region'])['currently_working'].transform(
            lambda x: x.fillna(x.mode()[0] if len(x.mode()) > 0 else 0)
        )
        df_clean['currently_working'] = df_clean['currently_working'].fillna(group_mode)
        
        # Global fallback
        if df_clean['currently_working'].isna().any():
            global_mode = df_clean['currently_working'].mode()[0]
            df_clean['currently_working'] = df_clean['currently_working'].fillna(global_mode)
        
        missing_after = df_clean['currently_working'].isna().sum()
        print(f"Working status: Imputed {missing_before - missing_after:,} values")
    
    # 9. Recalculate asset_count if has_electricity was imputed
    if 'has_electricity' in df_clean.columns and 'asset_count' in df_clean.columns:
        asset_cols = ['has_electricity', 'water_improved', 'toilet_improved', 'floor_finished']
        available_cols = [col for col in asset_cols if col in df_clean.columns]
        if available_cols:
            df_clean['asset_count'] = df_clean[available_cols].sum(axis=1)
            print(f"Asset count: Recalculated after imputation")
    
    return df_clean


def create_analysis_ready_datasets(master_df):
    """
    Create three versions of the dataset for different analytical purposes.
    
    Returns:
    --------
    datasets : dict
        'descriptive' - Keep all missing (for summary statistics)
        'early_marriage' - Ever-married only (for early marriage analysis)
        'modeling' - Imputed missing values (for ML models)
        'complete_cases' - No missing in key variables (for sensitivity analysis)
    """
    
    datasets = {}
    
    # Version 1: Descriptive statistics (keep all missing as-is)
    datasets['descriptive'] = master_df.copy()
    print(f"Descriptive dataset: {len(datasets['descriptive']):,} rows (missing preserved)")
    
    # Version 2: Early marriage analysis (ever-married women only)
    datasets['early_marriage_analysis'] = master_df[master_df['ever_married'] == 1].copy()
    print(f"Early marriage analysis: {len(datasets['early_marriage_analysis']):,} rows (ever-married only)")
    
    # Version 3: Modeling dataset (imputed missing values)
    datasets['modeling'] = handle_missing_values(master_df.copy(), impute_strategy='balanced')
    print(f"Modeling dataset: {len(datasets['modeling']):,} rows (missing imputed)")
    
    # Version 4: Complete cases (for sensitivity analysis)
    key_vars = ['age_first_marriage', 'education_level', 'residence', 'region', 'wealth_quintile']
    datasets['complete_cases'] = master_df.dropna(subset=key_vars)
    print(f"Complete cases: {len(datasets['complete_cases']):,} rows (no missing in key vars)")
    
    return datasets
if __name__ == "__main__":
    print("Data cleaner module loaded successfully")
    print("Functions available: clean_marital_status, clean_age_first_marriage, clean_education, etc.")