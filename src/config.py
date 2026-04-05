"""
config.py - Project configuration for Ethiopia Early Marriage DHS Study
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Data directories
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "interim"

# Reports directories
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# DHS survey years and file mapping
SURVEY_YEARS = [2000, 2005, 2011, 2016]

DHS_FILE_MAPPING = {
    2000: "ETIR41FL.DTA",
    2005: "ETIR51FL.DTA",
    2011: "ETIR61FL.DTA",
    2016: "ETIR71FL.DTA"
}

# Key DHS variables for early marriage study
# v prefix = woman-level variables (Individual Recode)
KEY_VARIABLES = {
    # Outcome variables (early marriage)
    'age_first_marriage': 'v511',        # Age at first marriage (in years)
    'marriage_cohort': 'v512',            # Year of first marriage
    'current_marital_status': 'v501',     # Current marital status
    
    # Consequences variables
    'children_ever_born': 'v201',         # Total children ever born
    'children_surviving': 'v202',         # Children surviving
    'current_contraceptive': 'v312',      # Current contraceptive method
    'ideal_number_children': 'v613',      # Ideal number of children
    'education_level': 'v106',            # Highest education level
    'education_years': 'v107',            # Years of education
    'literacy': 'v155',                   # Literacy
    'current_work': 'v714',               # Currently working
    'wealth_index': 'v190',               # Wealth quintile
    'media_exposure': 'v157',             # Reads newspaper/magazine
    
    # Demographic variables
    'current_age': 'v012',                # Current age in years
    'region': 'v024',                     # Region
    'residence': 'v025',                  # Urban/rural
    'religion': 'v130',                   # Religion
    'ethnicity': 'v131',                  # Ethnic group
    
    # Household variables
    'household_head_sex': 'v151',         # Sex of household head
    'water_source': 'v113',               # Source of drinking water
    'toilet_facility': 'v116',            # Type of toilet facility
    'electricity': 'v119',                # Has electricity
}

# Early marriage definition (marriage before age 18)
EARLY_MARRIAGE_THRESHOLD = 18

# Random seed for reproducibility
RANDOM_SEED = 42