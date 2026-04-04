# config.py - Project configuration 
 
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
 
# DHS survey years 
SURVEY_YEARS = [2000, 2005, 2011, 2016] 
