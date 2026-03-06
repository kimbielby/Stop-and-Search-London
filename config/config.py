from pathlib import Path

# Anchor to project root from this file's location
# __file__ = config/config.py
# parents[0] = config/
# parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Paths ---
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

# --- CRS constants ---
CRS_BNG = 27700     # British National Grid - use for all distance/area calculations
CRS_WGS84 = 4326    # Geographic - source format for S&S lat/lon and some APIs

# --- ONS geographic code prefixes ---
# Source: ONS Register of Geographic Codes
# https://geoportal.statistics.gov.uk/datasets/register-of-geographic-codes
LAD_CODE_LONDON = "E09"

# --- Data settings ---
SS_YEAR = 2025
LR_START = "2022-01"
LR_END = "2024-12"
LR_PRICE_MIN = 25000            # Minimum transaction value — filters out non-market sales

# --- Geography settings ---
LSOA_VINTAGE = 2021
MIN_TRANSACTIONS = 30       # Minimum Land Registry transactions per LSOA for Gini

# --- Model settings ---
SPATIAL_WEIGHTS = "queen"
N_SIMULATIONS = 1000        # Monte Carlo permutations for Moran's I
