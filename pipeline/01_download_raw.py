"""
pipeline/01_download_raw.py
===========================
This file documents the raw data sources used in this project.
All data was downloaded manually. File locations are relative to data/raw/.

No automated download is implemented. This file serves as a provenance
record for reproducibility.
"""

# =============================================================================
# STOP AND SEARCH DATA
# =============================================================================
#
# Source:   Metropolitan Police and City of London Police / data.police.uk
# URL:      https://data.police.uk/data/
# Files:    data/raw/met_police/sands/*.csv
# Coverage: Metropolitan Police and City of London Police, calendar year 2025
#           (12 monthly files)
# Downloaded: February 2026
# Notes:    On the data.police.uk download page, select the date range
#           (January 2025 to December 2025), select Metropolitan Police
#           and City of London Police, uncheck "Include crime data", and
#           check "Include stop and search data". Download and extract
#           the monthly CSV files into data/raw/met_police/sands/.

# =============================================================================
# STREET CRIME DATA (for drug offences control variable)
# =============================================================================
#
# Source:   Metropolitan Police and City of London Police / data.police.uk
# URL:      https://data.police.uk/data/
# Files:    data/raw/met_police/street/*.csv
# Coverage: Metropolitan Police and City of London Police, calendar year 2024
#           (12 monthly files, all crime categories)
# Downloaded: February 2026
# Notes:    On the data.police.uk download page, select the date range
#           (January 2024 to December 2024), select Metropolitan Police
#           and City of London Police, and leave only "Include crime data"
#           selected. Drug offences are filtered from these files in
#           pipeline 05.

# =============================================================================
# LAND REGISTRY PRICE PAID DATA (for Gini coefficient)
# =============================================================================
#
# Source:   HM Land Registry
# URL:      https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads
# Files:    data/raw/land_registry/pp-2022.csv
#           data/raw/land_registry/pp-2023.csv
#           data/raw/land_registry/pp-2024.csv
# Coverage: England and Wales, calendar years 2022, 2023, and 2024
# Downloaded: 22 February 2026
# Notes:    Download the complete yearly files for 2022, 2023, and 2024.
#           The files do not include column headers. A header definition
#           file is data/raw/land_registry/land_reg_cols.txt and the notebook
#           notebooks/utils_notebooks/get_land_reg_data.ipynb reads each file, adds headers,
#           and saves it back with the same filename. Update the year
#           variable in that notebook when adding a new year's data.
#           Filtering to London and joining to LSOAs is performed in
#           pipeline 04.

# =============================================================================
# LSOA 2021 BOUNDARIES
# =============================================================================
#
# Source:   ONS Open Geography Portal
# URL:      https://www.data.gov.uk/dataset/1a99061d-af4b-4ffb-9577-dbc3f5a6feb9/lower-layer-super-output-areas-december-2021-boundaries-ew-bfc-v10
# Files:    data/raw/boundaries/LSOA_2021_EW_BFC_V10.shp (and sidecar files)
# Coverage: England and Wales, December 2021 boundaries, full clipped
# Downloaded: 22 February 2026
# Notes:    Download the BFC (Full Clipped) version for accurate area
#           calculations. Used in pipelines 03, 07, and 08.

# =============================================================================
# LSOA 2011 TO 2021 LOOKUP
# =============================================================================
#
# Source:   ONS Open Geography Portal
# URL:      https://www.data.gov.uk/dataset/03a52a27-36e7-4f33-a632-83282faea36f/lsoa-2011-to-lsoa-2021-to-local-authority-district-2022-exact-fit-lookup-for-ew-v3
# Files:    data/raw/boundaries/LSOA_(2011)_to_LSOA_(2021)_to_LAD_(2022)_Lookup_for_EW_(V3).csv
# Coverage: England and Wales
# Downloaded: 22 February 2026
# Notes:    Used in pipeline 02 to align S&S stop locations (which may
#           carry 2011 LSOA codes) to 2021 boundaries.

# =============================================================================
# ONS POSTCODE DIRECTORY (ONSPD)
# =============================================================================
#
# Source:   ONS Open Geography Portal
# URL:      https://geoportal.statistics.gov.uk/datasets/ons-postcode-directory-february-2024
# Files:    data/raw/boundaries/ONSPD_FEB_2024_UK.csv
# Coverage: UK, February 2024 edition
# Downloaded: February 2026
# Notes:    Used in pipeline 04 to join Land Registry transactions
#           (which carry postcodes) to LSOA codes.

# =============================================================================
# CENSUS 2021 — TS001 USUAL RESIDENTS
# =============================================================================
#
# Source:   ONS / NOMIS
# URL:      https://www.nomisweb.co.uk/sources/census_2021_bulk
# Files:    data/raw/census_2021/census2021-ts001-lsoa.csv
# Coverage: England and Wales, LSOA level
# Downloaded: February 2026
# Notes:    Table TS001 — Residence type. Used in pipeline 06 to compute
#           the S&S rate per 1,000 usual residents.

# =============================================================================
# CENSUS 2021 — TS021 ETHNIC GROUP
# =============================================================================
#
# Source:   ONS / NOMIS
# URL:      https://www.nomisweb.co.uk/sources/census_2021_bulk
# Files:    data/raw/census_2021/census2021-ts021-lsoa.csv
# Coverage: England and Wales, LSOA level
# Downloaded: February 2026
# Notes:    Table TS021 — Ethnic group (21 categories). Used in
#           pipeline 05 to compute percentage non-white residents.

# =============================================================================
# CENSUS 2021 — WP001 WORKPLACE POPULATION
# =============================================================================
#
# Source:   ONS / NOMIS
# URL:      https://www.nomisweb.co.uk/sources/census_2021_wp
# Files:    data/raw/census_2021/WP001_lsoa.csv
# Coverage: England and Wales, LSOA level
# Downloaded: February 2026
# Notes:    From the NOMIS Census 2021 workplace population page, select
#           dataset WP001 — Workplace population. Used in pipeline 05 as
#           the denominator for the drug offences rate.

# =============================================================================
# INDEX OF MULTIPLE DEPRIVATION 2025 (IMD)
# =============================================================================
#
# Source:   Ministry of Housing, Communities and Local Government (MHCLG)
# URL:      https://www.gov.uk/government/statistics/english-indices-of-deprivation-2025
# Files:    data/raw/indices_deprivation/File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv
# Coverage: England, LSOA level
# Downloaded: February 2026
# Notes:    Download File 7 which contains all domain ranks, scores,
#           deciles, and population denominators in a single file. Used
#           in pipeline 05 for income deprivation score and crime
#           deprivation score. The 2025 edition uses updated methodology
#           compared to the 2019 edition used in the original paper.

# =============================================================================
# TFL / NAPTAN STATION LOCATIONS
# =============================================================================
#
# Source:   Department for Transport — NaPTAN (National Public Transport
#           Access Nodes)
# URL:      https://beta-naptan.dft.gov.uk/Download/National
# Files:    data/raw/tfl/Stops.csv
# Coverage: Great Britain, all public transport access nodes
# Downloaded: February 2026
# Notes:    On the NaPTAN download page, select CSV as the file type
#           and download the full national dataset. The dataset is
#           filtered to London TfL rail and tram stops (London
#           Underground, Overground, Elizabeth line, DLR, and tram)
#           in pipeline 05. Used to compute mean distance from S&S
#           stop locations to the nearest TfL station.

