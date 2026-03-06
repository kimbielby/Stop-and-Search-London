# Stop and Search London — Spatial Replication Study

A replication and update of Suss & Oliveira (2023), *Economic Inequality and the Spatial Distribution of Stop and Search: Evidence from London*, using 2025 data.

The original paper found that housing value economic inequality — measured by the Gini coefficient of property prices within Lower Layer Super Output Areas (LSOAs) — is positively associated with stop and search (S&S) incidence in London, even after controlling for prior crime rates, ethnic composition, population density, and deprivation. This project replicates that analysis using updated data sources and extends it by distinguishing between searches conducted under PACE (requiring reasonable suspicion) and Section 60 (suspicion-less).

---

## Key findings

- The central finding replicates: a one standard deviation increase in economic inequality is associated with a **15.3% increase** in expected stops (IRR = 1.153, p < 0.001), compared to 33% in the original paper
- Spatial autocorrelation in OLS residuals (Moran's I = 0.219, p < 0.0001) is eliminated by the Spatial Durbin Model (Moran's I = −0.005, p = 0.311)
- Results are robust to the exclusion of Westminster and to a rate-based outcome measure
- The inequality effect is driven entirely by **PACE searches** (IRR = 1.154). S60 searches show no association with neighbourhood economic inequality, consistent with prior research suggesting S60 spatial concentration is driven by the racial and ethnic composition of targeted areas rather than economic characteristics

---

## Data sources

All raw data was downloaded manually. See `pipeline/01_download_raw.py` for full provenance, URLs, and download instructions for each source.

| Source | Coverage | Location |
|---|---|---|
| Met Police / City of London S&S | Calendar year 2025 | `data/raw/met_police/sands/` |
| Met Police / City of London street crime | Calendar year 2024 | `data/raw/met_police/street/` |
| Land Registry price paid | 2022–2024 | `data/raw/land_registry/` |
| LSOA 2021 boundaries | December 2021 | `data/raw/boundaries/` |
| LSOA 2011→2021 lookup | — | `data/raw/boundaries/` |
| ONS Postcode Directory (ONSPD) | February 2024 | `data/raw/boundaries/` |
| Census 2021 TS001 usual residents | LSOA level | `data/raw/census_2021/` |
| Census 2021 TS021 ethnic group | LSOA level | `data/raw/census_2021/` |
| Census 2021 WP001 workplace population | LSOA level | `data/raw/census_2021/` |
| IMD 2025 (File 7) | LSOA level | `data/raw/indices_deprivation/` |
| NaPTAN station locations | Great Britain | `data/raw/tfl/` |

---

## Project structure

```
Stop_and_Search_London/
│
├── config/
│   └── config.py               # Paths, CRS constants, and project settings
│
├── data/
│   ├── raw/                    # Downloaded source files (not version controlled)
│   └── interim/                # Cleaned and intermediate outputs
│
├── notebooks/
│   ├── utils_notebooks/
│   │   ├── get_land_reg_data.ipynb     # Adds headers to Land Registry CSV files
│   │   └── build_lsoa_boundaries.ipynb # Interactive version of pipeline 03
│   ├── 01_data_exploration.ipynb
│   ├── 02_inequality_measure.ipynb
│   ├── 03_spatial_diagnostics.ipynb
│   ├── 04_control_variables.ipynb
│   ├── 05_modelling.ipynb
│   └── 06_results_visualisation.ipynb  # Full results write-up
│
├── outputs/
│   ├── figures/                # Choropleth maps and model figures (Figures 1–4)
│   └── tables/                 # Model results tables (Tables 1–3, A.2, A.3, S60/PACE)
│
├── pipeline/
│   ├── 01_download_raw.py      # Raw data provenance documentation
│   ├── 02_clean_ss_data.py     # Clean and geocode S&S records
│   ├── 03_build_lsoa_boundaries.py   # Build London LSOA boundary file
│   ├── 04_build_inequality_measure.py        # Compute LSOA-level Gini coefficients
│   ├── 05_build_controls.py    # Build control variables
│   ├── 06_merge_analysis_dataset.py  # Merge all variables into analytical dataset
│   ├── 07_spatial_diagnostics.py     # Produce Figures 1 and 2
│   └── 08_models.py            # OLS, SDM, negative binomial, robustness checks
│
├── tests/
│   ├── test_gini.py            # Unit tests for Gini coefficient calculation
│   └── test_spatial.py         # Unit tests for spatial weights and Moran's I
│
├── utils/
│   └── gini.py                 # Gini coefficient function
│
├── conftest.py                 # Adds project root to sys.path for pytest
└── pyproject.toml              # Project metadata and pytest configuration
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/kimbielby/Stop-and-Search-London.git
cd Stop-and-Search-London
```

### 2. Create and activate the conda environment

```bash
conda create -n stopsearch python=3.12
conda activate stopsearch
```

### 3. Install dependencies

```bash
pip install .
```

This reads all dependencies from `pyproject.toml` and installs them.

### 4. Add the project root to your path

```bash
conda env config vars set PYTHONPATH=/path/to/Stop-and-Search-London
conda activate stopsearch
```

### 5. Download raw data

Follow the instructions in `pipeline/01_download_raw.py` for each data source. For Land Registry files, run `notebooks/utils_notebooks/get_land_reg_data.ipynb` after downloading to add column headers.

---

## Running the pipeline

Run pipelines in order from the project root. Pipelines 07 and 08 require interim outputs from all preceding pipelines.

```bash
python pipeline/02_clean_ss_data.py
python pipeline/03_build_lsoa_boundaries.py
python pipeline/04_build_inequality_measure.py
python pipeline/05_build_controls.py
python pipeline/06_merge_analysis_dataset.py
python pipeline/07_spatial_diagnostics.py
python pipeline/08_models.py
```

Pipeline 08 takes several minutes due to the Spatial Durbin Model and Westminster robustness check.

---

## Running the tests

```bash
pytest -v
```

30 tests across two modules covering the Gini coefficient calculation and spatial weights construction.

---

## Methodological deviations from the original paper

| Aspect | Original paper | This replication |
|---|---|---|
| S&S data | 2019 | 2025 |
| Inequality measure | Zoopla estimated values (Sep 2019) | Land Registry transactions (2022–2024) |
| Gini threshold | 50 observations | 30 transactions |
| Census | 2011 | 2021 |
| IMD | 2019 | 2025 |
| TfL distance | LSOA centroids | Actual stop locations |
| Drug rate denominator | 2011 workplace population | 2021 workplace population |
| LSOA boundaries | 2011 | 2021 |
| S60/PACE distinction | Not available | Available and analysed |

---

## Reference

Suss, J. & Oliveira, T. (2023). Economic Inequality and the Spatial Distribution of Stop and Search: Evidence from London. *The British Journal of Criminology*, 63(1), 1–19. https://doi.org/10.1093/bjc/azac003