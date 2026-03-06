import logging
import os
import geopandas as gpd
import numpy as np
import pandas as pd
from config.config import DATA_INTERIM, DATA_RAW
from utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Minimum number of transactions required to calculate a reliable Gini estimate.
# Justified by bootstrap stability simulation in notebooks/02_inequality_measure.ipynb.
# Sensitivity analysis at n=50 (the original paper's threshold) is performed in
# the modelling notebook to confirm findings are not threshold-dependent.
MIN_TRANSACTIONS = 30


def main():
    # 1. Load and concatenate Land Registry Price Paid CSVs (2022-2024)
    lr_files = sorted((DATA_RAW / "land_registry").glob("*.csv"))
    logger.info(f"Land Registry files found: {[f.name for f in lr_files]}")

    lr_df = pd.concat([pd.read_csv(f) for f in lr_files], ignore_index=True)
    logger.info(f"Total rows after concatenation: {len(lr_df)}")
    logger.info(f"Date range: {lr_df['date_of_transfer'].min()} to "
                f"{lr_df['date_of_transfer'].max()}")

    # 2. Select required columns
    lr_df = lr_df[[
        "transaction_unique_identifier", "price", "date_of_transfer",
        "postcode", "property_type", "ppd_cat_type"
    ]]

    # 3. Non-geographic filtering:
    #    - Residential property types only (D=detached, S=semi, T=terraced, F=flat)
    #    - Category A transactions only (standard market sales)
    #    - Drop null postcodes
    lr_clean = lr_df[
        (lr_df["property_type"].isin(["D", "S", "T", "F"])) &
        (lr_df["ppd_cat_type"] == "A") &
        (lr_df["postcode"].notna())
        ].copy()
    logger.info(f"Rows after non-geographic filtering: {len(lr_clean)}")
    logger.info(f"Rows removed: {len(lr_df) - len(lr_clean)}")

    # 4. Standardise postcodes to match ONS Postcode Directory pcds format
    #    (strip whitespace, remove spaces, uppercase, reinsert space before last 3 chars)
    lr_clean["postcode"] = (lr_clean["postcode"]
                            .str.strip()
                            .str.replace(" ", "", regex=False)
                            .str.upper()
                            .apply(lambda x: f"{x[:-3]} {x[-3:]}" if pd.notna(x) else x))

    # Validate postcodes against UK format
    postcode_pattern = r'^[A-Z]{1,2}[0-9][0-9A-Z]?\s[0-9][A-Z]{2}$'
    valid_mask = lr_clean["postcode"].str.match(postcode_pattern, na=False)
    invalid_count = (~valid_mask).sum()
    if invalid_count > 0:
        logger.warning(f"Invalid postcodes after standardisation: {invalid_count} "
                       f"({invalid_count / len(lr_clean) * 100:.2f}%)")
    else:
        logger.info("All postcodes valid after standardisation")

    # 5. Join to ONS Postcode Directory to assign LSOA 2021 codes
    logger.info("Loading ONS Postcode Directory...")
    pcd = pd.read_csv(
        DATA_RAW / "boundaries" / "ONSPD_FEB_2024_UK.csv",
        usecols=["pcds", "lsoa21"],
        encoding="latin-1",
        low_memory=False
    )

    lr_lsoa = lr_clean.merge(pcd, left_on="postcode", right_on="pcds", how="left")
    logger.info(f"Rows before join: {len(lr_clean)}")
    logger.info(f"Rows after join: {len(lr_lsoa)}")

    null_lsoa = lr_lsoa["lsoa21"].isna().sum()
    logger.info(f"Unmatched postcodes (null lsoa21): {null_lsoa} "
                f"({null_lsoa / len(lr_lsoa) * 100:.2f}%)")

    lr_lsoa = lr_lsoa[lr_lsoa["lsoa21"].notna()].copy()
    logger.info(f"Rows after dropping unmatched: {len(lr_lsoa)}")

    # 6. Filter to London LSOAs
    london_lsoas = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    )[["LSOA21CD"]]

    lr_london = lr_lsoa[lr_lsoa["lsoa21"].isin(london_lsoas["LSOA21CD"])].copy()
    logger.info(f"Rows after filtering to London: {len(lr_london)}")
    logger.info(f"Unique London LSOAs with transactions: {lr_london['lsoa21'].nunique()} "
                f"of {len(london_lsoas)} total")

    # 7. Calculate Gini coefficient per LSOA
    #    LSOAs below MIN_TRANSACTIONS are excluded (assigned NaN)
    transactions_per_lsoa = lr_london.groupby("lsoa21").size()
    below_threshold = (transactions_per_lsoa < MIN_TRANSACTIONS).sum()
    logger.info(f"LSOAs below {MIN_TRANSACTIONS} transaction threshold "
                f"(excluded): {below_threshold}")
    logger.info(f"LSOAs above threshold (included): "
                f"{(transactions_per_lsoa >= MIN_TRANSACTIONS).sum()}")

    gini_by_lsoa = (lr_london.groupby("lsoa21")["price"]
                    .apply(lambda x: gini(x) if len(x) >= MIN_TRANSACTIONS else np.nan)
                    .reset_index()
                    .rename(columns={"lsoa21": "LSOA21CD", "price": "gini_housing"}))

    # Merge with full LSOA list to ensure all 4,994 LSOAs are represented
    # LSOAs with no transactions will have NaN gini_housing
    gini_by_lsoa = (london_lsoas[["LSOA21CD"]]
                    .merge(gini_by_lsoa, on="LSOA21CD", how="left"))

    logger.info(f"LSOAs with Gini calculated: "
                f"{gini_by_lsoa['gini_housing'].notna().sum()}")
    logger.info(f"Gini summary:\n"
                f"{gini_by_lsoa['gini_housing'].describe().to_string()}")

    # 8. Save to interim
    os.makedirs(DATA_INTERIM / "inequality", exist_ok=True)

    output_path = DATA_INTERIM / "inequality" / "gini_housing_2022_2024.csv"
    gini_by_lsoa.to_csv(output_path, index=False)
    logger.info(f"Saved {gini_by_lsoa['gini_housing'].notna().sum()} LSOA Gini "
                f"values to {output_path}")


if __name__ == "__main__":
    main()
