import logging
import os
import geopandas as gpd
import pandas as pd
from config.config import DATA_RAW, DATA_INTERIM, CRS_BNG, LAD_CODE_LONDON

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # Load boundary file
    logger.info("Loading LSOA boundary GeoPackage...")
    gdf = gpd.read_file(
        DATA_RAW / "boundaries" /
        "Lower_layer_Super_Output_Areas_December_2021_Boundaries_EW_BFC_V10_-672099234420024429.gpkg"
    )

    # Reproject to BNG if needed
    if gdf.crs.to_epsg() != CRS_BNG:
        logger.info(f"Reprojecting from {gdf.crs.to_epsg()} to {CRS_BNG}...")
        gdf = gdf.to_crs(CRS_BNG)

    # Load lookup and merge LAD codes
    logger.info("Merging LAD codes from lookup table...")
    lookup = pd.read_csv(
        DATA_RAW / "boundaries" /
        "LSOA_(2011)_to_LSOA_(2021)_to_Local_Authority_District_(2022)_Exact_Fit_Lookup_for_EW_(V3).csv"
    )
    gdf = gdf.merge(
        lookup[["LSOA21CD", "LAD22CD", "LAD22NM", "CHGIND"]],
        on="LSOA21CD",
        how="left"
    )

    # Validate merge
    null_lad = gdf["LAD22CD"].isna().sum()
    if null_lad > 0:
        logger.warning(f"{null_lad} LSOAs have no LAD code after merge")

    # Deduplicate — the lookup table contains multiple rows for some LSOA21CDs
    # due to boundary changes, resulting in duplicate geometries. Keep first occurrence.
    duplicates = gdf["LSOA21CD"].duplicated().sum()
    if duplicates > 0:
        logger.warning(f"{duplicates} duplicate LSOA21CD rows found after merge — "
                       f"deduplicating by keeping first occurrence")
        gdf = gdf.drop_duplicates(subset="LSOA21CD").copy()
    logger.info(f"Unique LSOAs after deduplication: {gdf['LSOA21CD'].nunique()}")

    # Filter to London
    london_gdf = gdf[gdf["LAD22CD"].str.startswith(LAD_CODE_LONDON)].copy()
    logger.info(f"London LSOAs: {len(london_gdf)}")

    # Save to interim
    os.makedirs(DATA_INTERIM / "boundaries", exist_ok=True)
    out_path = DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg"
    london_gdf.to_file(out_path, layer="london_lsoa_2021", driver="GPKG")
    logger.info(f"Saved to {out_path}")

if __name__ == "__main__":
    main()
