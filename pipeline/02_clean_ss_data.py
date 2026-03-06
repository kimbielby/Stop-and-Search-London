import os
import logging
import pandas as pd
import geopandas as gpd
from config.config import DATA_RAW, DATA_INTERIM, CRS_WGS84, CRS_BNG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    # 1. Concatenate all 12 monthly Stop and Search files
    ss_files = sorted((DATA_RAW / "met_police" / "sands").glob("*.csv"))
    ss_df = pd.concat([pd.read_csv(f) for f in ss_files], ignore_index=True)
    logger.info(f"Total rows before cleaning: {len(ss_df)}")
    logger.info("Null counts per column: ")
    logger.info(ss_df.isna().sum().to_string())
    logger.info("Columns that are entirely null:")
    logger.info(ss_df.columns[ss_df.isna().all()].tolist())

    # 2. Drop columns
    ss_df = ss_df.drop(columns=["Policing operation", "Part of a policing operation"])

    # 3. Add Section 60 flag before dropping rows
    ss_df["is_s60"] = ss_df["Legislation"].str.contains("section 60", na=False)

    # 4. Drop null coordinates
    null_coords = ss_df["Latitude"].isna() | ss_df["Longitude"].isna()
    logger.info(f"Dropping {null_coords.sum()} rows ({null_coords.sum()/len(ss_df)*100:.2f}%) with null coordinates")
    ss_df = ss_df[~null_coords].copy()

    # 5. Convert to GeoDataFrame and reproject
    ss_gdf = gpd.GeoDataFrame(
        ss_df,
        geometry=gpd.points_from_xy(ss_df["Longitude"], ss_df["Latitude"]),
        crs=CRS_WGS84
    ).to_crs(CRS_BNG)

    assert ss_gdf.crs.to_epsg() == CRS_BNG, f"Expected BNG, got {ss_gdf.crs.to_epsg()}"

    logger.info(f"CRS: {ss_gdf.crs.to_epsg()}")
    logger.info(f"Final shape: {ss_gdf.shape}")

    # 6. Load London LSOAs and spatial join
    london_gdf = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021",
        )[["LSOA21CD", "LAD22NM", "geometry"]]

    ss_gdf = gpd.sjoin(
        ss_gdf,
        london_gdf,
        how="left",
        predicate="within"
    )

    # 7. Check and drop unmatched stops
    unmatched = ss_gdf["LSOA21CD"].isna().sum()
    logger.info(f"Stops not matched to a London LSOA: {unmatched} ({unmatched/len(ss_gdf)*100:.2f}%)")
    ss_gdf = ss_gdf[ss_gdf["LSOA21CD"].notna()].copy()

    logger.info(f"Final rows after spatial join: {len(ss_gdf)}")
    logger.info(f"CRS: {ss_gdf.crs.to_epsg()}")

    # 8. Remove superfluous columns and save as csv file
    ss_df_clean = ss_gdf.drop(columns=["geometry", "index_right"])

    os.makedirs(DATA_INTERIM / "sands", exist_ok=True)

    ss_df_clean.to_csv(
        DATA_INTERIM / "sands" / "ss_2025_london.csv",
        index=False
    )

    logger.info(f"Saved {len(ss_df_clean)} stops to ss_2025_london.csv")


if __name__ == "__main__":
    main()
