import logging
import os
import geopandas as gpd
import pandas as pd
from shapely import MultiPoint
from shapely.ops import nearest_points
from config.config import CRS_BNG, DATA_INTERIM, DATA_RAW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_london_lsoas():
    """Load deduplicated London LSOA codes and names from interim boundaries."""
    london_lsoas = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    )[["LSOA21CD", "LAD22NM"]]
    logger.info(f"London LSOAs loaded: {len(london_lsoas)}")
    return london_lsoas


def build_drug_offences(london_lsoas):
    """
    Aggregate 2024 drug offences from police.uk street crime data to LSOA level.
    Used as a lagged control variable (paper uses prior year drug offences).
    """
    logger.info("Building drug offences control...")

    street_files = sorted((DATA_RAW / "met_police" / "street").glob("*.csv"))
    logger.info(f"Street crime files found: {len(street_files)}")

    street_df = pd.concat(
        [pd.read_csv(f) for f in street_files], ignore_index=True
    )
    logger.info(f"Total street crime rows: {len(street_df)}")

    # Drop entirely null column and rows with null LSOA code
    street_df = street_df.drop(columns=["Context"])
    null_lsoa = street_df["LSOA code"].isna().sum()
    logger.info(f"Dropping {null_lsoa} rows with null LSOA code "
                f"({null_lsoa / len(street_df) * 100:.2f}%)")
    street_df = street_df[street_df["LSOA code"].notna()].copy()

    # Filter to London and drugs only
    street_df = street_df[
        street_df["LSOA code"].isin(london_lsoas["LSOA21CD"])
    ].copy()
    logger.info(f"Rows after filtering to London: {len(street_df)}")

    drugs_df = street_df[street_df["Crime type"] == "Drugs"].copy()
    logger.info(f"Drug offence records: {len(drugs_df)}")

    # Aggregate to LSOA level, fill zeros for LSOAs with no drug offences
    drug_counts = (drugs_df.groupby("LSOA code")
                   .size()
                   .reset_index()
                   .rename(columns={"LSOA code": "LSOA21CD",
                                    0: "drug_count"}))

    # Standardise by workplace population to match paper's drug offences rate
    wp001_df = pd.read_csv(DATA_RAW / "census_2021" / "WP001_lsoa.csv")
    wp001 = wp001_df[["Lower layer Super Output Areas Code", "Count"]].rename(columns={
        "Lower layer Super Output Areas Code": "LSOA21CD",
        "Count": "workplace_pop"
    })

    # Merge onto full LSOA list — fill zeros BEFORE computing rate
    drug_offences_2024 = (london_lsoas[["LSOA21CD"]]
                          .merge(drug_counts, on="LSOA21CD", how="left")
                          .fillna({"drug_count": 0})
                          .merge(wp001, on="LSOA21CD", how="left"))

    drug_offences_2024["drug_rate_2024"] = (
            drug_offences_2024["drug_count"] / drug_offences_2024["workplace_pop"]
    )

    logger.info(f"Total LSOAs: {len(drug_offences_2024)}")
    logger.info(f"Drug offences summary:\n"
                f"{drug_offences_2024['drug_rate_2024'].describe().to_string()}")

    return drug_offences_2024


def build_imd(london_lsoas):
    """
    Extract IMD 2025 income and crime domain scores per LSOA.
    Uses domain scores rather than overall IMD score, following Suss & Oliveira (2023).
    """
    logger.info("Building IMD control...")

    imd_df = pd.read_csv(
        DATA_RAW / "indices_deprivation" /
        "File_7_IoD2025_All_Ranks_Scores_Deciles_Population_Denominators.csv"
    )

    imd_london = imd_df[[
        "LSOA code (2021)",
        "Income Score (rate)",
        "Crime Score"
    ]].rename(columns={
        "LSOA code (2021)": "LSOA21CD",
        "Income Score (rate)": "income_score",
        "Crime Score": "imd_crime_score"
    })

    imd_london = imd_london[
        imd_london["LSOA21CD"].isin(london_lsoas["LSOA21CD"])
    ].copy()

    logger.info(f"London LSOAs in IMD data: {len(imd_london)}")

    null_counts = imd_london.isna().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values in IMD data:\n{null_counts}")

    return imd_london


def build_ethnic_composition(london_lsoas):
    """
    Calculate percentage non-white residents per LSOA from Census 2021 TS021.
    """
    logger.info("Building ethnic composition control...")

    census_df = pd.read_csv(
        DATA_RAW / "census_2021" / "census2021-ts021-lsoa.csv"
    )

    census_london = census_df[[
        "geography code",
        "Ethnic group: Total: All usual residents",
        "Ethnic group: White"
    ]].rename(columns={
        "geography code": "LSOA21CD",
        "Ethnic group: Total: All usual residents": "total_residents",
        "Ethnic group: White": "white_residents"
    })

    census_london = census_london[
        census_london["LSOA21CD"].isin(london_lsoas["LSOA21CD"])
    ].copy()

    census_london["pct_non_white"] = (
            (census_london["total_residents"] - census_london["white_residents"])
            / census_london["total_residents"] * 100
    )

    logger.info(f"London LSOAs in ethnic composition data: {len(census_london)}")
    logger.info(f"Pct non-white summary:\n"
                f"{census_london['pct_non_white'].describe().to_string()}")

    return census_london[["LSOA21CD", "pct_non_white"]]


def build_tfl_distances(london_lsoas):
    """
    Calculate mean distance from S&S stop locations to nearest TfL station per LSOA.
    Uses actual stop coordinates rather than LSOA centroids — see notebook
    04_control_variables.ipynb for methodological justification.
    Stations sourced from NaPTAN national dataset, filtered to Underground (MET),
    Rail (RLY) and Platform (PLT) stop types within London boundary.
    Tramlink and cable car stops excluded.
    """
    logger.info("Building TfL distances control...")

    # Load and filter NaPTAN stations
    stations_df = pd.read_csv(
        DATA_RAW / "tfl" / "Stops.csv",
        low_memory=False
    )

    stations_df = stations_df[
        (stations_df["StopType"].isin(["MET", "RLY", "PLT"])) &
        (stations_df["Status"] == "active") &
        (~stations_df["CommonName"].str.contains("Tram Stop", na=False)) &
        (~stations_df["CommonName"].str.contains("IFS Cloud", na=False))
        ].copy()

    # Create GeoDataFrame using BNG coordinates
    stations_gdf = gpd.GeoDataFrame(
        stations_df,
        geometry=gpd.points_from_xy(
            stations_df["Easting"], stations_df["Northing"]
        ),
        crs=CRS_BNG
    )

    # Filter to London boundary
    london_boundary = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    ).dissolve()

    stations_london = gpd.sjoin(
        stations_gdf,
        london_boundary[["geometry"]],
        how="inner",
        predicate="within"
    ).drop(columns="index_right")

    # Deduplicate to one point per station using mean Easting/Northing
    stations_unique = (stations_london
                       .groupby("CommonName")[["Easting", "Northing"]]
                       .mean()
                       .reset_index())

    stations_unique = gpd.GeoDataFrame(
        stations_unique,
        geometry=gpd.points_from_xy(
            stations_unique["Easting"],
            stations_unique["Northing"]
        ),
        crs=CRS_BNG
    )
    logger.info(f"Unique TfL stations within London: {len(stations_unique)}")

    # Load S&S data and calculate distance to nearest station
    ss_df = pd.read_csv(DATA_INTERIM / "sands" / "ss_2025_london.csv")

    ss_gdf = gpd.GeoDataFrame(
        ss_df,
        geometry=gpd.points_from_xy(ss_df["Longitude"], ss_df["Latitude"]),
        crs="EPSG:4326"
    ).to_crs(CRS_BNG)

    logger.info(f"Calculating distances for {len(ss_gdf)} S&S stops...")
    stations_multipoint = MultiPoint(stations_unique.geometry.values)

    ss_gdf["dist_to_tfl_m"] = ss_gdf.geometry.apply(
        lambda pt: pt.distance(nearest_points(stations_multipoint, pt)[0])
    )

    logger.info(f"Distance summary (metres):\n"
                f"{ss_gdf['dist_to_tfl_m'].describe().to_string()}")

    # Aggregate to LSOA level as mean distance
    tfl_dist = (london_lsoas[["LSOA21CD"]]
    .merge(
        ss_gdf.groupby("LSOA21CD")["dist_to_tfl_m"]
        .mean()
        .reset_index()
        .rename(columns={"dist_to_tfl_m": "mean_dist_to_tfl_m"}),
        on="LSOA21CD",
        how="left"
    ))

    null_dist = tfl_dist["mean_dist_to_tfl_m"].isna().sum()
    logger.info(f"LSOAs with no S&S stops (null distance): {null_dist}")
    logger.info(f"Mean distance per LSOA summary (metres):\n"
                f"{tfl_dist['mean_dist_to_tfl_m'].describe().to_string()}")

    return tfl_dist


def build_avg_property_value(london_lsoas):
    """
    Calculate mean property transaction price per LSOA from Land Registry 2022-2024.
    Uses the cleaned London transactions saved by pipeline 04.
    """
    logger.info("Building average property value control...")

    lr_london = pd.read_csv(
        DATA_INTERIM / "land_registry" / "land_reg_london_2022_2024.csv"
    )

    avg_price = (london_lsoas[["LSOA21CD"]]
    .merge(
        lr_london.groupby("lsoa21")["price"]
        .mean()
        .reset_index()
        .rename(columns={"lsoa21": "LSOA21CD",
                         "price": "mean_price"}),
        on="LSOA21CD",
        how="left"
    ))

    null_price = avg_price["mean_price"].isna().sum()
    if null_price > 0:
        logger.warning(f"LSOAs with no transactions (null mean price): {null_price}")

    logger.info(f"Mean price summary:\n"
                f"{avg_price['mean_price'].describe().to_string()}")

    return avg_price


def build_pop_density(london_lsoas):
    """
    Calculate population density per LSOA as (usual residents + workplace population)
    divided by area in hectares. Follows Suss & Oliveira (2023) definition.
    Sources: Census 2021 TS001 (usual residents), WP001 (workplace population),
    LSOA area from boundaries GeoPackage.
    """
    logger.info("Building population density control...")

    # Usual residents
    ts001 = pd.read_csv(
        DATA_RAW / "census_2021" / "census2021-ts001-lsoa.csv"
    )[["geography code",
       "Residence type: Total; measures: Value"]].rename(columns={
        "geography code": "LSOA21CD",
        "Residence type: Total; measures: Value": "usual_residents"
    })

    # Workplace population
    wp001 = pd.read_csv(
        DATA_RAW / "census_2021" / "WP001_lsoa.csv"
    )[["Lower layer Super Output Areas Code",
       "Count"]].rename(columns={
        "Lower layer Super Output Areas Code": "LSOA21CD",
        "Count": "workplace_pop"
    })

    # LSOA area in hectares from boundaries
    london_gdf = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    )[["LSOA21CD", "geometry"]]
    london_gdf["area_ha"] = london_gdf.geometry.area / 10000

    # Merge and calculate density
    pop_density = (london_lsoas[["LSOA21CD"]]
                   .merge(ts001, on="LSOA21CD", how="left")
                   .merge(wp001, on="LSOA21CD", how="left")
                   .merge(london_gdf[["LSOA21CD", "area_ha"]],
                          on="LSOA21CD", how="left"))

    pop_density["pop_density"] = (
            (pop_density["usual_residents"] + pop_density["workplace_pop"])
            / pop_density["area_ha"]
    )

    null_counts = pop_density[["usual_residents",
                               "workplace_pop",
                               "pop_density"]].isna().sum()
    if null_counts.sum() > 0:
        logger.warning(f"Null values in population density data:\n{null_counts}")

    logger.info(f"Population density summary (persons per hectare):\n"
                f"{pop_density['pop_density'].describe().to_string()}")

    return pop_density[["LSOA21CD", "pop_density"]]


def main():
    os.makedirs(DATA_INTERIM / "controls", exist_ok=True)

    london_lsoas = load_london_lsoas()

    # 1. Drug offences 2024
    drug_offences = build_drug_offences(london_lsoas)
    drug_offences[["LSOA21CD", "drug_rate_2024"]].to_csv(
        DATA_INTERIM / "controls" / "drug_offences_2024.csv", index=False
    )
    logger.info("Saved drug_offences_2024.csv")

    # 2. IMD 2025 domain scores
    imd = build_imd(london_lsoas)
    imd.to_csv(
        DATA_INTERIM / "controls" / "imd_2025.csv", index=False
    )
    logger.info("Saved imd_2025.csv")

    # 3. Ethnic composition
    ethnic = build_ethnic_composition(london_lsoas)
    ethnic.to_csv(
        DATA_INTERIM / "controls" / "ethnic_composition_2021.csv", index=False
    )
    logger.info("Saved ethnic_composition_2021.csv")

    # 4. TfL distances
    tfl = build_tfl_distances(london_lsoas)
    tfl.to_csv(
        DATA_INTERIM / "controls" / "tfl_distances_2025.csv", index=False
    )
    logger.info("Saved tfl_distances_2025.csv")

    # 5. Average property value
    avg_price = build_avg_property_value(london_lsoas)
    avg_price.to_csv(
        DATA_INTERIM / "controls" / "avg_property_value.csv", index=False
    )
    logger.info("Saved avg_property_value.csv")

    # 6. Population density
    pop_density = build_pop_density(london_lsoas)
    pop_density.to_csv(
        DATA_INTERIM / "controls" / "pop_density_2021.csv", index=False
    )
    logger.info("Saved pop_density_2021.csv")

    logger.info("All control variables built successfully")


if __name__ == "__main__":
    main()

