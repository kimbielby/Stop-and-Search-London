import logging
import os
import pandas as pd
import geopandas as gpd
from config.config import DATA_INTERIM, DATA_RAW

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def aggregate_ss(london_lsoas):
    """
    Aggregate S&S stops to LSOA level.
    Produces:
      - ss_count: total stops per LSOA (main outcome variable)
      - ss_count_s60: S60 stops per LSOA (for robustness check)
      - ss_count_pace: PACE stops per LSOA (for robustness check)
      - ss_rate_per1000: stops per 1,000 residents (secondary outcome variable)
    Usual residents loaded from Census 2021 TS001 for rate calculation.
    """
    logger.info("Aggregating S&S stops to LSOA level...")

    ss_df = pd.read_csv(DATA_INTERIM / "sands" / "ss_2025_london.csv")
    logger.info(f"S&S stops loaded: {len(ss_df)}")

    # Total stop count per LSOA
    ss_count = (ss_df.groupby("LSOA21CD")
                .size()
                .reset_index()
                .rename(columns={0: "ss_count"}))

    # S60 count per LSOA
    ss_s60 = (ss_df[ss_df["is_s60"]]
              .groupby("LSOA21CD")
              .size()
              .reset_index()
              .rename(columns={0: "ss_count_s60"}))

    # PACE count per LSOA (non-S60)
    ss_pace = (ss_df[~ss_df["is_s60"]]
               .groupby("LSOA21CD")
               .size()
               .reset_index()
               .rename(columns={0: "ss_count_pace"}))

    # Usual residents for rate calculation
    ts001 = pd.read_csv(
        DATA_RAW / "census_2021" / "census2021-ts001-lsoa.csv"
    )[["geography code",
       "Residence type: Total; measures: Value"]].rename(columns={
        "geography code": "LSOA21CD",
        "Residence type: Total; measures: Value": "usual_residents"
        })

    # Merge all onto full LSOA list, fill zeros for LSOAs with no stops
    ss_agg = (london_lsoas[["LSOA21CD"]]
              .merge(ss_count, on="LSOA21CD", how="left")
              .merge(ss_s60, on="LSOA21CD", how="left")
              .merge(ss_pace, on="LSOA21CD", how="left")
              .merge(ts001[["LSOA21CD", "usual_residents"]], on="LSOA21CD", how="left")
              .fillna({"ss_count": 0, "ss_count_s60": 0, "ss_count_pace": 0}))

    for col in ["ss_count", "ss_count_s60", "ss_count_pace"]:
        ss_agg[col] = ss_agg[col].astype(int)

    # Rate per 1,000 residents
    ss_agg["ss_rate_per1000"] = (
            ss_agg["ss_count"] / ss_agg["usual_residents"] * 1000
    )

    logger.info(f"LSOAs with at least one stop: "
                f"{(ss_agg['ss_count'] > 0).sum()}")
    logger.info(f"LSOAs with zero stops: "
                f"{(ss_agg['ss_count'] == 0).sum()}")
    logger.info(f"S&S count summary:\n"
                f"{ss_agg['ss_count'].describe().to_string()}")

    return ss_agg.drop(columns="usual_residents")


def load_controls():
    """Load all six control variable files from interim/controls/."""
    controls_path = DATA_INTERIM / "controls"

    drug_offences = pd.read_csv(controls_path / "drug_offences_2024.csv")
    imd = pd.read_csv(controls_path / "imd_2025.csv")
    ethnicity = pd.read_csv(controls_path / "ethnic_composition_2021.csv")
    tfl = pd.read_csv(controls_path / "tfl_distances_2025.csv")
    avg_price = pd.read_csv(controls_path / "avg_property_value.csv")
    pop_density = pd.read_csv(controls_path / "pop_density_2021.csv")

    logger.info("Control variable files loaded")
    for name, df in [("drug_offences", drug_offences), ("imd", imd),
                     ("ethnicity", ethnicity), ("tfl", tfl),
                     ("avg_price", avg_price), ("pop_density", pop_density)]:
        logger.info(f"  {name}: {len(df)} rows")

    return drug_offences, imd, ethnicity, tfl, avg_price, pop_density


def main():
    # 1. Load London LSOAs with borough
    london_lsoas = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    )[["LSOA21CD", "LAD22NM"]]
    logger.info(f"London LSOAs: {len(london_lsoas)}")

    # 2. Aggregate S&S outcome variables
    ss_agg = aggregate_ss(london_lsoas)

    # 3. Load inequality measure
    gini = pd.read_csv(
        DATA_INTERIM / "inequality" / "gini_housing_2022_2024.csv"
    )
    logger.info(f"Gini values loaded: {gini['gini_housing'].notna().sum()} "
                f"non-null of {len(gini)}")

    # 4. Load control variables
    drug_offences, imd, ethnicity, tfl, avg_price, pop_density = load_controls()

    # 5. Merge everything onto full LSOA list
    logger.info("Merging all variables...")

    analytical = (london_lsoas
                  .merge(ss_agg, on="LSOA21CD", how="left")
                  .merge(gini, on="LSOA21CD", how="left")
                  .merge(drug_offences, on="LSOA21CD", how="left")
                  .merge(imd, on="LSOA21CD", how="left")
                  .merge(ethnicity, on="LSOA21CD", how="left")
                  .merge(tfl, on="LSOA21CD", how="left")
                  .merge(avg_price, on="LSOA21CD", how="left")
                  .merge(pop_density, on="LSOA21CD", how="left"))

    logger.info(f"Analytical dataset shape: {analytical.shape}")

    # 6. Document missing values
    logger.info("Missing value summary:")
    null_counts = analytical.isna().sum()
    for col, n in null_counts[null_counts > 0].items():
        logger.info(f"  {col}: {n} null ({n / len(analytical) * 100:.1f}%)")

    # 7. Log regression sample size
    #    Complete cases across all variables needed for main model
    main_model_vars = [
        "ss_count", "gini_housing", "drug_rate_2024",
        "income_score", "imd_crime_score", "pct_non_white",
        "mean_dist_to_tfl_m", "mean_price", "pop_density"
    ]
    complete_cases = analytical[main_model_vars].notna().all(axis=1).sum()
    logger.info(f"Complete cases for main model: {complete_cases} "
                f"of {len(analytical)} LSOAs "
                f"({complete_cases / len(analytical) * 100:.1f}%)")

    # 8. Save
    os.makedirs(DATA_INTERIM, exist_ok=True)
    output_path = DATA_INTERIM / "analytical_dataset.csv"
    analytical.to_csv(output_path, index=False)
    logger.info(f"Saved analytical dataset to {output_path}")


if __name__ == "__main__":
    main()
