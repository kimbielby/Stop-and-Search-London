import logging
import os
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib.gridspec import GridSpec

from config.config import DATA_INTERIM, OUTPUTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data():
    """Load LSOA boundaries merged with analytical dataset."""
    london_gdf = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    )[["LSOA21CD", "geometry"]]

    analytical = pd.read_csv(DATA_INTERIM / "analytical_dataset.csv")

    london_gdf = london_gdf.merge(
        analytical[["LSOA21CD", "ss_count", "gini_housing"]],
        on="LSOA21CD",
        how="left"
    )

    logger.info(f"LSOAs loaded: {len(london_gdf)}")
    logger.info(f"Null ss_count: {london_gdf['ss_count'].isna().sum()}")
    logger.info(f"Null gini_housing: {london_gdf['gini_housing'].isna().sum()}")

    return london_gdf


def classify_ss(london_gdf):
    """
    Classify S&S count into paper's break categories and assign colours.
    Breaks: 0, 1-10, 11-25, 26-99, 100-199, 200+
    Colour scheme: YlGnBu sequential palette.
    """
    bins = [-1, 0, 10, 25, 99, 199, london_gdf["ss_count"].max()]
    labels = ["0", "1–10", "11–25", "26–99", "100–199", "200+"]
    palette = ["#ffffcc", "#a1dab4", "#41b6c4", "#2c7fb8", "#253494", "#0a0a2a"]

    london_gdf["ss_class"] = pd.cut(
        london_gdf["ss_count"], bins=bins, labels=labels
    )
    color_map = dict(zip(labels, palette))
    london_gdf["ss_colour"] = london_gdf["ss_class"].map(color_map)

    return london_gdf, labels, palette


def plot_fig1(london_gdf, labels, palette, output_path):
    """Replicate Figure 1: S&S count choropleth."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    london_gdf.plot(
        ax=ax,
        color=london_gdf["ss_colour"].fillna("#cccccc"),
        linewidth=0.1,
        edgecolor="white"
    )

    legend_patches = [
        mpatches.Patch(color=palette[i], label=labels[i])
        for i in range(len(labels))
    ]
    ax.legend(
        handles=legend_patches,
        title="Stops",
        loc="lower right",
        frameon=True,
        fontsize=10,
        title_fontsize=11
    )
    ax.set_axis_off()
    ax.set_title(
        "Number of Stop and Searches — London, 2025, LSOA level",
        fontsize=13,
        pad=12
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path.name}")


def plot_fig2(london_gdf, output_path):
    """Replicate Figure 2: Gini coefficient choropleth."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    london_gdf.plot(
        ax=ax,
        color="#cccccc",
        linewidth=0.1,
        edgecolor="white"
    )

    london_gdf[london_gdf["gini_housing"].notna()].plot(
        ax=ax,
        column="gini_housing",
        cmap="RdPu",
        linewidth=0.1,
        edgecolor="white",
        vmin=london_gdf["gini_housing"].quantile(0.01),
        vmax=london_gdf["gini_housing"].quantile(0.99),
        legend=True,
        legend_kwds={
            "label": "Gini coefficient",
            "orientation": "vertical",
            "shrink": 0.5,
            "pad": 0.02
        }
    )

    ax.set_axis_off()
    ax.set_title(
        "Housing Value Inequality — London, 2025, LSOA level",
        fontsize=13,
        pad=12
    )
    ax.annotate(
        "Grey: fewer than 30 transactions (Gini not calculated)",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=8,
        color="#666666"
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path.name}")


def plot_combined(london_gdf, labels, palette, output_path):
    """Combined Figure 1 and Figure 2 side by side."""
    fig = plt.figure(figsize=(22, 10))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 0.05], wspace=0.05)

    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    # Left: S&S count
    london_gdf.plot(
        ax=ax_left,
        color=london_gdf["ss_colour"].fillna("#cccccc"),
        linewidth=0.1,
        edgecolor="white"
    )
    legend_patches = [
        mpatches.Patch(color=palette[i], label=labels[i])
        for i in range(len(labels))
    ]
    ax_left.legend(
        handles=legend_patches,
        title="Stops",
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        bbox_to_anchor=(0.01, 0.01),
        borderaxespad=0
    )
    ax_left.set_axis_off()

    # Right: Gini
    london_gdf.plot(
        ax=ax_right,
        color="#cccccc",
        linewidth=0.1,
        edgecolor="white"
    )

    vmin = london_gdf["gini_housing"].quantile(0.01)
    vmax = london_gdf["gini_housing"].quantile(0.99)

    london_gdf[london_gdf["gini_housing"].notna()].plot(
        ax=ax_right,
        column="gini_housing",
        cmap="RdPu",
        linewidth=0.1,
        edgecolor="white",
        vmin=vmin,
        vmax=vmax,
        legend=False
    )
    ax_right.set_axis_off()
    ax_right.annotate(
        "Grey: fewer than 30 transactions (Gini not calculated)",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=8,
        color="#666666"
    )

    # Colourbar in dedicated narrow axes
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap="RdPu", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax_cbar)
    cbar.set_label("Gini coefficient", fontsize=10)

    # Fixed-position titles aligned at same height
    fig.canvas.draw()
    left_bbox = ax_left.get_position()
    right_bbox = ax_right.get_position()
    title_y = max(left_bbox.y1, right_bbox.y1) + 0.02

    fig.text(
        (left_bbox.x0 + left_bbox.x1) / 2, title_y,
        "Number of Stop and Searches\nLondon, 2025, LSOA level",
        ha="center", va="bottom", fontsize=12
    )
    fig.text(
        (right_bbox.x0 + right_bbox.x1) / 2, title_y,
        "Housing Value Inequality\nLondon, 2025, LSOA level",
        ha="center", va="bottom", fontsize=12
    )
    plt.suptitle(
        "Spatial Distribution of Stop and Search and Housing Inequality"
        " — London, 2025",
        fontsize=14,
        y=title_y + 0.11
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path.name}")


def main():
    os.makedirs(OUTPUTS / "figures", exist_ok=True)

    london_gdf = load_data()
    london_gdf, labels, palette = classify_ss(london_gdf)

    plot_fig1(
        london_gdf, labels, palette,
        OUTPUTS / "figures" / "fig1_ss_count.png"
    )
    plot_fig2(
        london_gdf,
        OUTPUTS / "figures" / "fig2_gini.png"
    )
    plot_combined(
        london_gdf, labels, palette,
        OUTPUTS / "figures" / "fig1_fig2_combined.png"
    )

    logger.info("All spatial diagnostic figures saved")


if __name__ == "__main__":
    main()
