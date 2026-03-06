import logging
import os
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import spreg
import esda
import libpysal
import statsmodels.api as sm
from scipy import stats
from statsmodels.discrete.discrete_model import NegativeBinomial

from config.config import DATA_INTERIM, OUTPUTS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

X_COLS = [
    "gini_housing_std",
    "log_mean_price_std",
    "log_density_std",
    "income_score_std",
    "imd_crime_score_std",
    "drug_rate_2024_std",
    "pct_non_white_std",
    "mean_dist_to_tfl_m_std"
]

X_NAMES = [
    "Gini",
    "Average property value (log)",
    "Density (log)",
    "Income deprivation",
    "Crime deprivation",
    "Drugs rate",
    "Non-white (%)",
    "TfL station distance"
]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def load_and_prepare(analytical, london_gdf):
    """
    Filter to complete cases, apply variable transformations, standardise,
    create borough dummies, and build spatial weights matrix.
    """
    model_vars = [
        "LSOA21CD", "LAD22NM", "ss_count", "ss_count_s60", "ss_count_pace",
        "ss_rate_per1000", "gini_housing", "drug_rate_2024", "income_score",
        "imd_crime_score", "pct_non_white", "mean_dist_to_tfl_m",
        "mean_price", "pop_density"
    ]

    reg_df = analytical[model_vars].dropna().copy()
    logger.info(f"Regression sample: {len(reg_df)} LSOAs "
                f"({len(reg_df)/len(analytical)*100:.1f}% of full dataset)")

    # Log transforms
    reg_df["log_density"] = np.log(reg_df["pop_density"])
    reg_df["log_mean_price"] = np.log(reg_df["mean_price"])
    reg_df["log_ss_rate"] = np.log(reg_df["ss_rate_per1000"].clip(lower=0.001))

    # Standardise
    to_standardise = [
        "gini_housing", "log_mean_price", "log_density",
        "income_score", "imd_crime_score", "drug_rate_2024",
        "pct_non_white", "mean_dist_to_tfl_m", "log_ss_rate"
    ]
    for col in to_standardise:
        mean = reg_df[col].mean()
        std = reg_df[col].std()
        reg_df[f"{col}_std"] = (reg_df[col] - mean) / std

    # Borough dummies
    borough_dummies = pd.get_dummies(
        reg_df["LAD22NM"], drop_first=True, dtype=float
    )
    ref_borough = reg_df["LAD22NM"].sort_values().iloc[0]
    logger.info(f"Borough dummies: {borough_dummies.shape[1]} "
                f"(reference: {ref_borough})")

    # Spatial weights
    reg_gdf = london_gdf.merge(
        reg_df[["LSOA21CD"]], on="LSOA21CD", how="inner"
    )
    reg_gdf = (reg_gdf
               .set_index("LSOA21CD")
               .loc[reg_df["LSOA21CD"]]
               .reset_index())

    logger.info(f"Building Queen contiguity weights for {len(reg_gdf)} LSOAs...")
    w = libpysal.weights.Queen.from_dataframe(reg_gdf, geom_col="geometry")
    w.transform = "r"
    logger.info(f"Weights built: mean neighbours = {w.mean_neighbors:.2f}")

    return reg_df, borough_dummies, w


def build_matrices(reg_df, borough_dummies):
    """Assemble design matrices for all models."""
    X_base = reg_df[X_COLS].values
    borough_fe = borough_dummies.loc[reg_df.index].values
    X_with_fe = np.hstack([X_base, borough_fe])

    y = reg_df["ss_count"].values
    y_rate = reg_df["log_ss_rate_std"].values
    y_pace = reg_df["ss_count_pace"].fillna(0).astype(int).values
    y_s60 = reg_df["ss_count_s60"].fillna(0).astype(int).values

    gini_std = reg_df["gini_housing_std"].values
    income_std = reg_df["income_score_std"].values
    interaction = (gini_std * income_std).reshape(-1, 1)

    X_nb1 = sm.add_constant(np.hstack([gini_std.reshape(-1, 1), borough_fe]))
    X_nb2 = sm.add_constant(np.hstack([X_base, borough_fe]))
    X_nb3 = sm.add_constant(np.hstack([X_base, interaction, borough_fe]))

    return (y, y_rate, y_pace, y_s60,
            X_with_fe, X_nb1, X_nb2, X_nb3)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.1: return "*"
    return ""


def build_table2(ols_model, sdm_model, x_names):
    """Extract OLS and SDM coefficients for main regressors."""
    rows = []
    for i, name in enumerate(x_names):
        ols_coef = ols_model.betas[i + 1][0]
        ols_se = ols_model.std_err[i + 1]
        ols_t = ols_model.t_stat[i + 1][0]
        ols_p = 2 * (1 - stats.t.cdf(
            abs(ols_t), df=ols_model.n - ols_model.k
        ))

        sdm_coef = sdm_model.betas[i][0]
        sdm_se = sdm_model.std_err[i]
        sdm_p = sdm_model.z_stat[i][1]

        rows.append({
            "Variable": name,
            "OLS coef": f"{ols_coef:.3f}{stars(ols_p)}",
            "OLS SE": f"({ols_se:.3f})",
            "SDM coef": f"{sdm_coef:.3f}{stars(sdm_p)}",
            "SDM SE": f"({sdm_se:.3f})"
        })

    rows += [
        {"Variable": "Borough fixed effects",
         "OLS coef": "Y", "OLS SE": "",
         "SDM coef": "Y", "SDM SE": ""},
        {"Variable": "Rho",
         "OLS coef": "", "OLS SE": "",
         "SDM coef": f"{sdm_model.rho:.3f}", "SDM SE": ""},
        {"Variable": "Observations",
         "OLS coef": str(ols_model.n), "OLS SE": "",
         "SDM coef": str(sdm_model.n), "SDM SE": ""},
        {"Variable": "R²",
         "OLS coef": f"{ols_model.r2:.3f}", "OLS SE": "",
         "SDM coef": "", "SDM SE": ""},
        {"Variable": "Log-likelihood",
         "OLS coef": "", "OLS SE": "",
         "SDM coef": f"{sdm_model.logll:.3f}", "SDM SE": ""}
    ]
    return pd.DataFrame(rows)


def build_table3(nb1, nb2, nb3, x_names):
    """Extract negative binomial coefficients for main regressors."""
    rows = []
    for i, name in enumerate(x_names):
        idx = i + 1  # skip constant
        m1_coef = f"{nb1.params[1]:.3f}{stars(nb1.pvalues[1])}" if i == 0 else ""
        m1_se = f"({nb1.bse[1]:.3f})" if i == 0 else ""

        rows.append({
            "Variable": name,
            "Model 1 coef": m1_coef, "Model 1 SE": m1_se,
            "Model 2 coef": f"{nb2.params[idx]:.3f}{stars(nb2.pvalues[idx])}",
            "Model 2 SE": f"({nb2.bse[idx]:.3f})",
            "Model 3 coef": f"{nb3.params[idx]:.3f}{stars(nb3.pvalues[idx])}",
            "Model 3 SE": f"({nb3.bse[idx]:.3f})"
        })

    rows.append({
        "Variable": "Gini × Income deprivation",
        "Model 1 coef": "", "Model 1 SE": "",
        "Model 2 coef": "", "Model 2 SE": "",
        "Model 3 coef": f"{nb3.params[9]:.3f}{stars(nb3.pvalues[9])}",
        "Model 3 SE": f"({nb3.bse[9]:.3f})"
    })
    rows += [
        {"Variable": "Borough fixed effects",
         "Model 1 coef": "Y", "Model 1 SE": "",
         "Model 2 coef": "Y", "Model 2 SE": "",
         "Model 3 coef": "Y", "Model 3 SE": ""},
        {"Variable": "Observations",
         "Model 1 coef": str(int(nb1.nobs)), "Model 1 SE": "",
         "Model 2 coef": str(int(nb2.nobs)), "Model 2 SE": "",
         "Model 3 coef": str(int(nb3.nobs)), "Model 3 SE": ""},
        {"Variable": "Log-likelihood",
         "Model 1 coef": f"{nb1.llf:.3f}", "Model 1 SE": "",
         "Model 2 coef": f"{nb2.llf:.3f}", "Model 2 SE": "",
         "Model 3 coef": f"{nb3.llf:.3f}", "Model 3 SE": ""},
        {"Variable": "AIC",
         "Model 1 coef": f"{nb1.aic:.3f}", "Model 1 SE": "",
         "Model 2 coef": f"{nb2.aic:.3f}", "Model 2 SE": "",
         "Model 3 coef": f"{nb3.aic:.3f}", "Model 3 SE": ""}
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Marginal effects
# ---------------------------------------------------------------------------

def marginal_effects_nb(model, X, x_names, n_main):
    """Marginal effect at mean and average partial effect for NB Model 2."""
    betas = model.params[:-1]  # exclude dispersion parameter
    mu = np.exp(X @ betas)

    mem_list, ape_list = [], []
    for i in range(1, n_main + 1):
        beta_i = betas[i]
        mu_bar = np.exp(X.mean(axis=0) @ betas)
        mem_list.append(beta_i * mu_bar)
        ape_list.append(np.mean(beta_i * mu))

    n_boot = 500
    ape_boot = np.zeros((n_boot, n_main))
    rng = np.random.default_rng(42)
    for b in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        mu_b = np.exp(X[idx] @ betas)
        for i in range(n_main):
            ape_boot[b, i] = np.mean(betas[i + 1] * mu_b)

    return pd.DataFrame({
        "Variable": x_names,
        "MEM": mem_list,
        "APE": ape_list,
        "APE_CI_lo": np.percentile(ape_boot, 2.5, axis=0),
        "APE_CI_hi": np.percentile(ape_boot, 97.5, axis=0)
    })


def plot_marginal_effects(me_df, output_path):
    """Replicate Figure 3: marginal effects plot."""
    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(me_df))

    ax.scatter(me_df["MEM"], y_pos, marker="o", color="black",
               zorder=5, label="Average ME", s=40)
    ax.scatter(me_df["APE"], y_pos + 0.2, marker="^", color="black",
               zorder=5, label="Average PE", s=40)

    for i, row in me_df.iterrows():
        ax.plot([row["APE_CI_lo"], row["APE_CI_hi"]],
                [i + 0.2, i + 0.2], color="black", linewidth=1.2)

    ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_pos + 0.1)
    ax.set_yticklabels(me_df["Variable"], fontsize=10)
    ax.set_xlabel("Marginal Effect", fontsize=11)
    ax.set_title(
        "Marginal and Average Partial Effect of Covariates on S&S\n"
        "(Model 2, Negative Binomial)",
        fontsize=11
    )
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Interaction plot
# ---------------------------------------------------------------------------

def predict_nb_at_values(model, x_template, gini_idx, income_idx,
                         gini_range, income_levels,
                         has_interaction=False, interaction_idx=None):
    """Predicted S&S count over Gini range at fixed income levels."""
    betas = model.params[:-1]
    results = {}
    for inc_level in income_levels:
        preds = []
        for g in gini_range:
            x = x_template.copy()
            x[gini_idx] = g
            x[income_idx] = inc_level
            if has_interaction and interaction_idx is not None:
                x[interaction_idx] = g * inc_level
            preds.append(np.exp(x @ betas))
        results[inc_level] = preds
    return results


def plot_interaction(nb2, nb3, X_nb2, X_nb3, output_path):
    """Replicate Figure 4: interaction effect on predicted S&S."""
    gini_range = np.linspace(-2, 2, 100)
    income_levels = [-2, -1, 0, 1, 2]
    colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]

    x_template_m2 = np.zeros(X_nb2.shape[1])
    x_template_m2[0] = 1
    x_template_m3 = np.zeros(X_nb3.shape[1])
    x_template_m3[0] = 1

    preds_m2 = predict_nb_at_values(
        nb2, x_template_m2, 1, 4, gini_range, income_levels
    )
    preds_m3 = predict_nb_at_values(
        nb3, x_template_m3, 1, 4, gini_range, income_levels,
        has_interaction=True, interaction_idx=9
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, preds, title in zip(
            axes,
            [preds_m2, preds_m3],
            ["Linear (Model 2)", "Interaction (Model 3)"]
    ):
        for inc_level, color in zip(income_levels, colors):
            ax.plot(gini_range, preds[inc_level], color=color,
                    linewidth=2, label=str(inc_level))
        ax.set_xlabel("Gini coefficient (SD)", fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.set_xlim(-2, 2)

    axes[0].set_ylabel("Predicted S&S", fontsize=11)
    handles = [plt.Line2D([0], [0], color=c, linewidth=2) for c in colors]
    fig.legend(
        handles, [str(l) for l in income_levels],
        title="Income deprivation (SD)",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=9,
        title_fontsize=10
    )
    plt.suptitle(
        "Effect of Interaction Term on Predicted S&S",
        fontsize=13, y=1.02
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved {output_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUTS / "tables", exist_ok=True)
    os.makedirs(OUTPUTS / "figures", exist_ok=True)

    # Load data
    analytical = pd.read_csv(DATA_INTERIM / "analytical_dataset.csv")
    london_gdf = gpd.read_file(
        DATA_INTERIM / "boundaries" / "london_lsoa_2021.gpkg",
        layer="london_lsoa_2021"
    )[["LSOA21CD", "geometry"]]

    # Prepare
    reg_df, borough_dummies, w = load_and_prepare(analytical, london_gdf)
    (y, y_rate, y_pace, y_s60,
     X_with_fe, X_nb1, X_nb2, X_nb3) = build_matrices(reg_df, borough_dummies)

    # --- OLS ---
    logger.info("Fitting OLS...")
    ols = spreg.OLS(
        y.reshape(-1, 1), X_with_fe, w=w,
        spat_diag=True, moran=True,
        name_y="ss_count",
        name_x=X_NAMES + list(borough_dummies.columns)
    )
    logger.info(f"OLS R²: {ols.r2:.3f}")
    logger.info(f"OLS Moran's I: {ols.moran_res[0]:.3f} "
                f"(p={ols.moran_res[2]:.4f})")

    # --- SDM ---
    logger.info("Fitting Spatial Durbin Model...")
    sdm = spreg.ML_Lag(
        y.reshape(-1, 1), X_with_fe, w,
        slx_lags=1, spat_diag=True,
        name_y="ss_count",
        name_x=X_NAMES + list(borough_dummies.columns)
    )
    logger.info(f"SDM rho: {sdm.rho:.3f}")

    logger.info("Monte Carlo Moran's I on SDM residuals (1,000 permutations)...")
    mi_sdm = esda.Moran(sdm.u.flatten(), w, permutations=1000)
    logger.info(f"SDM Moran's I: {mi_sdm.I:.3f} (p={mi_sdm.p_sim:.3f})")

    # Table 2
    table2 = build_table2(ols, sdm, X_NAMES)
    table2.to_csv(OUTPUTS / "tables" / "table2_ols_sdm.csv", index=False)
    logger.info("Saved table2_ols_sdm.csv")

    # --- Negative binomial ---
    logger.info("Fitting NB Model 1...")
    nb1 = NegativeBinomial(y, X_nb1).fit(
        method="bfgs", maxiter=1000, disp=False
    )
    logger.info("Fitting NB Model 2...")
    nb2 = NegativeBinomial(y, X_nb2).fit(
        method="bfgs", maxiter=1000, disp=False
    )
    logger.info(f"Model 2 Gini coef: {nb2.params[1]:.3f} "
                f"(p={nb2.pvalues[1]:.4f}), "
                f"IRR: {np.exp(nb2.params[1]):.3f}")
    logger.info("Fitting NB Model 3...")
    nb3 = NegativeBinomial(y, X_nb3).fit(
        method="bfgs", maxiter=1000, disp=False
    )

    # Table 3
    table3 = build_table3(nb1, nb2, nb3, X_NAMES)
    table3.to_csv(OUTPUTS / "tables" / "table3_negative_binomial.csv", index=False)
    logger.info("Saved table3_negative_binomial.csv")

    # Marginal effects
    me_df = marginal_effects_nb(nb2, X_nb2, X_NAMES, len(X_NAMES))
    me_df.to_csv(OUTPUTS / "tables" / "marginal_effects_model2.csv", index=False)
    plot_marginal_effects(me_df, OUTPUTS / "figures" / "fig3_marginal_effects.png")

    # Interaction plot
    plot_interaction(
        nb2, nb3, X_nb2, X_nb3,
        OUTPUTS / "figures" / "fig4_interaction.png"
    )

    # --- Robustness: rate outcome ---
    logger.info("Fitting OLS and SDM (rate outcome)...")
    ols_rate = spreg.OLS(
        y_rate.reshape(-1, 1), X_with_fe, w=w,
        spat_diag=True, moran=True,
        name_y="log_ss_rate",
        name_x=X_NAMES + list(borough_dummies.columns)
    )
    sdm_rate = spreg.ML_Lag(
        y_rate.reshape(-1, 1), X_with_fe, w,
        slx_lags=1, spat_diag=True,
        name_y="log_ss_rate",
        name_x=X_NAMES + list(borough_dummies.columns)
    )
    table_a2 = build_table2(ols_rate, sdm_rate, X_NAMES)
    table_a2.to_csv(
        OUTPUTS / "tables" / "table_a2_rate_robustness.csv", index=False
    )
    logger.info("Saved table_a2_rate_robustness.csv")

    # --- Robustness: no Westminster ---
    logger.info("Fitting models without Westminster...")
    reg_df_nw = reg_df[reg_df["LAD22NM"] != "Westminster"].copy()
    borough_fe_nw = pd.get_dummies(
        reg_df_nw["LAD22NM"], drop_first=True, dtype=float
    )
    reg_gdf_nw = (london_gdf
                  .merge(reg_df_nw[["LSOA21CD"]], on="LSOA21CD", how="inner")
                  .set_index("LSOA21CD")
                  .loc[reg_df_nw["LSOA21CD"]]
                  .reset_index())
    w_nw = libpysal.weights.Queen.from_dataframe(
        reg_gdf_nw, geom_col="geometry"
    )
    w_nw.transform = "r"

    X_base_nw = reg_df_nw[X_COLS].values
    y_nw = reg_df_nw["ss_count"].values
    gini_nw = reg_df_nw["gini_housing_std"].values
    income_nw = reg_df_nw["income_score_std"].values
    interaction_nw = (gini_nw * income_nw).reshape(-1, 1)
    borough_fe_nw_vals = borough_fe_nw.values

    nb1_nw = NegativeBinomial(
        y_nw,
        sm.add_constant(np.hstack([gini_nw.reshape(-1, 1), borough_fe_nw_vals]))
    ).fit(method="bfgs", maxiter=1000, disp=False)

    nb2_nw = NegativeBinomial(
        y_nw,
        sm.add_constant(np.hstack([X_base_nw, borough_fe_nw_vals]))
    ).fit(method="bfgs", maxiter=1000, disp=False)

    nb3_nw = NegativeBinomial(
        y_nw,
        sm.add_constant(np.hstack([X_base_nw, interaction_nw, borough_fe_nw_vals]))
    ).fit(method="bfgs", maxiter=1000, disp=False)

    table_a3 = build_table3(nb1_nw, nb2_nw, nb3_nw, X_NAMES)
    table_a3.to_csv(
        OUTPUTS / "tables" / "table_a3_no_westminster.csv", index=False
    )
    logger.info("Saved table_a3_no_westminster.csv")

    # --- Robustness: S60 vs PACE ---
    logger.info("Fitting NB models for S60 and PACE...")
    nb2_pace = NegativeBinomial(y_pace, X_nb2).fit(
        method="bfgs", maxiter=1000, disp=False
    )
    nb2_s60 = NegativeBinomial(y_s60, X_nb2).fit(
        method="bfgs", maxiter=1000, disp=False
    )

    s60_pace_table = pd.DataFrame({
        "Variable": X_NAMES,
        "All searches coef": [f"{nb2.params[i+1]:.3f}"
                              for i in range(len(X_NAMES))],
        "All searches IRR": [f"{np.exp(nb2.params[i+1]):.3f}"
                             for i in range(len(X_NAMES))],
        "PACE coef": [f"{nb2_pace.params[i+1]:.3f}"
                      for i in range(len(X_NAMES))],
        "PACE IRR": [f"{np.exp(nb2_pace.params[i+1]):.3f}"
                     for i in range(len(X_NAMES))],
        "S60 coef": [f"{nb2_s60.params[i+1]:.3f}"
                     for i in range(len(X_NAMES))],
        "S60 IRR": [f"{np.exp(nb2_s60.params[i+1]):.3f}"
                    for i in range(len(X_NAMES))]
    })
    s60_pace_table.to_csv(
        OUTPUTS / "tables" / "table_s60_pace_robustness.csv", index=False
    )
    logger.info("Saved table_s60_pace_robustness.csv")

    # --- Moran's I summary ---
    logger.info("=" * 55)
    logger.info("Moran's I Summary")
    logger.info("=" * 55)
    logger.info(f"OLS residuals (count): I={ols.moran_res[0]:.3f}, "
                f"p={ols.moran_res[2]:.4f} → autocorrelation confirmed")
    logger.info(f"SDM residuals (count): I={mi_sdm.I:.3f}, "
                f"p={mi_sdm.p_sim:.3f} → autocorrelation eliminated")
    logger.info(f"OLS residuals (rate):  I={ols_rate.moran_res[0]:.3f}, "
                f"p={ols_rate.moran_res[2]:.4f}")
    logger.info("=" * 55)
    logger.info("All model outputs saved successfully")


if __name__ == "__main__":
    main()

