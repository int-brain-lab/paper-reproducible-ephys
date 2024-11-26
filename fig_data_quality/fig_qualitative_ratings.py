import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fig_data_quality.tables import tables_dir

from reproducible_ephys_functions import save_figure_path, save_data_path, figure_style

"""
Generates Fig. 1 Supp. 5
Also includes two way ANOVA with rater and dataset origin.
"""

def jitter(values, j):
    return values + np.random.normal(j, 0.05, values.shape)

def plot_qualitative_ratings():

    results = pd.read_csv(tables_dir.joinpath("double_blind_results.csv")).loc[:100]

    figure_style()
    width = 7
    height = 3
    fig, ax = plt.subplots(1, 3, figsize=(7, 3))

    err_kws = {"markersize": 20, "linewidth": 1.0}

    raters = ["GC", "NS", "FD"]

    for i, rater in enumerate(raters):
        sns.pointplot(
            x="Source",
            y=rater,
            data=results,
            ax=ax[i],
            markers="none",
            capsize=0.2,
            errorbar=("se", 1),
            color="black",
            err_kws=err_kws,
            linestyle="none",
            order=["IBL", "Steinmetz", "Allen"],
            zorder=100,
        )
        sns.stripplot(
            x="Source",
            y=jitter(results[rater], 0.2),
            data=results,
            ax=ax[i],
            alpha=0.6,
            size=5.0,
            order=["IBL", "Steinmetz", "Allen"],
        )
        ax[i].set_xlabel(None)
        ax[i].set_ylabel(None)
        ax[i].set_title(rater)
        ax[i].set_ylim(0, 10.5)
        tx = ax[i].get_xticks()
        ax[i].set_xticks(tx, ["IBL", "STE", "ALN"])

        if i == 0:
            ax[i].set_ylabel("Score")

    fig_path = save_figure_path(figure="fig_data_quality")
    adjust = 0.3
    fig.subplots_adjust(top=1 - adjust / height, bottom=adjust / height, left=(adjust + 0.1)/ width, right=1 - adjust / width)
    fig.savefig(fig_path.joinpath(f"fig_qualitative_ratings.svg"))
    fig.savefig(fig_path.joinpath(f"fig_qualitative_ratings.pdf"))

def save_qualitative_ratings_anova():

    results = pd.read_csv(tables_dir.joinpath("double_blind_results.csv")).loc[:100]

    df = results.drop(columns=["OID", "UUID"])
    df = df.melt(id_vars=["Source"])
    df.rename(columns={"value": "Score", "variable": "Rater"}, inplace=True)

    # ANOVA
    dataset_map = {"IBL": 1, "Steinmetz": 2, "Allen": 3}
    rater_map = {"GC": 1, "NS": 2, "FD": 3}
    df = df.replace({"Rater": rater_map, "Source": dataset_map})

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    model = ols(
        'Q("Score") ~ C(Q("Source")) + C(Q("Rater")) + C(Q("Source")):C(Q("Rater"))', df
    ).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    data_path = save_data_path(figure="fig_data_quality")
    anova.to_csv(data_path.joinpath("qualitative_ratings_anova.csv"))
