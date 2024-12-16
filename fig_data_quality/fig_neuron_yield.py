import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from fig_data_quality.tables import load_channels, load_clusters
from fig_data_quality.plot import compute_yield, get_3colors_region

from reproducible_ephys_functions import save_figure_path, save_data_path

"""
Generates Figure 1f
Also includes the ANOVA on dataset and region (Results > Neuropixels recordings during decision-making target the same brain location)
"""

regions = ["Isocortex", "HPF", "TH"]

def load_yield_info():
    
    # put all the invidual insertion yields into one DF
    dfs = []
    for region in regions:
        
        ibl_clusters = load_clusters("re_147", region)
        # channels unchanged by different sorting version
        ibl_channels = load_channels("re", region)
        # make sure we only include insertions listed in the clusters table
        ibl_channels = ibl_channels.loc[
            list(ibl_clusters.index.get_level_values(0).unique())
        ]

        al_clusters = load_clusters("allen", region)
        al_channels = load_channels("allen", region)

        st_clusters = load_clusters("steinmetz", region)
        st_channels = load_channels("steinmetz", region)

        re_yield, al_yield = compute_yield(
            ibl_clusters, ibl_channels, al_clusters, al_channels
        )
        _, st_yield = compute_yield(ibl_clusters, ibl_channels, st_clusters, st_channels)

        re_yield["region"] = region
        re_yield.drop(columns=["lab"], inplace=True)
        re_yield.rename(
            columns={
                "passing_units": "nunits",
                "num_sites": "nsites",
                "passing_per_site": "yield",
            },
            inplace=True,
        )
        re_yield["dataset"] = "IBL"

        al_yield["region"] = region
        al_yield.rename(
            columns={
                "passing_units": "nunits",
                "num_sites": "nsites",
                "passing_per_site": "yield",
            },
            inplace=True,
        )
        al_yield["dataset"] = "Allen"

        st_yield["region"] = region
        st_yield.rename(
            columns={
                "passing_units": "nunits",
                "num_sites": "nsites",
                "passing_per_site": "yield",
            },
            inplace=True,
        )
        st_yield["dataset"] = "Steinmetz"

        dfs.append(re_yield)
        dfs.append(al_yield)
        dfs.append(st_yield)

    return pd.concat(dfs)


def plot_neuron_yield(ax=None, save=True):

    df = load_yield_info()
    if ax is None:
        fontsize = 7.0
        plt.rcParams["axes.labelsize"] = fontsize
        fig, ax = plt.subplots(1, 3, figsize=(3, 2))
    else:
        fig = plt.gcf()

    region_fullname = {"Isocortex": "Cortex", "HPF": "Hippocampus", "TH": "Thalamus"}
    err_kws = {"markersize": 20, "linewidth": 1.0}
    for i, region in enumerate(regions):

        b = sns.stripplot(
            data=df[df.region == region],
            x="dataset",
            y="yield",
            ax=ax[i],
            zorder=-1,
            size=2,
            alpha=0.6,
            color='grey',
            order=["IBL", "Steinmetz", "Allen"],
        )

        sns.pointplot(
            data=df[df.region == region],
            x="dataset",
            y="yield",
            ax=ax[i],
            markersize=2,
            markers="none",
            capsize=0.2,
            errorbar=("se", 1),
            color="black",
            err_kws=err_kws,
            linestyle="none",
            order=["IBL", "Steinmetz", "Allen"],
        )

        colors, _ = get_3colors_region(region, blue=True)
        ax[i].set_title(region_fullname[region], fontsize=7, color=colors[0])
        ax[i].set_xlabel(None)
        tx = ax[i].get_xticks()
        ax[i].set_xticks(tx, ["IBL", "STE", "ALN"])

        ax[i].set_ylim(0, 1.5)

        if i != 0:
            ax[i].set_ylabel(None)
            ax[i].set_yticks([])
            sns.despine(ax=ax[i], left=True)

        else:
            ax[i].set_ylabel("Neuron yield")
            ax[i].set_yticks([0.0, 0.5, 1.0, 1.5])
            sns.despine(ax=ax[i], trim=True)
            ty = ax[i].get_yticks()
            ly = ax[i].get_yticklabels()
            ax[i].set_yticks(ty, ly)

    if save:
        fig_path = save_figure_path(figure="fig_data_quality")
        fig.savefig(fig_path.joinpath("fig_neuron_yield.svg"))
        fig.savefig(fig_path.joinpath("fig_neuron_yield.pdf"))

def save_neuron_yield_anova():

    df = load_yield_info()

    # 2-way ANOVA on region and dataset
    df = df.drop(columns=["nunits", "nsites"])
    dataset_map = {"IBL": 1, "Steinmetz": 2, "Allen": 3}
    region_map = {"Isocortex": 1, "TH": 2, "HPF": 3}
    df = df.replace({"region": region_map, "dataset": dataset_map})

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    model = ols(
        'Q("yield") ~ C(Q("dataset")) + C(Q("region")) + C(Q("dataset")):C(Q("region"))', df
    ).fit()
    anova = sm.stats.anova_lm(model, typ=2)

    data_path = save_data_path(figure="fig_data_quality")
    anova.to_csv(data_path.joinpath("neuron_yield_anova.csv"))
