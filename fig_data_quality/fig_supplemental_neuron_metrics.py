import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import figrid as fg

from fig_data_quality.plot import get_3colors_region
from fig_data_quality.tables import load_channels, load_clusters

from reproducible_ephys_functions import save_figure_path, figure_style, get_row_coord, get_label_pos

"""
Generates Fig. 1 Supp. 4
"""

def compute_yields(
    ibl_clusters, ibl_channels, ste_clusters, ste_channels, aln_clusters, aln_channels
):
    """
    Given cluster and channel tables (assumed to be already region-filtered) for IBL, Steinmetz, and Allen,
    return yield tables

    :param ibl_clusters:
    :param ibl_channels:
    :param ste_clusters:
    :param ste_channels:
    :param aln_clusters:
    :param aln_channels:

    :returns: yield tables with columns: ["insertion","passing_units",
        "num_sites", "passing_per_site", "all_units", "source"] (and "lab" for IBL)
    """
    passing_ibl = ibl_clusters.groupby("insertion").agg(
        passing_units=pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x == 1.0])),
        lab=pd.NamedAgg(column="lab", aggfunc="first"),
    )
    sites_ibl = ibl_channels.groupby("insertion").agg(
        num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count")
    )
    passing_per_site_ibl = passing_ibl.merge(sites_ibl, on="insertion")

    all_ibl = ibl_clusters.groupby("insertion").count()["label"].to_numpy()
    all_ste = ste_clusters.groupby("insertion").count()["label"].to_numpy()
    all_aln = aln_clusters.groupby("insertion").count()["label"].to_numpy()

    passing_ste = ste_clusters.groupby("insertion").agg(
        passing_units=pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x == 1.0]))
    )
    sites_ste = ste_channels.groupby("insertion").agg(
        num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count")
    )
    passing_per_site_ste = passing_ste.merge(sites_ste, on="insertion")

    passing_aln = aln_clusters.groupby("insertion").agg(
        passing_units=pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x == 1.0]))
    )
    sites_aln = aln_channels.groupby("insertion").agg(
        num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count")
    )
    passing_per_site_aln = passing_aln.merge(sites_aln, on="insertion")

    passing_per_site_ibl["passing_per_site"] = (
        passing_per_site_ibl["passing_units"] / passing_per_site_ibl["num_sites"]
    )
    passing_per_site_ste["passing_per_site"] = (
        passing_per_site_ste["passing_units"] / passing_per_site_ste["num_sites"]
    )
    passing_per_site_aln["passing_per_site"] = (
        passing_per_site_aln["passing_units"] / passing_per_site_aln["num_sites"]
    )

    passing_per_site_ibl["all_units"] = all_ibl
    passing_per_site_ste["all_units"] = all_ste
    passing_per_site_aln["all_units"] = all_aln

    passing_per_site_ibl["source"] = "IBL"
    passing_per_site_ste["source"] = "Steinmetz"
    passing_per_site_aln["source"] = "Allen"

    return passing_per_site_ibl, passing_per_site_ste, passing_per_site_aln


def plot_supplemental_neuron_metrics():

    figure_style()
    width = 7
    height = 8
    fig = plt.figure(figsize=(width, height))
    xspans = get_row_coord(width, [1], pad=0.6)
    yspans = get_row_coord(height, [1, 1, 1], hspace=0.8, pad=0.3)

    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0], dim=[2, 4], hspace=0.3, wspace=0.5),
          'B': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1], dim=[2, 4], hspace=0.3, wspace=0.5),
          'C': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[2], dim=[2, 4], hspace=0.3, wspace=0.5)
          }

    for ax_label, region in zip(['A', 'B', 'C'], ['Isocortex', 'HPF', 'TH']):
        plot_neuron_metrics_per_region(region, axes=np.array(ax[ax_label]).reshape(2, 4), save=False)

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width,xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              ]
    fg.add_labels(fig, labels)

    fig_path = save_figure_path(figure="fig_data_quality")
    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=adjust/height, left=(adjust)/width, right=1-adjust/width)
    fig.savefig(fig_path.joinpath(f"fig_supplemental_neuron_metrics_test.svg"))
    fig.savefig(fig_path.joinpath(f"fig_supplemental_neuron_metrics_test.pdf"))


def plot_neuron_metrics_per_region(region, axes=None, save=True):

    err_kws = {"markersize": 20, "linewidth": 1.0}
    region_fullname = {"Isocortex": "Cortex", "TH": "Thalamus", "HPF": "Hippocampus"}

    ibl_clusters = load_clusters("re_147", filter_region=region)
    # channels unchanged by different sorting version
    ibl_channels = load_channels("re", filter_region=region)
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[
        list(ibl_clusters.index.get_level_values(0).unique())
    ]
    ste_clusters = load_clusters("steinmetz", filter_region=region)
    ste_channels = load_channels("steinmetz", filter_region=region)
    aln_clusters = load_clusters("allen", filter_region=region)
    aln_channels = load_channels("allen", filter_region=region)
    colors, colors_translucent = get_3colors_region(region, blue=True)

    if axes is None:
        figure_style()
        fig, axes = plt.subplots(2, 4, figsize=(8, 5))
        ms = 2
    else:
        fig = plt.gcf()
        ms = 2.5

    passing_per_site_ibl, passing_per_site_ste, passing_per_site_aln = compute_yields(
        ibl_clusters,
        ibl_channels,
        ste_clusters,
        ste_channels,
        aln_clusters,
        aln_channels,
    )
    ## 0, 0 Bar plot of number of insertions
    num_ins_ste = len(ste_clusters.index.levels[0])
    num_ins_aln = len(aln_clusters.index.levels[0])
    num_ins_ibl = len(ibl_clusters.index.levels[0])

    axes[0, 0].bar(
        ["IBL", "Steinmetz", "Allen"],
        [num_ins_ibl, num_ins_ste, num_ins_aln],
        color=colors,
    )

    axes[0, 0].set_xticks(axes[0, 0].get_xticks())
    axes[0, 0].set_xticklabels(["IBL", "STE", "ALN"])
    axes[0, 0].set_ylabel("Number of insertions")

    ## 0, 1 Pointplot of number of sites (in region) per insertion
    sites_ste = ste_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites_aln = aln_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites_ibl = ibl_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites = pd.DataFrame.from_dict(
        dict(zip(["IBL", "Steinmetz", "Allen"], [sites_ibl, sites_ste, sites_aln])),
        orient="index",
    )

    sns.pointplot(
        x="source",
        y="num_sites",
        data=passing_per_site_ste,
        ax=axes[0, 1],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )

    sns.pointplot(
        x="source",
        y="num_sites",
        data=passing_per_site_ibl,
        ax=axes[0, 1],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.pointplot(
        x="source",
        y="num_sites",
        data=passing_per_site_aln,
        ax=axes[0, 1],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.stripplot(sites.T, ax=axes[0, 1], palette=colors_translucent, size=ms)
    axes[0, 1].set_xticks(axes[0, 1].get_xticks())
    axes[0, 1].set_xticklabels(["IBL", "STE", "ALN"])
    axes[0, 1].set_xlabel(None)
    axes[0, 1].set_ylabel("Electrode sites \n per insertion")

    ## 0, 2 Number of units per insertion (in this region)
    units_per_ins_ibl = ibl_clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins_ste = ste_clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins_aln = aln_clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins_all = pd.DataFrame.from_dict(
        dict(
            zip(
                ["IBL", "Steinmetz", "Allen"],
                [units_per_ins_ibl, units_per_ins_ste, units_per_ins_aln],
            )
        ),
        orient="index",
    )
    sns.pointplot(
        x="source",
        y="all_units",
        data=passing_per_site_ste,
        ax=axes[0, 2],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.pointplot(
        x="source",
        y="all_units",
        data=passing_per_site_ibl,
        ax=axes[0, 2],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.pointplot(
        x="source",
        y="all_units",
        data=passing_per_site_aln,
        ax=axes[0, 2],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.stripplot(units_per_ins_all.T, ax=axes[0, 2], palette=colors_translucent, size=ms)
    axes[0, 2].set_xticks(axes[0, 2].get_xticks())
    axes[0, 2].set_xticklabels(["IBL", "STE", "ALN"])
    axes[0, 2].set_xlabel(None)
    axes[0, 2].set_ylabel("Neurons per insertion")

    ## 0, 3 Bar plot of mean units per site
    units_per_site_ibl = len(ibl_clusters) / len(ibl_channels)
    units_per_site_ste = len(ste_clusters) / len(ste_channels)
    units_per_site_aln = len(aln_clusters) / len(aln_channels)
    axes[0, 3].bar(
        ["IBL", "Steinmetz", "Allen"],
        [units_per_site_ibl, units_per_site_ste, units_per_site_aln],
        color=colors,
    )
    axes[0, 3].set_xticks(axes[0, 3].get_xticks())
    axes[0, 3].set_xticklabels(["IBL", "STE", "ALN"])
    axes[0, 3].set_xlabel(None)
    axes[0, 3].set_ylabel("Neurons per \n electrode site")

    ## 1, 0 Fraction passing sliding refractory period metric
    ibl_SRP = ibl_clusters.slidingRP_viol
    ibl_frac = sum(ibl_SRP) / len(ibl_SRP)
    ste_SRP = ste_clusters.slidingRP_viol
    ste_frac = sum(ste_SRP) / len(ste_SRP)
    aln_SRP = aln_clusters.slidingRP_viol
    aln_frac = sum(aln_SRP) / len(aln_SRP)
    axes[1, 0].bar(
        ["IBL", "Steinmetz", "Allen"], [ibl_frac, ste_frac, aln_frac], color=colors
    )

    axes[1, 0].set_ylabel("Fraction pass (sliding \n refractory period metric)")
    axes[1, 0].set_xticks(axes[1, 0].get_xticks())
    axes[1, 0].set_xticklabels(["IBL", "STE", "ALN"])

    ## 1, 1 Fraction passing noise cutoff metric
    ibl_NC = ibl_clusters.noise_cutoff.to_numpy()
    frac_ibl = sum(ibl_NC < 5.0) / len(ibl_NC)
    ibl_NC = np.log10(ibl_NC)
    ste_NC = ste_clusters.noise_cutoff.to_numpy()
    frac_ste = sum(ste_NC < 5.0) / len(ste_NC)
    ste_NC = np.log10(ste_NC)
    aln_NC = aln_clusters.noise_cutoff.to_numpy()
    frac_aln = sum(aln_NC < 5.0) / len(aln_NC)
    aln_NC = np.log10(aln_NC)
    axes[1, 1].bar(
        ["IBL", "Steinmetz", "Allen"], [frac_ibl, frac_ste, frac_aln], color=colors
    )

    axes[1, 1].set_ylabel("Fraction pass \n (noise cutoff metric)")
    axes[1, 1].set_xticks(axes[1, 1].get_xticks())
    axes[1, 1].set_xticklabels(["IBL", "STE", "ALN"])
    axes[1, 1].set_xlabel(None)


    ## 1, 2 Fraction passing amplitude median metric
    ibl_amp = ibl_clusters.amp_median.to_numpy() * 1e6
    frac_ibl = sum(ibl_amp > 50) / len(ibl_amp)
    ste_amp = ste_clusters.amp_median.to_numpy() * 1e6
    frac_ste = sum(ste_amp > 50) / len(ste_amp)
    aln_amp = aln_clusters.amp_median.to_numpy() * 1e6
    frac_aln = sum(aln_amp > 50) / len(aln_amp)
    axes[1, 2].bar(
        ["IBL", "Steinmetz", "Allen"], [frac_ibl, frac_ste, frac_aln], color=colors
    )
    axes[1, 2].set_ylabel("Fraction pass \n (amplitude metric)")
    axes[1, 2].set_xticks(axes[1, 2].get_xticks())
    axes[1, 2].set_xticklabels(["IBL", "STE", "ALN"])
    axes[1, 2].set_xlabel(None)

    ## 1, 3 Passing units per site (Yield)
    (
        passing_per_site_ibl,
        passing_per_site_ste,
        passing_per_site_aln,
    ) = compute_yields(
        ibl_clusters,
        ibl_channels,
        ste_clusters,
        ste_channels,
        aln_clusters,
        aln_channels,
    )
    passing_per_site_all = dict(
        zip(
            ["IBL", "Steinmetz", "Allen"],
            [
                passing_per_site_ibl["passing_per_site"].to_numpy(),
                passing_per_site_ste["passing_per_site"].to_numpy(),
                passing_per_site_aln["passing_per_site"].to_numpy(),
            ],
        )
    )
    passing_per_site_all = pd.DataFrame.from_dict(passing_per_site_all, orient="index")
    sns.pointplot(
        x="source",
        y="passing_per_site",
        data=passing_per_site_ibl,
        ax=axes[1, 3],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.pointplot(
        x="source",
        y="passing_per_site",
        data=passing_per_site_ste,
        ax=axes[1, 3],
        linestyles="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.pointplot(
        x="source",
        y="passing_per_site",
        data=passing_per_site_aln,
        ax=axes[1, 3],
        linestyle="none",
        markersize=ms,
        marker="none",
        capsize=0.2,
        errorbar=("se", 1),
        color="black",
        err_kws=err_kws,
        zorder=100,
    )
    sns.stripplot(passing_per_site_all.T, ax=axes[1, 3], palette=colors_translucent, size=ms)
    axes[1, 3].set_ylabel("Neurons passing all \n QC per site")
    axes[1, 3].set_xticks(axes[1, 3].get_xticks())
    axes[1, 3].set_xticklabels(["IBL", "STE", "ALN"])
    axes[1, 3].set_xlabel(None)

    if save:
        fig.suptitle(f"IBL metrics: {region_fullname[region]}")
        fig.tight_layout()
        fig_path = save_figure_path(figure="fig_data_quality")
        fig.savefig(fig_path.joinpath(f"fig_supplemental_neuron_metrics_{region}.svg"))
        fig.savefig(fig_path.joinpath(f"fig_supplemental_neuron_metrics_{region}.pdf"))
