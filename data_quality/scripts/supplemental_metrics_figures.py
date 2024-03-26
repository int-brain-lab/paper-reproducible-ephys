import numpy as np
from data_quality.plots.utils import get_3colors_region
from data_quality.tables import load_channels, load_clusters
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def compute_yield_threeway(ibl_clusters, ibl_channels, 
                  ste_clusters, ste_channels, 
                  aln_clusters, aln_channels
                ):
    """
    Given cluster and channel tables (assumed to be ALREADY FILTERED) return
    yield computations per insertion.

    :param clusters: Clusters table
    :param channels: Channels table
    :param dset: Option name of dset (will set "source" column to this value for seaborn plots)

    :returns: yield table with columns 
    """
    passing_ibl = ibl_clusters.groupby("insertion").agg(
        passing_units = pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x==1.])), 
        lab = pd.NamedAgg(column="lab", aggfunc="first")
        )
    sites_ibl = ibl_channels.groupby("insertion").agg(num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count"))
    passing_per_site_ibl = passing_ibl.merge(sites_ibl, on="insertion")

    passing_ste = ste_clusters.groupby("insertion").agg(passing_units = pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x==1.])))
    sites_ste = ste_channels.groupby("insertion").agg(num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count"))
    passing_per_site_ste = passing_ste.merge(sites_ste, on="insertion")

    passing_aln = aln_clusters.groupby("insertion").agg(passing_units = pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x==1.])))
    sites_aln = aln_channels.groupby("insertion").agg(num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count"))
    passing_per_site_aln = passing_aln.merge(sites_aln, on="insertion")

    passing_per_site_ibl["passing_per_site"] = passing_per_site_ibl["passing_units"]/passing_per_site_ibl["num_sites"]
    passing_per_site_ste["passing_per_site"] = passing_per_site_ste["passing_units"]/passing_per_site_ste["num_sites"]
    passing_per_site_aln["passing_per_site"] = passing_per_site_aln["passing_units"]/passing_per_site_aln["num_sites"]


    passing_per_site_ibl["source"] = "IBL"
    passing_per_site_ste["source"] = "Steinmetz"
    passing_per_site_aln["source"] = "Allen"

    return passing_per_site_ibl, passing_per_site_ste, passing_per_site_aln


for region in ["Isocortex", "TH", "HPF"]:

    ibl_clusters = load_clusters("re_147", filter_region=region)
    ibl_channels = load_channels("re", filter_region=region)
    ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]

    ste_clusters = load_clusters("steinmetz", filter_region=region)
    ste_channels = load_channels("steinmetz", filter_region=region)

    aln_clusters = load_clusters("allen", filter_region=region)
    aln_channels = load_channels("allen", filter_region=region)


    colors, colors_translucent = get_3colors_region(region)

    fig, axes = plt.subplots(2, 4)

    ## num insertions
    num_ins_ste = len(ste_clusters.index.levels[0])
    num_ins_aln = len(aln_clusters.index.levels[0])
    num_ins_ibl = len(ibl_clusters.index.levels[0])
    axes[0,0].bar(["IBL", "Steinmetz", "Allen"], [num_ins_ibl, num_ins_ste, num_ins_aln], color=colors)
    axes[0,0].set_title("Number of insertions", fontsize=8)
    axes[0,0].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)

    ## sites per insertion (in this region)
    sites_ste = ste_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites_aln = aln_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites_ibl = ibl_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites = pd.DataFrame.from_dict(dict(zip(["IBL", "Steinmetz", "Allen"], [sites_ibl, sites_ste, sites_aln])), orient="index")
    sns.stripplot(sites.T, ax=axes[0, 1], palette=colors_translucent)
    axes[0, 1].set_title("Sites per insertion", fontsize=8)
    axes[0, 1].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)
    axes[0, 1].set_ylabel("channels in region")

    ## units per insertion (in this region)
    units_per_ins_ibl = ibl_clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins_ste = ste_clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins_aln = aln_clusters.groupby(["insertion"]).count()["label"].to_numpy()

    units_per_ins_all = pd.DataFrame.from_dict(dict(zip(["IBL", "Steinmetz", "Allen"], [units_per_ins_ibl, units_per_ins_ste, units_per_ins_aln])), orient="index")

    sns.stripplot(units_per_ins_all.T, ax=axes[0, 2], palette=colors_translucent)
    axes[0, 2].set_title("Units per insertion", fontsize=8)
    axes[0, 2].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)
    axes[0, 2].set_ylabel("units in region", fontsize=8)

    ## units per site
    units_per_site_ibl = len(ibl_clusters) / len(ibl_channels)
    units_per_site_ste = len(ste_clusters) / len(ste_channels)
    units_per_site_aln = len(aln_clusters) / len(aln_channels)
    axes[0, 3].bar(["IBL", "Steinmetz", "Allen"], [units_per_site_ibl, units_per_site_ste, units_per_site_aln], color=colors)
    axes[0, 3].set_title("Units per site", fontsize=8)
    axes[0, 3].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)
    axes[0, 3].set_ylabel("units/site in region", fontsize=8)

    ## sliding RP metric
    ibl_SRP = ibl_clusters.slidingRP_viol
    ibl_frac = sum(ibl_SRP) / len(ibl_SRP)
    ste_SRP = ste_clusters.slidingRP_viol
    ste_frac = sum(ste_SRP) / len(ste_SRP)
    aln_SRP = aln_clusters.slidingRP_viol
    aln_frac = sum(aln_SRP) / len(aln_SRP)
    axes[1, 0].bar(["IBL", "Steinmetz", "Allen"], [ibl_frac, ste_frac, aln_frac], color=colors)
    axes[1, 0].set_title("Sliding RP", fontsize=8)
    axes[1, 0].set_ylabel("Fraction of units passing", fontsize=8)
    axes[1, 0].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)

    ## noise cutoff
    ibl_NC = ibl_clusters.noise_cutoff.to_numpy()
    frac_ibl = sum(ibl_NC < 5.) / len(ibl_NC)
    ibl_NC = np.log10(ibl_NC)

    ste_NC = ste_clusters.noise_cutoff.to_numpy()
    frac_ste = sum(ste_NC < 5.) / len(ste_NC)
    ste_NC = np.log10(ste_NC)

    aln_NC = aln_clusters.noise_cutoff.to_numpy()
    frac_aln = sum(aln_NC < 5.) / len(aln_NC)
    aln_NC = np.log10(aln_NC)
    axes[1, 1].bar(["IBL", "Steinmetz", "Allen"], [frac_ibl, frac_ste, frac_aln], color=colors)
    axes[1, 1].set_title("Noise cutoff", fontsize=8)
    axes[1, 1].set_ylabel("Fraction of units passing", fontsize=8)
    axes[1, 1].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)

    ## amplitude median
    ibl_amp = ibl_clusters.amp_median.to_numpy() * 1e6
    frac_ibl = sum(ibl_amp > 50) / len(ibl_amp)
    ste_amp = ste_clusters.amp_median.to_numpy() * 1e6
    frac_ste = sum(ste_amp > 50) / len(ste_amp)
    aln_amp = aln_clusters.amp_median.to_numpy() * 1e6
    frac_aln = sum(aln_amp > 50) / len(aln_amp)
    axes[1, 2].bar(["IBL", "Steinmetz", "Allen"], [frac_ibl, frac_ste, frac_aln], color=colors)
    axes[1, 2].set_title("Amplitude median", fontsize=8)
    axes[1, 2].set_ylabel("Fraction of units passing", fontsize=8)
    axes[1, 2].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)

    ## passing units per site
    passing_per_site_ibl, passing_per_site_ste, passing_per_site_aln = compute_yield_threeway(ibl_clusters, ibl_channels, 
                                                                                            ste_clusters, ste_channels,
                                                                                            aln_clusters, aln_channels)

    passing_per_site_all = dict(zip(["IBL", "Steinmetz", "Allen"], [passing_per_site_ibl["passing_per_site"].to_numpy(), 
                                                                    passing_per_site_ste["passing_per_site"].to_numpy(),
                                                                    passing_per_site_aln["passing_per_site"].to_numpy()]))
    passing_per_site_all = pd.DataFrame.from_dict(passing_per_site_all, orient="index")

    mean_ibl = passing_per_site_ibl["passing_per_site"].mean()
    mean_ste = passing_per_site_ste["passing_per_site"].mean()
    mean_aln = passing_per_site_aln["passing_per_site"].mean()
    stdev_ibl = passing_per_site_ibl["passing_per_site"].std() / np.sqrt(len(passing_per_site_ibl))
    stdev_ste = passing_per_site_ste["passing_per_site"].std() / np.sqrt(len(passing_per_site_ste))
    stdev_aln = passing_per_site_aln["passing_per_site"].std() / np.sqrt(len(passing_per_site_aln))

    err_kws = {"markersize":20, 
                "linewidth":1.5}
    sns.pointplot(x="source", y="passing_per_site", data=passing_per_site_ibl, ax=axes[1, 3], join=False, markersize=2, markers=".", capsize=.2, errorbar=("se", 2), color="gray", err_kws=err_kws, zorder=100)
    sns.pointplot(x="source", y="passing_per_site", data=passing_per_site_ste, ax=axes[1, 3], join=False, markersize=2, markers=".", capsize=.2, errorbar=("se", 2), color="gray", err_kws=err_kws, zorder=100)
    sns.pointplot(x="source", y="passing_per_site", data=passing_per_site_aln, ax=axes[1, 3], join=False, markersize=2, markers=".", capsize=.2, errorbar=("se", 2), color="gray", err_kws=err_kws, zorder=100)
    sns.stripplot(passing_per_site_all.T, ax=axes[1, 3], palette=colors_translucent)
    axes[1, 3].set_title("Passing units per site", fontsize=8)
    axes[1, 3].set_xticklabels(["IBL", "STE", "ALN"], fontsize=8, rotation=90)
    axes[1, 3].set_xlabel(None)
    axes[1, 3].set_ylabel("insertion yield", fontsize=8)

    fig.suptitle(f"IBL metrics: {region}")
    fig.tight_layout()
