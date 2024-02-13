import numpy as np
from data_quality.plots.utils import get_colors_region
from data_quality.tables import load_channels, load_clusters
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d


def metrics_plot(dset, region="Isocortex", axes=None):
    """
    :param dset: one of "steinmetz", "allen"
    :param region: Cosmos acronym of region
    :param axes: 
    """

    assert dset in ["steinmetz", "allen"], "dset must be one of 'steinemtz', 'allen'"

    if axes is None:
        fig, axes = plt.subplots(2, 4)
        
    if dset == "steinmetz":
        ibl_dset = "bwm"
    if dset == "allen":
        ibl_dset = "re"

    colors, colors_translucent = get_colors_region(region)

    clusters = load_clusters(dset, filter_region=region)
    ibl_clusters = load_clusters(ibl_dset, filter_region=region)
    channels = load_channels(dset, filter_region=region)
    ibl_channels = load_channels(dset, filter_region=region)

    ## num insertions
    num_ins = len(clusters.index.levels[0])
    num_ins_ibl = len(ibl_clusters.index.levels[0])
    axes[0,0].bar(["IBL", dset], [num_ins_ibl, num_ins], color=colors)
    axes[0,0].set_title("Number of insertions", fontsize=8)

    ## sites per insertion (in this region)
    sites = channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites_ibl = ibl_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites = pd.DataFrame.from_dict(dict(zip(["IBL", dset], [sites_ibl, sites])), orient="index")
    sns.stripplot(sites.T, ax=axes[0, 1], palette=colors)
    axes[0, 1].set_title("Sites per insertion", fontsize=8)

    ## units per insertion (in this region)
    units_per_ins_ibl = ibl_clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins = clusters.groupby(["insertion"]).count()["label"].to_numpy()
    units_per_ins_all = pd.DataFrame.from_dict(dict(zip(["IBL", dset], [units_per_ins_ibl, units_per_ins])), orient="index")
    sns.stripplot(units_per_ins_all.T, ax=axes[0, 2], palette=colors)
    axes[0, 2].set_title("Units per insertion", fontsize=8)

    ## units per site
    units_per_site_ibl = len(ibl_clusters) / len(ibl_channels)
    units_per_site = len(clusters) / len(channels)
    axes[0, 3].bar(["IBL", dset], [units_per_site_ibl, units_per_site], color=colors)
    axes[0, 3].set_title("Units per site", fontsize=8)

    ## sliding RP metric
    ibl_SRP = ibl_clusters.slidingRP_viol
    ibl_frac = sum(ibl_SRP) / len(ibl_SRP)
    SRP = clusters.slidingRP_viol
    frac = sum(SRP) / len(SRP)
    axes[1, 0].bar(["IBL", dset], [ibl_frac, frac], color=colors)
    axes[1, 0].set_title("Sliding RP", fontsize=8)
    axes[1, 0].set_ylabel("Fraction passing", fontsize=8)

    ## noise cutoff
    ibl_NC = ibl_clusters.noise_cutoff.to_numpy()
    NC = clusters.noise_cutoff.to_numpy()
    frac_ibl = sum(ibl_NC < 5.) / len(ibl_NC)
    frac = sum(NC < 5.) / len(NC)
    # for plotting
    ibl_NC = np.log10(ibl_NC)
    NC = np.log10(NC)

    bins = np.linspace(-2.5, 4.5, 100)
    h_ibl, b_ibl = np.histogram(ibl_NC, bins=bins, density=True)
    h, b = np.histogram(NC, bins=bins, density=True)
    curve_ibl = gaussian_filter1d(h_ibl, 2)
    curve = gaussian_filter1d(h, 2)
    axes[1, 1].plot(bins[:-1], curve_ibl, color=colors[0])
    axes[1, 1].plot(bins[:-1], curve, color=colors[1])
    axes[1, 1].set_xticks([-2,0,2,4])
    axes[1, 1].set_xticklabels(['0.01','1','100','10k'])
    axes[1, 1].set_yticklabels([])
    axes[1, 1].set_yticks([])
    axes[1, 1].set_title("Noise cutoff", fontsize=8)
    axes[1, 1].set_ylabel("Fraction passing", fontsize=8, y=0.7)
    axes[1, 1].vlines(
        np.log10(5. + 1. ),
        axes[1, 1].get_ylim()[0],
        axes[1, 1].get_ylim()[1],
        linestyles="dashed",
        label="noise cutoff",
        color="black",
        linewidths=1.0,
    )
    inset_NC = inset_axes(axes[1,1], width="35%", height="35%", loc="lower left")
    inset_NC.bar([1, 2], [frac_ibl, frac], color=colors_translucent)
    inset_NC.set_xticklabels([])
    inset_NC.set_xticks([])
    inset_NC.patch.set_alpha(0.2)
    inset_NC.set_frame_on(False)

    ## amplitude median
    ibl_amp = ibl_clusters.amp_median.to_numpy() * 1e6
    amp = clusters.amp_median.to_numpy() * 1e6
    frac_ibl = sum(ibl_amp > 50) / len(ibl_amp)
    frac = sum(amp > 50) / len(amp)

    bins = np.linspace(0, 600, 50)
    h_ibl, b_ibl = np.histogram(ibl_amp, bins=bins, density=True)
    h, b = np.histogram(amp, bins=bins, density=True)
    curve_ibl = gaussian_filter1d(h_ibl, 1)
    curve = gaussian_filter1d(h, 1)
    axes[1, 2].plot(bins[:-1], curve_ibl, color=colors[0])
    axes[1, 2].plot(bins[:-1], curve, color=colors[1])
    axes[1, 2].vlines(
        50,
        axes[1, 2].get_ylim()[0],
        axes[1, 2].get_ylim()[1],
        linestyles="dashed",
        label="50 uV",
        color="black",
        linewidths=1.0,
    )
    axes[1, 2].set_yticklabels([])
    axes[1, 2].set_yticks([])
    axes[1, 2].set_title("Amplitude median (uV)", fontsize=8)
    axes[1, 2].set_ylabel("Fraction passing", fontsize=8, y=0.7)
    inset_amp = inset_axes(axes[1,2], width="35%", height="35%", loc="lower left")
    inset_amp.bar([1, 2], [frac_ibl, frac], color=colors_translucent)
    inset_amp.set_xticklabels([])
    inset_amp.set_xticks([])
    inset_amp.patch.set_alpha(0.2)
    inset_amp.set_frame_on(False)






