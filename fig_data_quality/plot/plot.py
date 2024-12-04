import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import colorsys

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter1d
from iblatlas.regions import BrainRegions

from fig_data_quality.tables import load_channels, load_clusters

## Utilities

def scale_lightness(rgb, scale_l):
    """
    Given an RGB color vector, return with lightness
    scaled by `scale_l`.
    """
    # convert rgb to hls
    h, l, s = colorsys.rgb_to_hls(*rgb)
    # manipulate h, l, s values and return as rgb
    return colorsys.hls_to_rgb(h, min(1, l * scale_l), s = s)

def get_colors_region(region):
    """
    Get standard atlas colors for given Cosmos region.

    :param region: Cosmos acronym
    :returns: colors, colors_translucent
    """
    br = BrainRegions()
    region_idx = br.acronym2index([region])[1][0]
    region_rgb = br.rgb[region_idx][0]
    colors = [region_rgb / 255., scale_lightness(region_rgb/255., 1.2)]
    colors_translucent = [np.array(list(colors[0]) + [0.75]), np.array(list(colors[1]) + [0.75])]
    return colors, colors_translucent

def get_3colors_region(region, blue=True):
    """
    Get 3 increasingly lighter shades of standard atlas color for a Cosmos region.

    :param region: Cosmos acronym
    :returns: colors, colors_translucent (both 3-vectors)
    """

    if region == 'Isocortex' and blue:
        region = 'VISam'
    #     colors = [np.array([95, 100, 184]) / 255 , np.array([160, 180, 226]) / 255, np.array([211, 219, 242]) / 255]
    # else:
    br = BrainRegions()
    region_idx = br.acronym2index([region])[1][0]
    region_rgb = br.rgb[region_idx][0]
    colors = [region_rgb / 255., scale_lightness(region_rgb/255., 1.15), scale_lightness(region_rgb/255., 1.3)]
    colors_translucent = [np.array(list(colors[0]) + [0.75]), np.array(list(colors[1]) + [0.75]), np.array(list(colors[2]) + [0.75])]
    return colors, colors_translucent

## Plotting functions (all deprecated)

def metrics_plot(dset, region="Isocortex", axes=None, ibl_clusters=None, ibl_channels=None, clusters=None, channels=None):
    """
    -- Deprecated function --
    :param dset: one of "steinmetz", "allen"
    :param region: Cosmos acronym of region
    :param axes: matplotlib Axes
    :param ibl_clusters: IBL clusters DF
    :param ibl_channels: IBL channels DF
    :param clusters: non-IBL clusters DF
    :param channels: non-IBL clusters DF
    """
    assert dset in ["steinmetz", "allen"], "dset must be one of 'steinemtz', 'allen'"
        
    if dset == "steinmetz":
        ibl_dset = "bwm"
    if dset == "allen":
        ibl_dset = "re"

    colors, colors_translucent = get_colors_region(region)

    if clusters is None:
        clusters = load_clusters(dset, filter_region=region)
    if ibl_clusters is None:
        ibl_clusters = load_clusters(ibl_dset, filter_region=region)
    if channels is None:
        channels = load_channels(dset, filter_region=region)
    if ibl_channels is None:
        ibl_channels = load_channels(ibl_dset, filter_region=region)

    if len(ibl_clusters) == len(clusters) == 0:
        print(f"[metrics_plot] No IBL or {dset} clusters in {region}")
        return
    elif len(ibl_clusters) == 0:
        print(f"[metrics_plot] No IBL clusters in {region}")
        return
    elif len(clusters) == 0:
        print(f"[metrics_plot] No {dset} clusters in {region}")
        return

    if axes is None:
        fig, axes = plt.subplots(2, 4)

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

    ## passing units per site
    passing_per_site_ibl, passing_per_site = compute_yield(ibl_clusters, ibl_channels, clusters, channels, dset)

    passing_per_site_all = dict(zip(["IBL", dset], [passing_per_site_ibl["passing_per_site"].to_numpy(), passing_per_site["passing_per_site"].to_numpy()]))
    passing_per_site_all = pd.DataFrame.from_dict(passing_per_site_all, orient="index")

    mean_ibl = passing_per_site_ibl["passing_per_site"].mean()
    mean = passing_per_site["passing_per_site"].mean()
    stdev_ibl = passing_per_site_ibl["passing_per_site"].std() / np.sqrt(len(passing_per_site_ibl))
    stdev = passing_per_site["passing_per_site"].std() / np.sqrt(len(passing_per_site))

    axes[1, 3].bar(["IBL", dset], [mean_ibl, mean], yerr=[stdev_ibl], capsize=5, ecolor="gray", color=colors)
    axes[1, 3].set_title("Passing units per site", fontsize=8)

    fig.suptitle(f"IBL metrics: {region}")
    fig.tight_layout()
    

    return mean_ibl, mean

def histograms(dset, region="Isocortex", ibl_clusters=None, ibl_channels=None, clusters=None, channels=None):
    """
    -- Deprecated function --
    """
    
    assert dset in ["steinmetz", "allen"], "dset must be one of 'steinemtz', 'allen'"
    if dset == "steinmetz":
        ibl_dset = "bwm"
    if dset == "allen":
        ibl_dset = "re"

    colors, colors_translucent = get_colors_region(region)

    if clusters is None:
        clusters = load_clusters(dset, filter_region=region)
    if ibl_clusters is None:
        ibl_clusters = load_clusters(ibl_dset, filter_region=region)
    if channels is None:
        channels = load_channels(dset, filter_region=region)
    if ibl_channels is None:
        ibl_channels = load_channels(ibl_dset, filter_region=region)

    if len(ibl_clusters) == len(clusters) == 0:
        print(f"[histograms] No IBL or {dset} clusters in {region}")
        return
    elif len(ibl_clusters) == 0:
        print(f"[histograms] No IBL clusters in {region}")
        return
    elif len(clusters) == 0:
        print(f"[histograms] No {dset} clusters in {region}")
        return

    # used for yield computation below
    sites_ibl = ibl_channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()
    sites = channels.groupby(["insertion"]).count()["cosmos_acronym"].to_numpy()

    # convert from V to uV
    clusters["amp_median"] *= 1e6
    ibl_clusters["amp_median"] *= 1e6
    
    fig, axf = plt.subplot_mosaic([["upper left", "upper right"], ["bottom", "bottom"], ], figsize=(9, 6))
    ibl_passing = ibl_clusters[(ibl_clusters.label==1.0)]
    ibl_clusters.plot.scatter(x="firing_rate", y="amp_median", loglog=True, title=f"IBL", ax=axf["upper left"], color="grey", alpha=0.5, label="all")
    ibl_passing.plot.scatter(x="firing_rate", y="amp_median", loglog=True, ax=axf["upper left"], color=colors[0], alpha=0.5, label="passing")
    axf["upper left"].set_xlabel("log(FR) (Hz)", fontsize=8)
    axf["upper left"].set_ylabel("log(amp_median) (uV)", fontsize=8)
    
    passing = clusters[(clusters.label==1.0)&(clusters.use==1.0)]
    clusters.plot.scatter(x="firing_rate", y="amp_median", loglog=True, title=dset, ax=axf["upper right"], color="grey", alpha=0.5, label="all")
    passing.plot.scatter(x="firing_rate", y="amp_median", loglog=True, ax=axf["upper right"], color=colors[1], alpha=0.5, label="passing")
    axf["upper right"].set_xlabel("log(FR) (Hz)", fontsize=8)
    axf["upper right"].set_ylabel("log(amp_median) (uV)", fontsize=8)
    
    fr_xmax = max(ibl_clusters.firing_rate.max(), ibl_clusters.firing_rate.max())
    fr_xmin = min(ibl_clusters.firing_rate.min(), ibl_clusters.firing_rate.min())
    amp_ymax = max(clusters.amp_median.max(), clusters.amp_median.max())
    amp_ymin = min(clusters.amp_median.min(), clusters.amp_median.min())

    axf["upper left"].set_xlim(fr_xmin, fr_xmax)
    axf["upper left"].set_ylim(amp_ymin, amp_ymax)
    axf["upper right"].set_xlim(fr_xmin, fr_xmax)
    axf["upper right"].set_ylim(amp_ymin, amp_ymax)
    
    
    bins = np.logspace(0, np.log10(2000), 50)
    ampassing_ibl = ibl_clusters[["amp_median", "label"]][ibl_clusters.label==1.]
    ampassing_ibl["binned"] = pd.cut(ampassing_ibl["amp_median"], bins=bins) 
    ampassing_ibl["start"] = [interval.left if not isinstance(interval, float) else bins[-1] for interval in ampassing_ibl["binned"] ]
    to_plot = ampassing_ibl.groupby("start").agg("count")
    to_plot["cumu"] = to_plot.loc[::-1, "binned", ].cumsum()[::-1] / sum(sites_ibl)
    to_plot["cumu"].plot(drawstyle="steps", color=colors[0], ax=axf["bottom"])
    
    ampassing = clusters[["amp_median", "label"]][clusters.label==1.]
    ampassing["binned"] = pd.cut(ampassing["amp_median"], bins=bins) 
    ampassing["start"] = [interval.left for interval in ampassing["binned"]]
    to_plot = ampassing.groupby("start").agg("count")
    to_plot["cumu"] = to_plot.loc[::-1, "binned", ].cumsum()[::-1] / sum(sites)
    to_plot["cumu"].plot(drawstyle="steps", color=colors[1], ax=axf["bottom"])
    axf["bottom"].set_title("Passing units per site >= amplitude", fontsize=10)
    axf["bottom"].set_xscale("log")
    axf["bottom"].set_xlabel("log(amp_median) (uV)", fontsize=8)
    axf["bottom"].set_ylabel("Passing units per site\n>=amp", fontsize=8)
    
    fig.suptitle(f"Firing rate vs Amplitude {region}")
    fig.tight_layout()
    
    fig, axh = plt.subplots(1, 2, figsize=(10, 4))
    num_units_ibl = ibl_clusters.index.get_level_values(0).nunique()
    num_units_ste = clusters.index.get_level_values(0).nunique()
    nbins = int((num_units_ibl + num_units_ste))
    bins = np.logspace(0, np.log10(2000), num=nbins)
    hist_kwargs = {"bins":bins, "histtype":"step"}
    ibl_clusters["amp_median"].hist(ax=axh[0], linestyle="dashed", edgecolor="grey", label="IBL-all", **hist_kwargs)
    ibl_passing["amp_median"].hist(ax=axh[0], edgecolor="grey", label="IBL-passing",**hist_kwargs)
    clusters["amp_median"].hist(ax=axh[0], linestyle="dashed", edgecolor=colors[1], label=f"{dset}-all", **hist_kwargs)
    passing["amp_median"].hist(ax=axh[0], edgecolor=colors[1], label=f"{dset}-passing", **hist_kwargs)
    axh[0].set_xlim(1, 2000)
    axh[0].set_xlabel("log(amp_median) (uV)", fontsize=8)
    axh[0].set_ylabel("# units")
    axh[0].set_title(f"Amplitude histogram {region}", fontsize=10)
    axh[0].set_xscale("log")
    axh[0].grid(False)
    axh[0].vlines(50,
        axh[0].get_ylim()[0],
        axh[0].get_ylim()[1],
        linestyles="dashed",
        label="50 uV",
        color="black",
        linewidths=2.0,)
        
    bins = np.logspace(-3, 3, nbins)
    hist_kwargs = {"bins":bins, "histtype":"step"}
    ibl_clusters["firing_rate"].hist(ax=axh[1], linestyle="dashed", edgecolor="grey", label="IBL-all", **hist_kwargs)
    ibl_passing["firing_rate"].hist(ax=axh[1], edgecolor="grey", label="IBL-passing",**hist_kwargs)
    clusters["firing_rate"].hist(ax=axh[1], linestyle="dashed", edgecolor=colors[1], label=f"{dset}-all", **hist_kwargs)
    passing["firing_rate"].hist(ax=axh[1], edgecolor=colors[1], label=f"{dset}-passing", **hist_kwargs)
    axh[1].set_xscale("log")
    axh[1].set_xlabel("log(FR) (Hz)", fontsize=8)
    axh[1].set_ylabel("")
    axh[1].legend()
    axh[1].grid(False)
    axh[1].set_title(f"FR histogram {region}", fontsize=12)
    fig.tight_layout()

    fig.suptitle(f"Histograms: {region}")

def yield_detail(dset, region="Isocortex",ibl_clusters=None, ibl_channels=None, clusters=None, channels=None):
    """
    -- Deprecated function --
    """
    
    assert dset in ["steinmetz", "allen"], "dset must be one of 'steinemtz', 'allen'"

    if dset == "steinmetz":
        ibl_dset = "bwm"
    if dset == "allen":
        ibl_dset = "re"

    colors, colors_translucent = get_colors_region(region)

    if clusters is None:
        clusters = load_clusters(dset, filter_region=region)
    if ibl_clusters is None:
        ibl_clusters = load_clusters(ibl_dset, filter_region=region)
    if channels is None:
        channels = load_channels(dset, filter_region=region)
    if ibl_channels is None:
        ibl_channels = load_channels(ibl_dset, filter_region=region)

    if len(ibl_clusters) == len(clusters) == 0:
        print(f"[yield_detail] No IBL or {dset} clusters in {region}")
        return
    elif len(ibl_clusters) == 0:
        print(f"[yield_detail] No IBL clusters in {region}")
        return
    elif len(clusters) == 0:
        print(f"[yield_detail] No {dset} clusters in {region}")
        return
        
    lab_hue_order = sorted(list(ibl_clusters.lab.unique()))

    sites_ibl = ibl_channels.groupby(["insertion"]).count()["cosmos_acronym"]
    sites = channels.groupby(["insertion"]).count()["cosmos_acronym"]

    fig, axd = plt.subplot_mosaic([['left', 'upper right'],
                                   ['left', 'lower right']])

    ## passing units per site
    passing_per_site_ibl, passing_per_site = compute_yield(ibl_clusters, ibl_channels, clusters, channels, dset)
    passing_per_site_all = dict(zip(["IBL", dset], [passing_per_site_ibl["passing_per_site"].to_numpy(), passing_per_site["passing_per_site"].to_numpy()]))
    passing_per_site_all = pd.DataFrame.from_dict(passing_per_site_all, orient="index")

    mean_ibl = passing_per_site_ibl["passing_per_site"].mean()  
    mean = passing_per_site["passing_per_site"].mean()
    stdev_ibl = passing_per_site_ibl["passing_per_site"].std() / np.sqrt(len(passing_per_site_ibl))
    stdev = passing_per_site["passing_per_site"].std() / np.sqrt(len(passing_per_site))

    sns.stripplot(x="source", y="passing_per_site", data=passing_per_site_ibl, hue="lab", hue_order=lab_hue_order, ax=axd["left"])
    sns.stripplot(x="source", y="passing_per_site", data=passing_per_site, color=colors[1], ax=axd["left"])
    
    err_kws = {"markersize":20, 
               "linewidth":1.5}
    sns.pointplot(x="source", y="passing_per_site", data=passing_per_site_ibl, ax=axd["left"], join=False, markersize=10, markers="_", capsize=None, errorbar=("se", 2), color="gray", err_kws=err_kws)
    sns.pointplot(x="source", y="passing_per_site", data=passing_per_site, ax=axd["left"], join=False, markersize=10, markers="_", capsize=None, errorbar=("se", 2), color="gray", err_kws=err_kws)
    axd["left"].set_ylabel("passing units per site")
    axd["left"].legend(loc='upper center', bbox_to_anchor=(0.5, -.05), ncol=3)
    axd["left"].set_xlabel("")

    sns.kdeplot(passing_per_site_all.loc["IBL"], ax=axd["upper right"], color=colors[0])
    # ibl mean and se
    datax = axd["upper right"].lines[0].get_xdata()
    datay = axd["upper right"].lines[0].get_ydata()
    left = mean_ibl - stdev_ibl
    right = mean_ibl + stdev_ibl
    axd["upper right"].vlines(mean_ibl, 0, np.interp(mean_ibl, datax, datay), ls=":", color="black")
    axd["upper right"].fill_between(datax, 0, datay, where=(left <= datax) & (datax <= right), interpolate=True, alpha=0.5, color=colors[0])
    axd["upper right"].set_ylabel(None)
    axd["upper right"].set_xlabel("passing units per site")
    axd["upper right"].set_title("IBL")
    axd["upper right"].text(0.6, 0.8, f"Mean: {round(mean_ibl, 3)}", transform=axd["upper right"].transAxes, fontsize=14)
    axd["upper right"].set_xlim(0., 0.8)

    sns.kdeplot(passing_per_site_all.loc[dset], ax=axd["lower right"], color=colors[0])
    # alleninmetz mean and se
    datax = axd["lower right"].lines[0].get_xdata()
    datay = axd["lower right"].lines[0].get_ydata()
    left = mean - stdev
    right = mean + stdev
    axd["lower right"].vlines(mean, 0, np.interp(mean, datax, datay), ls=":", color="black")
    axd["lower right"].fill_between(datax, 0, datay, where=(left <= datax) & (datax <= right), interpolate=True, alpha=0.5, color=colors[0])
    axd["lower right"].set_ylabel(None)
    axd["lower right"].set_xlabel("passing units per site")
    axd["lower right"].set_title(dset)
    axd["lower right"].text(0.6, 0.8, f"Mean: {round(mean, 3)}", transform=axd["lower right"].transAxes, fontsize=14)
    axd["lower right"].set_xlim(0., 0.8)

    

    # statistical tests
    from scipy import stats

    tt = stats.ttest_ind(passing_per_site_all.loc["IBL"].dropna().to_list(), passing_per_site_all.loc[dset].dropna().to_list())

    ks = stats.kstest(passing_per_site_all.loc["IBL"].dropna().to_list(), passing_per_site_all.loc[dset].dropna().to_list())

    box_str = f"T-test:\nStatistic: {round(tt.statistic, 5)}\np-Value: {round(tt.pvalue, 5)}\n"
    box_str += f"\nKS-test:\nStatistic: {round(ks.statistic, 5)}\np-Value: {round(ks.pvalue, 5)}"

    box = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axd["left"].text(0.6, 0.95, box_str, transform=axd["left"].transAxes, fontsize=10, verticalalignment='top', bbox=box)
    fig.suptitle(f"Yield detail: {region}")
    #fig.tight_layout()

def compute_yield(ibl_clusters, ibl_channels, clusters, channels, dset=None):
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

    passing = clusters.groupby("insertion").agg(passing_units = pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x==1.])))
    sites = channels.groupby("insertion").agg(num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count"))
    passing_per_site = passing.merge(sites, on="insertion")

    passing_per_site_ibl["passing_per_site"] = passing_per_site_ibl["passing_units"]/passing_per_site_ibl["num_sites"]
    passing_per_site["passing_per_site"] = passing_per_site["passing_units"]/passing_per_site["num_sites"]

    if dset:
        passing_per_site_ibl["source"] = "IBL"
        passing_per_site["source"] = dset

    return passing_per_site_ibl, passing_per_site
    
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

    all_ibl = ibl_clusters.groupby("insertion").count()["label"].to_numpy()
    all_ste = ste_clusters.groupby("insertion").count()["label"].to_numpy()
    all_aln = aln_clusters.groupby("insertion").count()["label"].to_numpy()

    passing_ste = ste_clusters.groupby("insertion").agg(passing_units = pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x==1.])))
    sites_ste = ste_channels.groupby("insertion").agg(num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count"))
    passing_per_site_ste = passing_ste.merge(sites_ste, on="insertion")

    passing_aln = aln_clusters.groupby("insertion").agg(passing_units = pd.NamedAgg(column="label", aggfunc=lambda x: len(x[x==1.])))
    sites_aln = aln_channels.groupby("insertion").agg(num_sites=pd.NamedAgg(column="cosmos_acronym", aggfunc="count"))
    passing_per_site_aln = passing_aln.merge(sites_aln, on="insertion")

    passing_per_site_ibl["passing_per_site"] = passing_per_site_ibl["passing_units"]/passing_per_site_ibl["num_sites"]
    passing_per_site_ste["passing_per_site"] = passing_per_site_ste["passing_units"]/passing_per_site_ste["num_sites"]
    passing_per_site_aln["passing_per_site"] = passing_per_site_aln["passing_units"]/passing_per_site_aln["num_sites"]

    passing_per_site_ibl["all_units"] = all_ibl
    passing_per_site_ste["all_units"] = all_ste
    passing_per_site_aln["all_units"] = all_aln


    passing_per_site_ibl["source"] = "IBL"
    passing_per_site_ste["source"] = "Steinmetz"
    passing_per_site_aln["source"] = "Allen"

    return passing_per_site_ibl, passing_per_site_ste, passing_per_site_aln
