from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import metrics_plot, histograms, yield_detail, compute_yield
import pandas as pd

regions  = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'CNU', 'TH','HY', 'MB',]

dset = "allen"
re_yields = {}
re147_yields = {}
al_yields = {}
for region in regions:
    
    ibl_dset="re"

    ibl_clusters = load_clusters("re_147", region)

    ibl_channels = load_channels(ibl_dset, region)
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]
    
    out = metrics_plot(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    histograms(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    yield_detail(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)

    #out2 = metrics_plot(dset, region)

    if out is not None:
        mean_ibl147, mean_allen = out
        re147_yields[region] = mean_ibl147
        al_yields[region] = mean_allen

    # if out2 is not None:
    #     mean_ibl, _ = out2
    #     re_yields[region] = mean_ibl

yield_df = pd.DataFrame({"Allen": al_yields, "RE": re_yields, "RE_1.4.7": re147_yields})
ax = yield_df.plot.bar(color=["green", "pink", "red"])
ax.set_title("Yield comparison: old PyKS vs 1.4.7\nAllen/RE")
ax.set_ylabel("Yield")

yield_df["pct_change"] = (yield_df["RE_1.4.7"] - yield_df["RE"]) / (yield_df["RE"]) * 100
ax1 = yield_df["pct_change"].plot.bar(color="lightblue")
ax1.set_title("% change in yield old PyKS vs 1.4.7\nAllen/RE")
ax1.set_ylabel("% change in yield")

dset="steinmetz"
bwm_yields = {}
bwm147_yields = {}
st_yields = {}
for region in regions:
    ibl_dset="bwm"


    ibl_clusters = load_clusters("bwm_147", region)
    ibl_channels = load_channels(ibl_dset, region)
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]
    
    out = metrics_plot(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    histograms(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    yield_detail(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)

    #out2 = metrics_plot(dset, region)

    if out is not None:
        mean_ibl147, mean_ste = out
        bwm147_yields[region] = mean_ibl147
        st_yields[region] = mean_ste

    # if out2 is not None:
    #     mean_ibl, _ = out2
    #     bwm_yields[region] = mean_ibl

yield_df = pd.DataFrame({"Steinmetz": st_yields, "BWM": bwm_yields, "BWM_1.4.7": bwm147_yields})
ax = yield_df.plot.bar(color=["green", "pink", "red"])
ax.set_title("Yield comparison: old PyKS vs 1.4.7\nSteinmetz/BWM")
ax.set_ylabel("Yield")

yield_df["pct_change"] = (yield_df["BWM_1.4.7"] - yield_df["BWM"]) / (yield_df["BWM"]) * 100
ax1 = yield_df["pct_change"].plot.bar(color="lightblue")
ax1.set_title("% change in yield old PyKS vs 1.4.7\nSteinmetz/BWM")
ax1.set_ylabel("% change in yield")