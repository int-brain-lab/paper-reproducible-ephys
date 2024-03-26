from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import compute_yield
from data_quality.plots.utils import get_colors_region
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

regions  = ['Isocortex', 'TH', 'HPF']

by_region = {}
for region in regions:
    by_region[region] = {}

    ibl_channels = load_channels("re", region)
    ibl_clusters = load_clusters("re_147", region)
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]

    al_clusters = load_clusters("allen", region)
    al_channels = load_channels("allen", region)

    st_clusters = load_clusters("steinmetz", region)
    st_channels = load_channels("steinmetz", region)

    re_yield, al_yield = compute_yield(ibl_clusters, ibl_channels, al_clusters, al_channels)
    _, st_yield = compute_yield(ibl_clusters, ibl_channels, st_clusters, st_channels)

    re_yield["region"] = region
    re_yield.drop(columns=["lab"], inplace=True)
    re_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    
    al_yield["region"] = region
    al_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    
    st_yield["region"] = region
    st_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    
    by_region[region]["re"] = re_yield
    by_region[region]["allen"] = al_yield
    by_region[region]["steinmetz"] = st_yield

dsets = ["allen", "steinmetz", "re"]
ctx_mean = {}
ctx_std = {}
for d in dsets:
    ctx_mean[d] = by_region["Isocortex"][d]["yield"].mean()
    ctx_std[d] = by_region["Isocortex"][d]["yield"].std() / np.sqrt(len(by_region["Isocortex"][d]))

th_mean = {}
th_std = {}
for d in dsets:
    th_mean[d] = by_region["TH"][d]["yield"].mean()
    th_std[d] = by_region["TH"][d]["yield"].std() / np.sqrt(len(by_region["TH"][d]))
    
hpf_mean = {}
hpf_std = {}
for d in dsets:
    hpf_mean[d] = by_region["HPF"][d]["yield"].mean()
    hpf_std[d] = by_region["HPF"][d]["yield"].std() / np.sqrt(len(by_region["HPF"][d]))

# bar plot

x_axis = np.arange(len(regions))

error_kw=dict(ecolor='gray', lw=2, capsize=5, capthick=2)
plt.bar(x_axis, [ctx_mean["allen"], th_mean["allen"], hpf_mean["allen"]], 0.2, 
        yerr=[ctx_std["allen"], th_std["allen"], hpf_std["allen"]], 
        label="Allen", error_kw=error_kw)

plt.bar(x_axis + .2, [ctx_mean["steinmetz"], th_mean["steinmetz"], hpf_mean["steinmetz"]], 0.2, 
        yerr=[ctx_std["steinmetz"], th_std["steinmetz"], hpf_std["steinmetz"]], 
        label= "Steinmetz", error_kw=error_kw)

plt.bar(x_axis + .4, [ctx_mean["re"], th_mean["re"], hpf_mean["re"]], 0.2, 
        yerr=[ctx_std["re"], th_std["re"], hpf_std["re"]], 
        label="IBL", error_kw=error_kw)
plt.xticks(x_axis + .2, regions)

plt.legend()
plt.title("Mean yield per region")
plt.ylabel("mean yield")
plt.xlabel("Brain region")

