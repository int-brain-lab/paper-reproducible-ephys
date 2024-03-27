from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import compute_yield
from data_quality.plots.utils import get_colors_region
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

regions  = ['Isocortex', 'TH', 'HPF']

dfs = []
for region in regions:
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
    re_yield["dataset"] = "IBL"
    
    al_yield["region"] = region
    al_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    al_yield["dataset"] = "Allen"
    
    st_yield["region"] = region
    st_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    st_yield["dataset"] = "Steinmetz"

    dfs.append(re_yield)
    dfs.append(al_yield)
    dfs.append(st_yield)

df = pd.concat(dfs)

# the below code is nice but does not play nice with error bars
# fig = sns.catplot(data=df, 
#             x="dataset", order=["Allen", "Steinmetz", "IBL"],
#             col_order=["Isocortex", "TH", "HPF"],
#             y="yield", col="region",
#             height=4, aspect=.6, hue="dataset",
#             errorbar=("ci", "se"),
#             units="yield",
#             legend=False,
#             zorder=-1
# )
# fig.set_axis_labels("", "mean passing units per site")
# fig.set_titles("{col_name}")
# fig.map(plt.errorbar, "dataset", "yield", "std",
#         linestyle='none', marker="o")

fig, ax = plt.subplots(1, 3)
for i, region in enumerate(regions):
    sns.stripplot(data=df[df.region==region], x="dataset", 
                y="yield", ax = ax[i], hue="dataset", zorder=-1,
                alpha=0.6, order=["IBL", "Steinmetz", "Allen"])

    err_kws = {"markersize":20, 
                "linewidth":1.5}
    sns.pointplot(data=df[df.region==region],x="dataset", y="yield", 
                ax=ax[i], markersize=2, markers="none", capsize=.2, 
                errorbar=("se", 1), color="black", err_kws=err_kws,
                linestyle="none", order=["IBL", "Steinmetz", "Allen"])
    ax[i].set_title(region)
    ax[i].set_xlabel(None)
    if i != 0:
        ax[i].set_ylabel(None)
        ax[i].set_yticks([])
    else:
        ax[i].set_ylabel("mean passing units per site")

fig.tight_layout()

# 2-way ANOVA on region and dataset
df = df.drop(columns=["nunits", "nsites"])
dataset_map = {"IBL":1, "Steinmetz":2, "Allen":3}
region_map = {"Isocortex":1, "TH":2, "HPF":3}
df = df.replace({"region":region_map,
                 "dataset":dataset_map})

import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

model = ols('Q("yield") ~ C(Q("dataset")) + C(Q("region")) + C(Q("dataset")):C(Q("region"))',
            df).fit()
sm.stats.anova_lm(model, typ=2)

dataset_effect = sp.posthoc_conover(df, val_col="yield", group_col="dataset", p_adjust="holm")
region_effect = sp.posthoc_conover(df, val_col="yield", group_col="dataset", p_adjust="holm")