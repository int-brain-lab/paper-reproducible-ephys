from data_quality.tables import load_clusters, load_channels
from data_quality.plots.metrics import compute_yield, compute_yield_threeway
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

og_clusters = load_clusters("re")
ibl_clusters = load_clusters("re_147")
aln_clusters = load_clusters("allen")
ste_clusters = load_clusters("steinmetz")

ibl_channels = load_channels("re")
aln_channels = load_channels("allen")
ste_channels = load_channels("steinmetz")

ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]

# overall mean yield
re_yield, aln_yield = compute_yield(ibl_clusters, ibl_channels, aln_clusters, aln_channels)
_, ste_yield = compute_yield(ibl_clusters, ibl_channels, ste_clusters, ste_channels)
og_yield, _ = compute_yield(og_clusters, ibl_channels, ste_clusters, ste_channels)


mean_ibl_yield = re_yield["passing_per_site"].mean()
mean_aln_yield = aln_yield["passing_per_site"].mean()
mean_ste_yield = ste_yield["passing_per_site"].mean()
mean_og_yield = og_yield["passing_per_site"].mean()

print(mean_ibl_yield, mean_aln_yield, mean_ste_yield, mean_og_yield)

# yield across regs, before rerun 
# by Cosmos region
regions  = ['Isocortex', 'TH', 'HPF']
before_dfs = []
after_dfs = []
for region in regions:
    og_clusters = load_clusters("re", region)
    aln_clusters = load_clusters("allen", region)
    ste_clusters = load_clusters("steinmetz", region)
    new_clusters = load_clusters("re_147", region)

    og_channels = load_channels("re", region)
    aln_channels = load_channels("allen", region)
    ste_channels = load_channels("steinmetz", region)
    new_channels = load_channels("re", region)

    og_channels = og_channels.loc[list(og_clusters.index.get_level_values(0).unique())]
    new_channels = new_channels.loc[list(new_clusters.index.get_level_values(0).unique())]

    og_yield, al_yield = compute_yield(og_clusters, og_channels, aln_clusters, aln_channels)
    new_yield, st_yield = compute_yield(new_clusters, new_channels, ste_clusters, ste_channels)
    

    og_yield["region"] = region
    og_yield.drop(columns=["lab"], inplace=True)
    og_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    og_yield["dataset"] = "IBL"

    new_yield["region"] = region
    new_yield.drop(columns=["lab"], inplace=True)
    new_yield.rename(columns={"passing_units":"nunits", 
                     "num_sites": "nsites",
                     "passing_per_site": "yield"}, inplace=True)
    new_yield["dataset"] = "IBL"
    
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

    before_dfs.append(og_yield)
    before_dfs.append(al_yield)
    before_dfs.append(st_yield)

    after_dfs.append(new_yield)
    after_dfs.append(al_yield)
    after_dfs.append(st_yield)


before_df = pd.concat(before_dfs)
after_df = pd.concat(after_dfs)


fontsize = 16.
plt.rcParams["axes.labelsize"] = fontsize
fig, ax = plt.subplots(1, 3, figsize=(12, 5))
region_fullname = {
    "Isocortex": "Cortex",
    "TH": "Thalamus",
    "HPF": "Hippocampus"
}
err_kws = {
    "markersize": 20, 
    "linewidth": 1.0
}

for i, region in enumerate(regions):
    b = sns.boxplot(data=before_df[before_df.region==region], x="dataset", 
                y="yield", ax = ax[i], zorder=-1,
                order=["IBL", "Steinmetz", "Allen"],
                palette={"IBL":"tab:red",
                         "Allen":"tab:blue",
                         "Steinmetz":"tab:blue"})

    ax[i].set_title(region_fullname[region], fontsize=fontsize)
    ax[i].set_xlabel(None)
    tx = ax[i].get_xticks()
    ax[i].set_xticks(tx, ["IBL", "STE", "ALN"], fontsize=fontsize)

    ax[i].set_ylim(-0.1, 1.25)

    if i != 0:
        ax[i].set_ylabel(None)
        ax[i].set_yticks([])
        sns.despine(ax=ax[i], left=True) 
        
    else:
        ax[i].set_ylabel("Yield", fontsize=fontsize)
        ax[i].set_yticks([0., 0.5, 1., 1.5])
        sns.despine(ax=ax[i], trim=True) 
        ty = ax[i].get_yticks()
        ly = ax[i].get_yticklabels()
        ax[i].set_yticks(ty, ly, fontsize=fontsize)


fig, ax = plt.subplots(1, 3, figsize=(12, 5))

for i, region in enumerate(regions):
    b = sns.boxplot(data=after_df[after_df.region==region], x="dataset", 
                y="yield", ax = ax[i], zorder=-1,
                order=["IBL", "Steinmetz", "Allen"],
                palette={"IBL":"tab:green",
                         "Allen":"tab:blue",
                         "Steinmetz":"tab:blue"})

    ax[i].set_title(region_fullname[region], fontsize=fontsize)
    ax[i].set_xlabel(None)
    tx = ax[i].get_xticks()
    ax[i].set_xticks(tx, ["IBL", "STE", "ALN"], fontsize=fontsize)

    ax[i].set_ylim(-0.1, 1.25)

    if i != 0:
        ax[i].set_ylabel(None)
        ax[i].set_yticks([])
        sns.despine(ax=ax[i], left=True) 
        
    else:
        ax[i].set_ylabel("Yield", fontsize=fontsize)
        ax[i].set_yticks([0., 0.5, 1., 1.5])
        sns.despine(ax=ax[i], trim=True) 
        ty = ax[i].get_yticks()
        ly = ax[i].get_yticklabels()
        ax[i].set_yticks(ty, ly, fontsize=fontsize)