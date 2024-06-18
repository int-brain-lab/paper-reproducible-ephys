from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import compute_yield
from data_quality.plots.utils import get_colors_region
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

regions  = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'CNU', 'TH','HY', 'MB',]
dfs = {}
for region in regions:
    print(region)
    og_clusters = load_clusters("bwm", region)
    ibl_channels = load_channels("bwm", region)

    ste_clusters = load_clusters("steinmetz", region)
    ste_channels = load_channels("steinmetz", region)

    og_yield, _ = compute_yield(og_clusters, ibl_channels, ste_clusters, ste_channels)
    og_yield.drop(columns=["lab"], inplace=True)
    og_yield.rename(columns={"passing_units":"og_nunits", 
                     "num_sites": "og_nsites",
                     "passing_per_site": "og_yield"}, inplace=True)

    rerun_clusters = load_clusters("bwm_147", region)
    
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(rerun_clusters.index.get_level_values(0).unique())]

    rerun_yield, _ = compute_yield(rerun_clusters, ibl_channels, ste_clusters, ste_channels)
    rerun_yield["region"] = region
    rerun_yield.drop(columns=["lab"], inplace=True)
    rerun_yield.rename(columns={"passing_units":"rerun_nunits", 
                     "num_sites": "rerun_nsites",
                     "passing_per_site": "rerun_yield"}, inplace=True)
    _df = rerun_yield.join(og_yield)

    dfs[region] = _df

alldf = pd.concat(dfs.values(), ignore_index=True)

fig, ax = plt.subplots()
sns.scatterplot(x="og_yield", y="rerun_yield", hue="region", data=alldf, alpha=0.4, ax=ax)
ax.set_xlabel("Original yield")
ax.set_ylabel("1.4.7 yield")
ax.set_xlim(0.0, 1.75)
ax.set_ylim(0.0, 1.75)
ax.set_title("BWM rerun yield comparison\nAll regions")
ax.plot([0,1.6], [0, 1.6], linestyle="dashed", color="gray")
ax.scatter(alldf["og_yield"].mean(), 0.03, marker="v", s=64)
ax.scatter(0.025, alldf["rerun_yield"].mean(), marker="<", s=64)
fig.tight_layout()
fig.savefig("/Users/chris/Downloads/147_yield_scatters/BWM_allregions.png")

for region, df in dfs.items():
    if len(df) == 0:
        continue
    fig, ax = plt.subplots()
    sns.scatterplot(x="og_yield", y="rerun_yield", c=get_colors_region(region)[0][0], data=df, alpha=0.8, ax=ax)
    ax.set_xlabel("Original yield")
    ax.set_ylabel("1.4.7 yield")
    ax.set_xlim(0.0, 1.75)
    ax.set_ylim(0.0, 1.75)
    ax.set_title(f"BWM rerun yield comparison\n{region}")
    ax.plot([0,1.6], [0, 1.6], linestyle="dashed", color="gray")
    ax.scatter(df["og_yield"].mean(), 0.03, marker="v", s=64)
    ax.scatter(0.025, df["rerun_yield"].mean(), marker="<", s=64)
    fig.tight_layout()
    fig.savefig(f"/Users/chris/Downloads/147_yield_scatters/BWM_{region}.png")

# change in yield
alldf = pd.concat(dfs.values())
alldf["yield_change"] = (alldf["rerun_yield"] - alldf["og_yield"])/(alldf["og_yield"])*100
from data_quality.tables import tables_dir
bwm_wm = pd.read_csv(tables_dir.joinpath("wm_cond_rerun_BWM.csv"))
bwm_wm["cond_change"] = (bwm_wm["cond_no_rerun"] - bwm_wm["cond_no_original"]) / bwm_wm["cond_no_original"] * 100
bwm_wm.rename(columns={"Unnamed: 0":"pid"}, inplace=True)
alldf["pid"] = alldf.index
comb_df = pd.merge(alldf, bwm_wm)

fig, ax = plt.subplots()
sns.scatterplot(x="cond_change", y="yield_change", data=comb_df, ax=ax, hue="region", alpha=0.5)
ax.set_xlabel("condition # change (%)")
ax.set_ylabel("yield change (%)")
fig.suptitle("BWM Condition No. change vs yield change")

fig, ax = plt.subplots()
sns.scatterplot(x="cond_change", y="yield_change", data=comb_df[comb_df.cond_change < 5000], ax=ax, hue="region", alpha=0.5)
ax.set_xlabel("condition # change (%)")
ax.set_ylabel("yield change (%)")
fig.suptitle("BWM Condition No. change vs yield change\nExcl. outliers")
