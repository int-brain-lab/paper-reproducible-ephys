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
    og_clusters = load_clusters("re", region)
    og_clusters = og_clusters[og_clusters.firing_rate > 0.1]
    ibl_channels = load_channels("re", region)

    al_clusters = load_clusters("allen", region)
    al_channels = load_channels("allen", region)

    og_yield, _ = compute_yield(og_clusters, ibl_channels, al_clusters, al_channels)
    og_yield.drop(columns=["lab"], inplace=True)
    og_yield.rename(columns={"passing_units":"og_nunits", 
                     "num_sites": "og_nsites",
                     "passing_per_site": "og_yield"}, inplace=True)

    rerun_clusters = load_clusters("re_147", region)
    rerun_clusters = rerun_clusters[rerun_clusters.firing_rate > 0.1]
    
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(rerun_clusters.index.get_level_values(0).unique())]

    rerun_yield, _ = compute_yield(rerun_clusters, ibl_channels, al_clusters, al_channels)
    rerun_yield["region"] = region
    rerun_yield.drop(columns=["lab"], inplace=True)
    rerun_yield.rename(columns={"passing_units":"rerun_nunits", 
                     "num_sites": "rerun_nsites",
                     "passing_per_site": "rerun_yield"}, inplace=True)
    _df = rerun_yield.join(og_yield)

    dfs[region] = _df

alldf = pd.concat(dfs.values(), ignore_index=True)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x="og_yield", y="rerun_yield", hue="region", data=alldf, alpha=0.4, ax=ax,
)
ax.set_xlabel("Original yield")
ax.set_ylabel("1.4.7 yield")
# ax.set_xlim(0.0, 1.5)
# ax.set_ylim(0.0, 1.5)
ax.set_title("RE yield comparison")
ax.plot([0,1.6], [0, 1.6], linestyle="dashed", color="gray")
ax.scatter(alldf["og_yield"].mean(), 0.03, marker="v", s=64)
ax.scatter(0.025, alldf["rerun_yield"].mean(), marker="<", s=64)
ax.margins(x=0, y=0)
ax.axis("square")
fig.tight_layout()
fig.savefig("/Users/chris/Downloads/147_yield_scatters/RE_spikecount_allregions.png")

for region, df in dfs.items():
    if len(df) == 0:
        continue
    fig, ax = plt.subplots()
    sns.scatterplot(x="og_yield", y="rerun_yield", c=get_colors_region(region)[0][0], data=df, alpha=0.8, ax=ax)
    ax.set_xlabel("Original yield")
    ax.set_ylabel("1.4.7 yield")
    ax.set_xlim(0.0, 1.75)
    ax.set_ylim(0.0, 1.75)
    ax.set_title(f"RE rerun yield comparison\n{region}")
    ax.plot([0,1.6], [0, 1.6], linestyle="dashed", color="gray")
    ax.scatter(df["og_yield"].mean(), 0.03, marker="v", s=64)
    ax.scatter(0.025, df["rerun_yield"].mean(), marker="<", s=64)
    fig.tight_layout()
    fig.savefig(f"/Users/chris/Downloads/147_yield_scatters/RE_{region}.png")

ax.axis("equal")