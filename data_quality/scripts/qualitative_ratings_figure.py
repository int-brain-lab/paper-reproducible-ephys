from data_quality.tables import tables_dir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv(tables_dir.joinpath("double_blind_results.csv"))
results = results.loc[:100]

def jitter(values,j):
    return values + np.random.normal(j,0.1,values.shape)

fig, ax = plt.subplots(1, 3, figsize=(8, 5))

err_kws = {
    "markersize": 20, 
    "linewidth": 1.0
}

raters = ["GC", "NS", "FD"]

for i, rater in enumerate(raters):

    sns.pointplot(
    x="Source", y=rater, data=results, ax=ax[i],
    markers="none", capsize=.2, errorbar=("se", 1), color="black", err_kws=err_kws,
    linestyle="none", order=["IBL", "Steinmetz", "Allen"], zorder=100
    )
    sns.stripplot(
        x="Source", y=jitter(results[rater], 1), 
        data=results, ax=ax[i], alpha=0.6, size=5.,
        order=["IBL", "Steinmetz", "Allen"])
    ax[i].set_xlabel(None)
    ax[i].set_ylabel(None)
    ax[i].set_title(rater)
    ax[i].set_ylim(0, 10.5)
    tx = ax[i].get_xticks()
    ax[i].set_xticks(tx, ["IBL", "STE", "ALN"], fontsize=12)

    if i == 0:
        ax[i].set_ylabel("Score", fontsize=12)

fig.tight_layout()
fig.savefig("/Users/chris/Downloads/qual_ratings_vector.svg")