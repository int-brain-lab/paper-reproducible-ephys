from data_quality.tables import tables_dir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as ss


results = pd.read_csv(tables_dir.joinpath("double_blind_results.csv"))
results = results.loc[:100]

def jitter(values,j):
    return values + np.random.normal(j,0.05,values.shape)

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
        x="Source", y=jitter(results[rater], 0.2), 
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

df = results.drop(columns=["OID", "UUID"])
df = df.melt(id_vars=["Source"])
df.rename(columns={"value":"Score", "variable":"Rater"}, inplace=True)

# anova
dataset_map = {"IBL":1, "Steinmetz":2, "Allen":3}
rater_map = {"GC":1, "NS":2, "FD":3}
df = df.replace({"Rater":rater_map,
                 "Source":dataset_map})

import statsmodels.api as sm
from statsmodels.formula.api import ols

model = ols('Q("Score") ~ C(Q("Source")) + C(Q("Rater")) + C(Q("Source")):C(Q("Rater"))',
            df).fit()
sm.stats.anova_lm(model, typ=2)
