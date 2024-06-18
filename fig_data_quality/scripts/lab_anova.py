from data_quality.tables import load_clusters, load_channels
from data_quality.plots.metrics import compute_yield
import scipy.stats as ss

ibl_clusters = load_clusters("re_147")
ibl_channels = load_channels("re")
ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]

_clusters = load_clusters("allen")
_channels = load_channels("allen")

re_yield, _ = compute_yield(ibl_clusters, ibl_channels, _clusters, _channels)
re_yield = re_yield[["lab", "passing_per_site", ]]

labs = re_yield.groupby("lab").groups

groups = [re_yield.loc[labs[x]]["passing_per_site"] for x in labs.keys()]

anova = ss.f_oneway(*groups)

print(f"F statistic: {anova.statistic}")
print(f"p: {anova.pvalue}")