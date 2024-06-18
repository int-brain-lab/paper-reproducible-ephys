from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import compute_yield
from data_quality.plots.utils import get_colors_region
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

regions  = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'CNU', 'TH','HY', 'MB',]
fr_cutoff = .5
og_clusters = load_clusters("re")
rerun_clusters = load_clusters("re_147")
og_clusters = og_clusters[og_clusters.firing_rate > fr_cutoff]
rerun_clusters = rerun_clusters[rerun_clusters.firing_rate > fr_cutoff]

fig, ax = plt.subplots()
og_clusters[og_clusters.label==1.]["spike_count"].plot.hist(
    bins=200, ax=ax, color="red", alpha=0.5, label="Original",
    histtype="step")
rerun_clusters[rerun_clusters.label==1.]["spike_count"].plot.hist(
    bins=200, ax=ax, color="blue", alpha=0.5, label="Rerun1.4.7",
    histtype="step")
ax.legend()
ax.set_xlim(-5000, 300_000)
ax.set_xlabel("spike count")
ax.set_ylabel("# good units")
ax.set_title(f"Spike count histograms: good units\nFR cutoff = {fr_cutoff}")