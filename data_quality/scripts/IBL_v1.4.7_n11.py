from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import metrics_plot, histograms, yield_detail
import pandas as pd

regions  = ['Isocortex', 'OLF', 'HPF', 'CTXsp', 'CNU', 'TH','HY', 'MB',]

dset = "allen"
for region in regions:
    
    ibl_dset="re"

    ibl_clusters = load_clusters("IBL_like_allen", region)
    ibl_clusters.amp_median *= 2.34375e-06

    ibl_channels = load_channels(ibl_dset, region)
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]
    
    metrics_plot(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    histograms(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    yield_detail(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)

dset="steinmetz"
for region in regions:
    ibl_dset="bwm"


    ibl_clusters = load_clusters("IBL_like_steinmetz", region)
    ibl_channels = load_channels(ibl_dset, region)
    # make sure we only include insertions listed in the clusters table
    ibl_channels = ibl_channels.loc[list(ibl_clusters.index.get_level_values(0).unique())]
    
    metrics_plot(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    histograms(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)
    yield_detail(dset, region, ibl_clusters=ibl_clusters, ibl_channels=ibl_channels)

    