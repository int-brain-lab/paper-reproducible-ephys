from data_quality.tables import load_channels, load_clusters
from data_quality.plots.metrics import metrics_plot, histograms, yield_detail

# 100 uV
amp_cutoff = 100. * 1e-6
nc_cutoff = 5

for dset in ["steinmetz", "allen"]:
    for region in ["Isocortex", "HPF", "TH"]:
        if dset=="steinmetz":
            ibl_dset="bwm"
        if dset=="allen":
            ibl_dset="re"

        clusters = load_clusters(dset, region)
        channels = load_channels(dset, region)
        ibl_clusters = load_clusters(ibl_dset, region)
        ibl_channels = load_channels(ibl_dset, region)

        clusters["label"] = 0.0
        ibl_clusters["label"] = 0.0

        ibl_pass = (ibl_clusters.slidingRP_viol==1.0)&(ibl_clusters.noise_cutoff<5)&(ibl_clusters.amp_median>amp_cutoff)
        ibl_clusters.loc[ibl_pass, "label"] = 1.0
        npass = (clusters.slidingRP_viol==1.0)&(clusters.noise_cutoff<5)&(clusters.amp_median>amp_cutoff)
        clusters.loc[npass, "label"] = 1.0

        metrics_plot(dset, region, ibl_clusters=ibl_clusters, clusters=clusters)
        histograms(dset, region, ibl_clusters=ibl_clusters, clusters=clusters)
        yield_detail(dset, region, ibl_clusters=ibl_clusters, clusters=clusters)

        