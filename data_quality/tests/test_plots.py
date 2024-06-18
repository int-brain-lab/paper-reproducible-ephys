from data_quality.plots.metrics import metrics_plot, histograms, yield_detail

def test_all_plots():
    for dset in ["steinmetz", "allen"]:
        for region in ["Isocortex", "HPF", "TH"]:
            metrics_plot(dset, region)
            histograms(dset, region)
            yield_detail(dset, region)