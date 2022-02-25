import numpy as np

def bin_spikes(spike_times, align_times, pre_time, post_time, bin_size, weights=None):

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        r = np.bincount(xind, minlength=tscale.shape[0], weights=w)
        bins[i, :] = r[:-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


def bin_spikes2D(spike_times, spike_clusters, cluster_ids, align_times, pre_time, post_time, bin_size, weights=None):

    n_bins_pre = int(np.ceil(pre_time / bin_size))
    n_bins_post = int(np.ceil(post_time / bin_size))
    n_bins = n_bins_pre + n_bins_post
    tscale = np.arange(-n_bins_pre, n_bins_post + 1) * bin_size
    ts = np.repeat(align_times[:, np.newaxis], tscale.size, axis=1) + tscale
    epoch_idxs = np.searchsorted(spike_times, np.c_[ts[:, 0], ts[:, -1]])
    bins = np.zeros(shape=(align_times.shape[0], cluster_ids.shape[0], n_bins))

    for i, (ep, t) in enumerate(zip(epoch_idxs, ts)):
        xind = (np.floor((spike_times[ep[0]:ep[1]] - t[0]) / bin_size)).astype(np.int64)
        w = weights[ep[0]:ep[1]] if weights is not None else None
        yscale, yind = np.unique(spike_clusters[ep[0]:ep[1]], return_inverse=True)
        nx, ny = [tscale.size, yscale.size]
        ind2d = np.ravel_multi_index(np.c_[yind, xind].transpose(), dims=(ny, nx))
        r = np.bincount(ind2d, minlength=nx * ny, weights=w).reshape(ny, nx)

        bs_idxs = np.isin(cluster_ids, yscale)
        bins[i, bs_idxs, :] = r[:, :-1]

    tscale = (tscale[:-1] + tscale[1:]) / 2

    return bins, tscale


def normalise_fr(fr_mean, fr_base, method='subtract'):

    base = np.mean(fr_base, axis=1)[:, np.newaxis]
    if method == 'subtract':
        fr_norm = fr_mean - base

    elif method == 'z_score':
        fr_norm = (fr_mean - base) / (0.5 + base)

    return fr_norm

def compute_psth(spike_times, spike_clusters, cluster_ids, align_events, align_epoch=(-0.4, 0.8), bin_size=0.01,
                 smoothing=None, baseline_events=None, base_epoch=(-0.5, 0), norm_method=None):

    if smoothing != 'sliding':
        bins, t = bin_spikes2D(spike_times, spike_clusters, cluster_ids, align_events, align_epoch[0], align_epoch[1], bin_size)
        bins_mean = np.mean(bins, axis=0)

        if norm_method is not None:
            baseline_events = baseline_events if any(baseline_events) else align_events
            bins_base, t_base = bin_spikes2D(spike_times, spike_clusters, cluster_ids,
                                             baseline_events, base_epoch[0], base_epoch[1], bin_size)
            bins = normalise_fr(bins_mean / bin_size, bins_base_mean / bin_size, norm_method=norm_method)

            if norm_method == 'subtract':
                bins = normalise_fr(bins_mean / bin_size, bins_base_mean / bin_size, norm_method=norm_method)
            elif norm_method == 'z_score':
                bins = normalise_fr(bins_mean, bins_base_mean, norm_method=norm_method)


def smoothing_kernel(values, kernel=None):

    if kernel is not None:
        kernel_len = kernel.shape[0]
        kernel_area = np.sum(kernel)
    else:
        kernel_len = 10
        kernel = np.exp(-np.arange(kernel_len) * 0.45)
        kernel_area = np.sum(kernel)

    smoothed_values = np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=-1, arr=values) / kernel_area
    smoothed_values = np.take(smoothed_values, np.arange(kernel_len - 1, smoothed_values.shape[-1] - kernel_len + 1), axis=-1)

    return smoothed_values


def smoothing_sliding():
    pass

def cluster_peths_FR_FF_sliding_2D(spike_times, spike_clusters, cluster_ids, align_times, pre_time=0.2, post_time=0.5,
                                    hist_win=0.1, N_SlidesPerWind=5, causal=0):

    epoch = np.r_[-1 * pre_time, post_time]
    tshift = hist_win / N_SlidesPerWind

    if causal == 1:  # Place time points at the end of each hist_win, i.e., only past events are taken into account.
        epoch[0] = epoch[0] - hist_win / 2  # to start earlier since we're shifting the time later

    for s in range(N_SlidesPerWind):

        BinnedSpikes, tscale = bin_spikes2D(spike_times, spike_clusters, cluster_ids, (align_times + s * tshift),
                                            np.abs(epoch[0]), epoch[1] - (s * tshift), hist_win)

        if s == 0:
            FR_TrialAvg = np.nanmean(BinnedSpikes, axis=0) / hist_win
            FR_TrialSTD = np.nanstd(BinnedSpikes, axis=0) / hist_win
            FF_TrialAvg = np.nanvar(BinnedSpikes, axis=0) / np.nanmean(BinnedSpikes, axis=0)
            TimeVect = tscale + s * tshift

        else:
            FR_TrialAvg = np.c_[FR_TrialAvg,  np.nanmean(BinnedSpikes, axis=0) / hist_win]
            FR_TrialSTD = np.c_[FR_TrialSTD, np.nanstd(BinnedSpikes, axis=0) / hist_win]
            FF_TrialAvg = np.c_[FF_TrialAvg,  np.nanvar(BinnedSpikes, axis=0) / np.nanmean(BinnedSpikes, axis=0)]
            TimeVect = np.r_[TimeVect, tscale + s * tshift]

    if causal == 1:
        TimeVect = TimeVect + hist_win / 2

    sort_idx = np.argsort(TimeVect)

    FR_TrialAvg = FR_TrialAvg[:, sort_idx]
    FR_TrialSTD = FR_TrialSTD[:, sort_idx]
    FF_TrialAvg = FF_TrialAvg[:, sort_idx]
    TimeVect = TimeVect[sort_idx]

    return FR_TrialAvg, FR_TrialSTD, FF_TrialAvg, TimeVect