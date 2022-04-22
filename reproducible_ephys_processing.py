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


def bin_norm(times, events, pre_time, post_time, bin_size, weights):
    bin_vals, t = bin_spikes(times, events, pre_time, post_time, bin_size, weights=weights)
    bin_count, _ = bin_spikes(times, events, pre_time, post_time, bin_size)
    bin_count[bin_count == 0] = 1
    bin_vals = bin_vals / bin_count

    return bin_vals, t

def normalise_fr(bin_mean, bin_base, bin_size, method='subtract'):

    base = np.mean(bin_base, axis=1)[:, np.newaxis]
    if method == 'subtract':
        bin_norm = bin_mean - base
        fr_norm = bin_norm / bin_size

    elif method == 'z_score':
        bin_norm = (bin_mean - base) / (0.5 + base)
        fr_norm = bin_norm

    return fr_norm



def compute_psth(spike_times, spike_clusters, cluster_ids, align_events, align_epoch=(-0.4, 0.8), bin_size=0.01,
                 smoothing=None, baseline_events=None, base_epoch=(-0.5, 0), norm=None, return_ff=False, slide_kwargs={},
                 kernel_kwargs={}):

    if smoothing == 'sliding':
        bins, t = smoothing_sliding(spike_times, spike_clusters, cluster_ids, align_events, align_epoch=align_epoch,
                                    bin_size=bin_size, **slide_kwargs)
        bins_mean = np.nanmean(bins, axis=0)
        fr_std = np.nanstd(bins, axis=0) / bin_size
        if return_ff:
            ff = np.nanvar(bins, axis=0) / bins_mean
        if norm is not None:
            baseline_events = baseline_events if any(baseline_events) else align_events
            bins_base, t_base = smoothing_sliding(spike_times, spike_clusters, cluster_ids, baseline_events,
                                                  align_epoch=base_epoch, bin_size=bin_size, **slide_kwargs)
            fr_mean = normalise_fr(bins_mean, np.nanmean(bins_base, axis=0), bin_size, method=norm)
        else:
            fr_mean = bins_mean / bin_size

    elif smoothing == 'kernel':
        bins, t_bin = bin_spikes2D(spike_times, spike_clusters, cluster_ids, align_events, np.abs(align_epoch[0]), align_epoch[1],
                                   bin_size)
        bins_mean = np.nanmean(bins, axis=0)
        fr_std = np.nanstd(smoothing_kernel(bins, t_bin, **kernel_kwargs)[0], axis=0) / bin_size

        if return_ff:
            ff = np.nanvar(bins, axis=0) / bins_mean
            ff, _ = smoothing_kernel(ff, t_bin, **kernel_kwargs)
        if norm is not None:
            baseline_events = baseline_events if any(baseline_events) else align_events
            bins_base, t_base = bin_spikes2D(spike_times, spike_clusters, cluster_ids,
                                             baseline_events, np.abs(base_epoch[0]), base_epoch[1], bin_size)
            fr_mean = normalise_fr(bins_mean, np.mean(bins_base, axis=0), bin_size, method=norm)
            fr_mean, t = smoothing_kernel(fr_mean, t_bin, **kernel_kwargs)
        else:
            fr_mean, t = smoothing_kernel(bins_mean, t_bin, **kernel_kwargs)
            fr_mean = fr_mean / bin_size

    else:
        bins, t = bin_spikes2D(spike_times, spike_clusters, cluster_ids, align_events, np.abs(align_epoch[0]), align_epoch[1],
                               bin_size)
        bins_mean = np.nanmean(bins, axis=0)
        fr_std = np.nanstd(bins, axis=0) / bin_size

        if return_ff:
            ff = np.nanvar(bins, axis=0) / bins_mean
        if norm is not None:
            baseline_events = baseline_events if any(baseline_events) else align_events
            bins_base, t_base = bin_spikes2D(spike_times, spike_clusters, cluster_ids,
                                             baseline_events, np.abs(base_epoch[0]), base_epoch[1], bin_size)
            fr_mean = normalise_fr(bins_mean, np.mean(bins_base, axis=0), bin_size, method=norm)
        else:
            fr_mean = bins_mean / bin_size

    if return_ff:
        return fr_mean, fr_std, ff, t
    else:
        return fr_mean, fr_std, t


def smoothing_kernel(values, t, kernel=None):

    if kernel is not None:
        kernel_len = kernel.shape[0]
        kernel_area = np.sum(kernel)
    else:
        kernel_len = 10
        kernel = np.exp(-np.arange(kernel_len) * 0.45)
        kernel_area = np.sum(kernel)

    smoothed_t = t[kernel_len-1:]

    smoothed_values = np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=-1, arr=values) / kernel_area
    smoothed_values = np.take(smoothed_values, np.arange(kernel_len - 1, smoothed_values.shape[-1] - kernel_len + 1), axis=-1)

    return smoothed_values, smoothed_t


def smoothing_sliding(spike_times, spike_clusters, cluster_ids, align_times, align_epoch=(-0.2, 0.5), bin_size=0.1, n_win=5,
                      causal=1):
    t_shift = bin_size / n_win
    epoch = [align_epoch[0], align_epoch[1]]
    if causal:
        epoch[0] = epoch[0] - (bin_size / 2)

    for w in range(n_win):

        bins, tscale = bin_spikes2D(spike_times, spike_clusters, cluster_ids, (align_times + w * t_shift), np.abs(epoch[0]),
                               epoch[1] - (w * t_shift), bin_size)
        if w == 0:
            all_bins = bins
            all_times = tscale + w * t_shift
        else:
            all_bins = np.c_[all_bins, bins]
            all_times = np.r_[all_times, tscale + w * t_shift]

    if causal == 1:
        all_times= all_times + bin_size / 2

    sort_idx = np.argsort(all_times)
    all_bins = all_bins[:, :, sort_idx]
    all_times = all_times[sort_idx]

    return all_bins, all_times


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