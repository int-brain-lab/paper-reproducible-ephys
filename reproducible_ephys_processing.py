import numpy as np
from brainbox.metrics.single_units import METRICS_PARAMS
from pathlib import Path


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


# Added by MT:
def compute_psth_rxn_time(spike_times, spike_clusters, cluster_ids,
                          align_events, eventsStim, eventsMove, 
                          align_epoch=(-0.2, 0.2), bin_size=0.01, smoothing='sliding', baseline_events=None, 
                          base_epoch=(-0.5, 0), norm=None, return_ff=False, slide_kwargs={}, kernel_kwargs={}):

    #For now, only smoothing = sliding is considered:
    bins, t = smoothing_sliding(spike_times, spike_clusters, cluster_ids, align_events, align_epoch=align_epoch,
                                bin_size=bin_size, **slide_kwargs)
    
    rxntimes = eventsMove - eventsStim
    #First, remove trials with short (<50 ms) rxn times:
    bins[rxntimes<0.05] = np.nan
    #Second, if rxn time is <200 ms, then limit bins to ~rxn time:
    loc = np.where(np.all([rxntimes>=0.05, rxntimes<0.2], axis=0))
    loc = loc[0]
    for i in loc:
        bin_index = np.where(t >= -rxntimes[i])
        start_index = bin_index[0][0] #- 1
        bins[i][0][0:start_index] = np.nan #set all values prior to rxn time window to nan; Is this accurate enough?
        
    
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

    if return_ff:
        return fr_mean, fr_std, ff, t
    else:
        return fr_mean, fr_std, t

    
    # #compute_psth_rxn_time(eid): 
    # # COMPUTE FIRING RATES DURING RXN TIME
    # # For this computation we use correct, non zero contrast trials
    # #trials = one.load_object(eid, 'trials')
    # trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
    #                            np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))

    # # Trials with nan values in stimOn_times or firstMovement_times
    # nanStimMove = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))

    # # Find event times of interest and remove nan values
    # eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nanStimMove)]
    # eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]   
    # return eventStim, eventMove



def smoothing_kernel(values, t, kernel=None):

    if kernel is not None:
        kernel_len = kernel.shape[0]
        kernel_area = np.sum(kernel)
    else:
        kernel_len = 10
        kernel = np.exp(-np.arange(kernel_len) * 0.45)
        kernel_area = np.sum(kernel)

    smoothed_t = t[kernel_len - 1:]

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
        all_times = all_times + bin_size / 2

    sort_idx = np.argsort(all_times)
    all_bins = all_bins[:, :, sort_idx]
    all_times = all_times[sort_idx]

    return all_bins, all_times


def compute_new_label(spikes, clusters, save_path=None):

    cluster_ids = np.arange(clusters.channels.size)
    nclust = cluster_ids.size
    noise_vals = np.full((nclust,), np.nan)

    for ic in np.arange(nclust):
        ispikes = spikes.clusters == cluster_ids[ic]
        if np.all(~ispikes):  # if this cluster has no spikes, continue
            continue
        amps = spikes.amps[ispikes]
        noise_vals[ic] = noise_cutoff(amps)[0]

    new_label = np.mean(np.c_[clusters.slidingRP_viol, noise_vals,
                              clusters.amp_median > METRICS_PARAMS['med_amp_thresh_uv'] / 1e6], axis=1)
    if save_path is not None:
        np.save(Path(save_path).joinpath('clusters.new_labels.npy'), new_label)

    return new_label


def noise_cutoff(amps, quantile_length=.25, n_bins=100, n_low_bins=1, low_bin_start = 1, nc_threshold = 5, percent_peak = 0.10):
    """
    A new metric to determine whether a unit's amplitude distribution is cut off
    (at floor), without assuming a Gaussian distribution.

    This metric takes the amplitude distribution, computes the mean and std
    of an upper quartile of the distribution, and determines how many standard
    deviations away from that mean a lower quartile lies.

    Parameters
    ----------
    amps : ndarray_like
        The amplitudes (in uV) of the spikes.
    quantile_length : float
        The size of the upper quartile of the amplitude distribution.
    n_bins : int
        The number of bins used to compute a histogram of the amplitude
        distribution.
    n_low_bins : int
        The number of bins used in the lower part of the distribution (where
        cutoff is determined).
    Returns
    -------
    cutoff : float
        Number of standard deviations that the lower mean is outside of the
        mean of the upper quartile.

    See Also
    --------
    missed_spikes_est

    Examples
    --------
    1) Compute whether a unit's amplitude distribution is cut off
        >>> amps = spks_b['amps'][unit_idxs]
        >>> cutoff = bb.metrics.noise_cutoff(amps, quartile_length=.2,
                                             n_bins=100, n_low_bins=2)
    """
    nc_threshold = 5 #the noise cutoff result has to be greater than 5 for neuron to fail
    percent_threshold = 0.10 # the first bin has to be greater than 10% the peak bin for neuron to fail


    if len(amps) > 1: #ensure there are amplitudes available to analyze
        bins_list = np.linspace(0, np.max(amps), n_bins) #list of bins to compute the amplitude histogram
        n, bins = np.histogram(amps, bins=bins_list) #construct amplitude histogram
        idx_nz = np.nonzero(np.diff(n))  #indices of nonzero bins; this ensures we are discarding many early bins mostly below detection threshold
        idx_peak = np.argmax(n)  #peak of amplitude distribution
        length_top_half = len(np.where(n[idx_peak:-1]>0)[0])  #don't count zeros #len(n) - idx_peak   #compute the length of the top half of the distribution -- ignoring zero bins
        high_quantile = 2 * quantile_length  #the remaining part of the distribution, which we will compare the low quantile to
        high_quantile_start_ind = int(np.ceil(high_quantile * length_top_half + idx_peak)) #the first bin (index) of the high quantile part of the distribution
        indices_bins_high_quantile = np.arange(high_quantile_start_ind,len(n)) # bins to consider in the high quantile (of all non-zero bins)
        idx_use = np.where(n[indices_bins_high_quantile]>=1)[0]

        if len(n[indices_bins_high_quantile]) > 0: #e nsure there are amplitudes in these bins
            mean_high_quantile = np.mean(n[indices_bins_high_quantile][idx_use]) # mean of all amp values in high quantile bins
            std_high_quantile = np.std(n[indices_bins_high_quantile][idx_use])
            if std_high_quantile > 0:
                first_low_quantile = n[(n != 0)][1]  # take the second bin
                cutoff = (first_low_quantile - mean_high_quantile) / std_high_quantile
                peak_bin_height = np.max(n)
                percent_of_peak = percent_threshold * peak_bin_height

                fail_criteria = (cutoff > nc_threshold) & (first_low_quantile > percent_of_peak)
            else:
                cutoff = np.float64(np.nan)
                first_low_quantile = np.float64(np.nan)
                fail_criteria = np.ones(1).astype(bool)[0]
        else:
            cutoff = np.float64(np.nan)
            first_low_quantile = np.float64(np.nan)
            fail_criteria = np.ones(1).astype(bool)[0]

    else:
        cutoff = np.float64(np.nan)
        first_low_quantile = np.float64(np.nan)
        fail_criteria = np.ones(1).astype(bool)[0]

    nc_pass = ~fail_criteria
    return nc_pass, cutoff, first_low_quantile
