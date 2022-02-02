import numpy as np
import pandas as pd
from pathlib import Path
import time

from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from brainbox.io.one import SpikeSortingLoader

from reproducible_ephys_functions import combine_regions, get_insertions, labs, BRAIN_REGIONS
from reproducible_ephys_paths import DATA_PATH
from figure5.figure5_functions import cluster_peths_FR_FF_sliding_2D

one = ONE()
ba = AllenAtlas()
lab_number_map, institution_map, lab_colors = labs()
insertions = get_insertions(level=2, as_dataframe=False)

all_df = []
for iIns, ins in enumerate(insertions):

    print(f'processing {iIns + 1}/{len(insertions)}')

    data = {}

    # Load in data
    pid = ins['probe_insertion']
    eid = ins['session']['id']
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
    clusters = sl.merge_clusters(spikes, clusters, channels)
    clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
    trials = one.load_object(eid, 'trials')

    # Find clusters that are in the regions we are interested in and are good
    cluster_idx = np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1)
    data['cluster_ids'] = clusters['cluster_id'][cluster_idx]
    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], data['cluster_ids'])

    # COMPUTE AVERAGE FIRING RATE ACROSS SESSION
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                   np.c_[spikes['times'][0], spikes['times'][-1]])
    data['avg_fr'] = counts.ravel() / (spikes['times'][-1] - spikes['times'][0])

    # COMPUTE FIRING RATES DURING DIFFERENT PARTS OF TASK
    # For this computation we use correct, non zero contrast trials
    trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                               np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))

    # Trials with nan values in stimOn_times or firstMovement_times
    nanStimMove = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))

    # Find event times of interest and remove nan values
    eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nanStimMove)]
    eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]
    eventMoveL = trials['firstMovement_times'][np.bitwise_and(np.bitwise_and(trial_idx, ~nanStimMove), trials['choice'] == 1)]
    eventMoveR = trials['firstMovement_times'][np.bitwise_and(np.bitwise_and(trial_idx, ~nanStimMove), trials['choice'] == -1)]
    eventFeedback = trials['feedback_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

    # Baseline firing rate
    intervals = np.c_[eventStim - 0.2, eventStim]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_base = counts / (intervals[:, 1] - intervals[:, 0])

    data['avg_fr_base'] = np.nanmean(fr_base, axis=1)

    # Trial firing rate
    intervals = np.c_[eventStim, eventStim + 0.4]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_trial = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_trial'] = np.nanmean(fr_trial, axis=1)

    # Reaction time firing rate
    intervals = np.c_[eventStim, eventMove]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_rxn = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_rxn_time'] = np.nanmean(fr_rxn, axis=1)

    # Post-stimulus firing rate
    intervals = np.c_[eventStim + 0.05, eventStim + 0.15]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_post_stim = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_post_stim'] = np.nanmean(fr_post_stim, axis=1)

    # Pre-move firing rate
    intervals = np.c_[eventMove - 0.1, eventMove + 0.05]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_pre_move = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_pre_move'] = np.nanmean(fr_pre_move, axis=1)

    # Pre-move left firing rate
    intervals = np.c_[eventMoveL - 0.1, eventMoveL + 0.05]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_pre_moveL = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_pre_moveL'] = np.nanmean(fr_pre_moveL, axis=1)

    # Pre-move right firing rate
    intervals = np.c_[eventMoveR - 0.1, eventMoveR + 0.05]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_pre_moveR = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_pre_moveR'] = np.nanmean(fr_pre_moveR, axis=1)

    # Time warped pre move firing rate
    # Cap reaction times to 0.2s
    rxn_times = eventMove - eventStim
    rxn_times[rxn_times > 0.2] = 0.2
    # Only keep trials where reaction time > 0.05
    rxn_idx = rxn_times > 0.05
    intervals = np.c_[eventMove[rxn_idx] - rxn_times[rxn_idx], eventMove[rxn_idx]]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_pre_move_tw = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_pre_move_tw'] = np.nanmean(fr_pre_move_tw, axis=1)

    # Post-move firing rate
    intervals = np.c_[eventMove - 0.05, eventMove + 0.2]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_post_move = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_post_move'] = np.nanmean(fr_post_move, axis=1)

    # Post-reward firing rate
    intervals = np.c_[eventFeedback, eventFeedback + 0.15]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    assert np.array_equal(cluster_ids, data['cluster_ids'])
    fr_post_reward = counts / (intervals[:, 1] - intervals[:, 0])
    data['avg_fr_post_reward'] = np.nanmean(fr_post_reward, axis=1)

    # COMPARE FIRING RATES TO FIND RESPONSIVE UNITS
    # Trial vs Baseline
    data['trial'], data['p_trial'], _ = \
        compute_comparison_statistics(fr_base, fr_trial, test='signrank')

    # Post-stimulus vs Baseline
    data['post_stim'], data['p_post_stim'], _ = \
        compute_comparison_statistics(fr_base, fr_post_stim, test='signrank')

    # Pre-movement vs Baseline
    data['pre_move'], data['p_pre_move'], _ = \
        compute_comparison_statistics(fr_base, fr_pre_move, test='signrank')

    # Pre-movement left versus right
    data['pre_move_lr'], data['p_pre_move_lr'], _ = \
        compute_comparison_statistics(fr_pre_moveL, fr_pre_moveR, test='ranksums')

    # Time warped start-to-move vs Baseline
    data['start_to_move'], data['p_start_to_move'], _ = \
        compute_comparison_statistics(fr_base[:, rxn_idx], fr_pre_move_tw, test='signrank')

    # Post-movement vs Baseline
    data['post_move'], data['p_post_move'], _ = \
        compute_comparison_statistics(fr_base, fr_post_move, test='signrank')

    # Post-reward vs Baseline
    data['post_reward'], data['p_post_reward'], _ = \
        compute_comparison_statistics(fr_base, fr_post_reward, test='signrank')

    # COMPUTE FANOFACTOR
    # For this computation we use correct, stim right, 100 % contrast trials
    trial_idx = np.bitwise_and(trials['feedbackType'] == 1, trials['contrastRight'] == 1)
    eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

    ff_bin = 0.1
    ff_n_slide = 5

    _, _, ff, time_ff = cluster_peths_FR_FF_sliding_2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                       data['cluster_ids'], eventMove, pre_time=0, post_time=0.4,
                                                       hist_win=ff_bin, N_SlidesPerWind=ff_n_slide, causal=1)
    # Compute average FF for time period 40 - 200 ms after movement onset
    data['avg_ff_post_move'] = np.nanmean(ff[:, np.bitwise_and(time_ff >= 0.04, time_ff <= 0.2)], axis=1)

    # Add some extra cluster data to store
    data['region'] = clusters['rep_site_acronym'][cluster_idx]
    data['x'] = clusters['x'][cluster_idx]
    data['y'] = clusters['y'][cluster_idx]
    data['z'] = clusters['z'][cluster_idx]
    data['p2t'] = clusters['peakToTrough'][cluster_idx]
    data['amp'] = clusters['amps'][cluster_idx]

    # Annotate any regions that have less than threshold units
    data['include'] = np.ones_like(data['cluster_ids'])
    reg, reg_counts = np.unique(data['region'], return_counts=True)
    n_counts = 4
    if np.any(reg_counts < n_counts):
        lt_idx = np.where(reg_counts < n_counts)[0]
        for idx in lt_idx:
            data['include'][data['region'] == reg[idx]] = 0

    df = pd.DataFrame.from_dict(data)
    df['eid'] = eid
    df['pid'] = pid
    df['subject'] = ins['session']['subject']
    df['probe'] = ins['probe_name']
    df['date'] = ins['session']['start_time'][:10]
    df['lab'] = ins['session']['lab']
    df['institute'] = df['lab'].map(institution_map)
    df['lab_number'] = df['lab'].map(lab_number_map)

    all_df.append(df)

# Save data frame
concat_df = pd.concat(all_df, ignore_index=True)
save_path = Path(DATA_PATH).joinpath('figure5')
save_path.mkdir(exist_ok=True, parents=True)
concat_df.to_csv(save_path.joinpath('figure5_figure6_dataframe.csv'))
