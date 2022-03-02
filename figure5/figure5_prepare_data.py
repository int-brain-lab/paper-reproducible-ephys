import numpy as np
import pandas as pd
from pathlib import Path
import time

from one.api import ONE, One
from ibllib.atlas import AllenAtlas
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from brainbox.io.one import SpikeSortingLoader

from reproducible_ephys_functions import combine_regions, get_insertions, labs, BRAIN_REGIONS
from reproducible_ephys_paths import DATA_PATH
from figure5.figure5_functions import cluster_peths_FR_FF_sliding_2D

one = ONE()
one_local = One()
ba = AllenAtlas()
lab_number_map, institution_map, lab_colors = labs()
insertions = get_insertions(level=2, as_dataframe=False, one=one)
save_waveforms = True
start0 = time.time()
all_df = []
for iIns, ins in enumerate(insertions):
    start = time.time()
    print(f'processing {iIns + 1}/{len(insertions)}')

    data = {}

    # Load in data
    pid = ins['probe_insertion']
    eid = ins['session']['id']
    probe = ins['probe_name']
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])
    clusters = sl.merge_clusters(spikes, clusters, channels)
    clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])

    # Find clusters that are in the regions we are interested in and are good
    cluster_idx = np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1)
    data['cluster_ids'] = clusters['cluster_id'][cluster_idx]
    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], data['cluster_ids'])
    if np.sum(spike_idx) == 0:

        continue

    # COMPUTE AVERAGE FIRING RATE ACROSS SESSION
    counts, cluster_ids = get_spike_counts_in_bins(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                   np.c_[spikes['times'][0], spikes['times'][-1]])
    data['avg_fr'] = counts.ravel() / (spikes['times'][-1] - spikes['times'][0])

    # COMPUTE FIRING RATES DURING DIFFERENT PARTS OF TASK
    # For this computation we use correct, non zero contrast trials
    trials = one.load_object(eid, 'trials')
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
    data['trial'], _, data['p_trial'] = \
        compute_comparison_statistics(fr_base, fr_trial, test='signrank')

    # Post-stimulus vs Baseline
    data['post_stim'], _, data['p_post_stim'] = \
        compute_comparison_statistics(fr_base, fr_post_stim, test='signrank')

    # Pre-movement vs Baseline
    data['pre_move'], _, data['p_pre_move'] = \
        compute_comparison_statistics(fr_base, fr_pre_move, test='signrank')

    # Pre-movement left versus right
    data['pre_move_lr'], _, data['p_pre_move_lr'] = \
        compute_comparison_statistics(fr_pre_moveL, fr_pre_moveR, test='ranksums')

    # Time warped start-to-move vs Baseline
    data['start_to_move'], _, data['p_start_to_move'] = \
        compute_comparison_statistics(fr_base[:, rxn_idx], fr_pre_move_tw, test='signrank')

    # Post-movement vs Baseline
    data['post_move'], _, data['p_post_move'] = \
        compute_comparison_statistics(fr_base, fr_post_move, test='signrank')

    # Post-reward vs Baseline
    data['post_reward'], _, data['p_post_reward'] = \
        compute_comparison_statistics(fr_base, fr_post_reward, test='signrank')

    # COMPUTE FANOFACTOR
    # For this computation we use correct, stim right, 100 % contrast trials
    trial_idx = np.bitwise_and(trials['feedbackType'] == 1, trials['contrastRight'] == 1)
    eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

    ff_bin = 0.1
    ff_n_slide = 5
    pre_time = 0.4
    post_time = 0.8

    _, _, ff_r, time_ff = cluster_peths_FR_FF_sliding_2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                       data['cluster_ids'], eventMove, pre_time=pre_time, post_time=post_time,
                                                       hist_win=ff_bin, N_SlidesPerWind=ff_n_slide, causal=1)
    data['avg_ff_post_move'] = np.nanmean(ff_r[:, np.bitwise_and(time_ff >= 0.04, time_ff <= 0.2)], axis=1)

    # COMPUTE FANOFACTOR AND FIRING RATE WAVEFORMS (for figure 6 supplement)
    if save_waveforms:
        # Compute firing rate waveforms for right 100% contrast
        fr_bin = 0.04
        fr_n_slide = 2

        fr_r, _, _, time_fr = cluster_peths_FR_FF_sliding_2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                           data['cluster_ids'], eventMove, pre_time=pre_time, post_time=post_time,
                                                           hist_win=fr_bin, N_SlidesPerWind=fr_n_slide, causal=1)

        # Compute fanofactor and firing rate waveforms for left 100% contrast
        trial_idx = np.bitwise_and(trials['feedbackType'] == 1, trials['contrastLeft'] == 1)
        eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

        # Compute fanofactor
        _, _, ff_l, time_ff = cluster_peths_FR_FF_sliding_2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                             data['cluster_ids'], eventMove, pre_time=pre_time,
                                                             post_time=post_time, hist_win=ff_bin, N_SlidesPerWind=ff_n_slide,
                                                             causal=1)

        # Compute firing rate
        fr_l, _, _, time_fr = cluster_peths_FR_FF_sliding_2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx],
                                                           data['cluster_ids'], eventMove, pre_time=pre_time, post_time=post_time,
                                                           hist_win=fr_bin, N_SlidesPerWind=fr_n_slide, causal=1)
        if iIns == 0:
            all_frs_l = fr_l
            all_frs_r = fr_r
            all_ffs_l = ff_l
            all_ffs_r = ff_r
        else:
            all_frs_l = np.r_[all_frs_l, fr_l]
            all_frs_r = np.r_[all_frs_r, fr_r]
            all_ffs_l = np.r_[all_ffs_l, ff_l]
            all_ffs_r = np.r_[all_ffs_r, ff_r]

    # Add some extra cluster data to store
    data['region'] = clusters['rep_site_acronym'][cluster_idx]
    data['x'] = clusters['x'][cluster_idx]
    data['y'] = clusters['y'][cluster_idx]
    data['z'] = clusters['z'][cluster_idx]
    data['p2t'] = clusters['peakToTrough'][cluster_idx]
    data['amp'] = clusters['amps'][cluster_idx]


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
    print(time.time() - start)

# Save data frame
concat_df = pd.concat(all_df, ignore_index=True)
save_path = Path(DATA_PATH).joinpath('figure5')
save_path.mkdir(exist_ok=True, parents=True)
concat_df.to_csv(save_path.joinpath('figure5_figure6_dataframe.csv'))
if save_waveforms:
    np.savez(save_path.joinpath('figure6_data_fr_ff.npy'), all_frs_l=all_frs_l, all_frs_r=all_frs_r, all_ffs_l=all_ffs_l,
             all_ffs_r=all_ffs_r, time_ff=time_ff, time_fr=time_fr)
print(time.time() - start0)

# import matplotlib.pyplot as plt
# from reproducible_ephys_functions import filter_recordings
# concat_df_reg = filter_recordings(concat_df)
# 
# concat_df_reg = concat_df.groupby('region')
# for reg in BRAIN_REGIONS:
#     fig, ax = plt.subplots(2, 2)
#     df_reg = concat_df_reg.get_group(reg)
#     reg_idx = concat_df_reg.groups[reg]
#     ffs_r_reg = all_ffs_r[reg_idx, :]
#     frs_r_reg = all_frs_r[reg_idx, :]
#     ffs_l_reg = all_ffs_l[reg_idx, :]
#     frs_l_reg = all_frs_l[reg_idx, :]
#     ax[0][0].plot(time_fr, np.nanmean(frs_r_reg, axis=0))
#     ax[1][0].plot(time_ff, np.nanmean(ffs_r_reg, axis=0))
#     ax[0][1].plot(time_fr, np.nanmean(frs_l_reg, axis=0))
#     ax[1][1].plot(time_ff, np.nanmean(ffs_l_reg, axis=0))
#     fig.suptitle(reg)
#     
# from matplotlib import cm, colors
# for reg in BRAIN_REGIONS:
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     df_reg = concat_df_reg.get_group(reg)
#     norm = colors.Normalize(vmin=np.nanmin(df_reg['avg_ff_post_move']), vmax=np.nanmax(df_reg['avg_ff_post_move']),
#                                        clip=False)
#     mapper = cm.ScalarMappable(norm=norm, cmap=cm.get_cmap('viridis'))
#     cluster_color = np.array([mapper.to_rgba(col) for col in df_reg['avg_ff_post_move']])
#     s = np.ones_like(df_reg['x']) * 2
#     s[df_reg['avg_ff_post_move'] < 1] = 6
#     scat = ax.scatter(df_reg['x'], df_reg['y'], df_reg['z'], c=cluster_color, marker='o', s=s)
#     cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='viridis'), ax=ax)
#     break