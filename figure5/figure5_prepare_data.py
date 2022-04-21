import numpy as np
import pandas as pd
import time

from one.api import ONE, One
from ibllib.atlas import AllenAtlas
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember

from reproducible_ephys_functions import combine_regions, get_insertions, BRAIN_REGIONS, save_data_path, save_dataset_info
from reproducible_ephys_processing import compute_psth
from figure5.figure5_load_data import load_dataframe

ba = AllenAtlas()

default_params = {'fr_bin_size': 0.04,
                  'ff_bin_size': 0.1,
                  'align_event': 'move',
                  'event_epoch': [-0.4, 0.8],
                  'base_event': 'stim',
                  'base_epoch': [-0.4, -0.2],
                  'norm': None,
                  'smoothing': 'sliding',
                  'slide_kwargs_ff': {'n_win': 5, 'causal': 1},
                  'slide_kwargs_fr': {'n_win': 2, 'causal': 1}}


def prepare_data(insertions, one, figure='figure5', recompute=False, **kwargs):

    fr_bin_size = kwargs.get('fr_bin_size', default_params['fr_bin_size'])
    ff_bin_size = kwargs.get('ff_bin_size', default_params['ff_bin_size'])
    align_event = kwargs.get('align_event', default_params['align_event'])
    event_epoch = kwargs.get('event_epoch', default_params['event_epoch'])
    base_event = kwargs.get('base_event', default_params['base_event'])
    base_epoch = kwargs.get('base_epoch', default_params['base_epoch'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])
    slide_kwargs_fr = kwargs.get('slide_kwargs_fr', default_params['slide_kwargs_fr'])
    slide_kwargs_ff = kwargs.get('slide_kwargs_ff', default_params['slide_kwargs_ff'])

    params = {'fr_bin_size': fr_bin_size,
              'ff_bin_size': ff_bin_size,
              'align_event': align_event,
              'event_epoch': event_epoch,
              'base_event': base_event,
              'base_epoch': base_epoch,
              'norm': norm,
              'smoothing': smoothing,
              'slide_kwargs_fr': slide_kwargs_fr,
              'slide_kwargs_ff': slide_kwargs_ff}

    if not recompute:
        df_exists = load_dataframe(exists_only=True)
        if df_exists:
            df = load_dataframe()
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                return df


    all_df = []
    for iIns, ins in enumerate(insertions):
        start = time.time()
        try:
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
            eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

            if align_event == 'move':
                eventTimes = eventMove
            elif align_event == 'stim':
                eventTimes = eventStim

            if base_event == 'move':
                eventBase = eventMove
            elif base_event == 'stim':
                eventBase = eventStim

            _, _, ff_r, time_ff = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                               eventTimes, align_epoch=event_epoch, bin_size=ff_bin_size, baseline_events=eventBase,
                                               base_epoch=base_epoch, smoothing=smoothing,norm=norm, slide_kwargs=slide_kwargs_ff,
                                               return_ff=True)

            data['avg_ff_post_move'] = np.nanmean(ff_r[:, np.bitwise_and(time_ff >= 0.04, time_ff <= 0.2)], axis=1)

            # COMPUTE FANOFACTOR AND FIRING RATE WAVEFORMS (for figure 6 supplement)
            if figure == 'figure6':
                # Compute firing rate waveforms for right 100% contrast
                fr_r, _, time_fr = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                                eventTimes, align_epoch=event_epoch, bin_size=fr_bin_size,
                                                baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                                slide_kwargs=slide_kwargs_fr)

                # Compute fanofactor and firing rate waveforms for left 100% contrast
                trial_idx = np.bitwise_and(trials['feedbackType'] == 1, trials['contrastLeft'] == 1)
                eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]
                eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

                if align_event == 'move':
                    eventTimes = eventMove
                elif align_event == 'stim':
                    eventTimes = eventStim

                if base_event == 'move':
                    eventBase = eventMove
                elif align_event == 'stim':
                    eventBase = eventStim

                # Compute fanofactor
                _, _, ff_l, time_ff = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                                   eventTimes, align_epoch=event_epoch, bin_size=ff_bin_size,
                                                   baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing,norm=norm,
                                                   slide_kwargs=slide_kwargs_ff, return_ff=True)

                # Compute firing rate
                fr_l, _, time_fr = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                                   eventTimes, align_epoch=event_epoch, bin_size=fr_bin_size,
                                                   baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                                   slide_kwargs=slide_kwargs_fr)

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

            all_df.append(df)
            print(time.time() - start)

        except Exception as err:
            print(f'{pid} errored: {err}')

    # Save data frame
    concat_df = pd.concat(all_df, ignore_index=True)
    save_path = save_data_path(figure=figure)
    concat_df.to_csv(save_path.joinpath(f'{figure}_dataframe.csv'))

    if figure == 'figure6':
        data = {'all_frs_l': all_frs_l,
                'all_ffs_l': all_ffs_l,
                'all_frs_r': all_frs_r,
                'all_ffs_r': all_ffs_r,
                'time_fr': time_fr,
                'time_ff': time_ff,
                'params': params}

        smoothing = smoothing or 'none'
        norm = norm or 'none'
        np.savez(save_path.joinpath(f'{figure}_data_event_{align_event}_smoothing_{smoothing}_norm_{norm}.npz'), **data)

        return concat_df, data
    else:
        return concat_df


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=2, one=one)
    prepare_data(insertions, one=one, **default_params)
    save_dataset_info(one, figure='figure5')
