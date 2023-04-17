import numpy as np
import pandas as pd
import time
import pickle

from one.api import ONE
from ibllib.atlas import AllenAtlas
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember

from reproducible_ephys_functions import combine_regions, get_insertions, BRAIN_REGIONS, save_data_path, save_dataset_info, compute_metrics, filter_recordings
from reproducible_ephys_processing import compute_psth
from fig_taskmodulation.fig_taskmodulation_load_data import load_dataframe, tests, filtering_criteria
from permutation_test import permut_test, distribution_dist_approx, shuffle_labels, distribution_dist_approx_max


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


def prepare_data(insertions, one, figure='fig_taskmodulation', recompute=False, **kwargs):

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
            eventMoveL = trials['firstMovement_times'][np.bitwise_and(np.bitwise_and(trial_idx, ~nanStimMove),
                                                                      trials['choice'] == 1)]
            eventMoveR = trials['firstMovement_times'][np.bitwise_and(np.bitwise_and(trial_idx, ~nanStimMove),
                                                                      trials['choice'] == -1)]
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

            # MT: This needs further fixing so that any neuron can be selected
            # Save the data that we need for the
            #if pid == 'ce397420-3cd2-4a55-8fd1-5e28321981f4' and figure == 'figure5':
            if pid == '36362f75-96d8-4ed4-a728-5e72284d0995' and figure == 'figure5':
                intervals = np.c_[eventStim[rxn_idx] - 0.2, eventStim[rxn_idx]]
                counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
                assert np.array_equal(cluster_ids, data['cluster_ids'])
                fr_base_example = counts / (intervals[:, 1] - intervals[:, 0])
                neuron_id = 265 #614
                clu_idx = np.where(cluster_ids == neuron_id)[0]
                fr_example = np.c_[fr_base_example[clu_idx, :][0], fr_pre_move_tw[clu_idx, :][0]]
                np.save(save_data_path(figure=figure).joinpath(f'figure5_example_neuron{neuron_id}_{pid}.npy'), fr_example)


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
            data['mean_fr_diff_trial'] = np.mean(fr_trial - fr_base, axis=1)
            data['trial'], _, data['p_trial'] = \
                compute_comparison_statistics(fr_base, fr_trial, test='signrank')

            # Post-stimulus vs Baseline
            data['mean_fr_diff_post_stim'] = np.mean(fr_post_stim - fr_base, axis=1)
            data['post_stim'], _, data['p_post_stim'] = \
                compute_comparison_statistics(fr_base, fr_post_stim, test='signrank')

            # Pre-movement vs Baseline
            data['mean_fr_diff_pre_move'] = np.mean(fr_pre_move - fr_base, axis=1)
            data['pre_move'], _, data['p_pre_move'] = \
                compute_comparison_statistics(fr_base, fr_pre_move, test='signrank')

            # Pre-movement left versus right
            data['mean_fr_diff_pre_move_lr'] = np.mean(fr_pre_moveL, axis=1) - np.mean(fr_pre_moveR, axis=1)
            data['pre_move_lr'], _, data['p_pre_move_lr'] = \
                compute_comparison_statistics(fr_pre_moveL, fr_pre_moveR, test='ranksums')

            # Time warped start-to-move vs Baseline
            data['mean_fr_diff_start_to_move'] = np.mean(fr_pre_move_tw - fr_base[:, rxn_idx], axis=1)
            data['start_to_move'], _, data['p_start_to_move'] = \
                compute_comparison_statistics(fr_base[:, rxn_idx], fr_pre_move_tw, test='signrank')

            # Post-movement vs Baseline
            data['mean_fr_diff_post_move'] = np.mean(fr_post_move - fr_base, axis=1)
            data['post_move'], _, data['p_post_move'] = \
                compute_comparison_statistics(fr_base, fr_post_move, test='signrank')

            # Post-reward vs Baseline
            data['mean_fr_diff_post_reward'] = np.mean(fr_post_reward - fr_base, axis=1)
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
                                               eventTimes, align_epoch=event_epoch, bin_size=ff_bin_size,
                                               baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing, norm=norm,
                                               slide_kwargs=slide_kwargs_ff, return_ff=True)

            data['avg_ff_post_move'] = np.nanmean(ff_r[:, np.bitwise_and(time_ff >= 0.04, time_ff <= 0.2)], axis=1)

            # Extra computations for figure 7 (now figure 6)
            if figure != 'fig_taskmodulation':
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
                                                   baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing,
                                                   norm=norm, slide_kwargs=slide_kwargs_ff, return_ff=True)

                # Compute firing rate
                fr_l, _, time_fr = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                                eventTimes, align_epoch=event_epoch, bin_size=fr_bin_size,
                                                baseline_events=eventBase, base_epoch=base_epoch, smoothing=smoothing,
                                                norm=norm, slide_kwargs=slide_kwargs_fr)

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

            # Computations for figure 4
            else:
                # update some parameters for these computations
                fr_bin_size = 0.06
                params['fr_bin_size'] = fr_bin_size
                norm = 'subtract'
                params['norm'] = norm
                slide_kwargs_fr = {'n_win': 3, 'causal': 1}
                params['slide_kwargs_fr'] = slide_kwargs_fr
                event_epoch = [-0.2, 0.2]
                params['event_epoch'] = event_epoch
                base_epoch = [-0.2, 0.0]
                params['base_epoch'] = base_epoch

                # COMPUTE AVG ACTIVITY
                # For this computation we use correct, non zero contrast trials
                trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                                           np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))

                eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nanStimMove)]
                eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nanStimMove)]

                if align_event == 'move':
                    eventTimes = eventMove
                    trial_l_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nanStimMove)] == 1)[0]
                    trial_r_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nanStimMove)] == -1)[0]
                elif align_event == 'stim':
                    eventTimes = eventStim
                    trial_l_idx = np.where(trials['contrastLeft'][np.bitwise_and(trial_idx, ~nanStimMove)] > 0)[0]
                    trial_r_idx = np.where(trials['contrastRight'][np.bitwise_and(trial_idx, ~nanStimMove)] > 0)[0]

                # Find baseline event times
                if base_event == 'move':
                    eventBase = eventMove
                elif base_event == 'stim':
                    eventBase = eventStim

                # Compute firing rates for left side events
                fr_l, fr_l_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                                 eventTimes[trial_l_idx], align_epoch=event_epoch, bin_size=fr_bin_size,
                                                 baseline_events=eventBase[trial_l_idx], base_epoch=base_epoch,
                                                 smoothing=smoothing, norm=norm)
                fr_l_std = fr_l_std / np.sqrt(trial_l_idx.size)  # convert to standard error

                # Compute firing rates for right side events
                fr_r, fr_r_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                                 eventTimes[trial_r_idx], align_epoch=event_epoch, bin_size=fr_bin_size,
                                                 baseline_events=eventBase[trial_r_idx], base_epoch=base_epoch,
                                                 smoothing=smoothing, norm=norm)
                fr_r_std = fr_r_std / np.sqrt(trial_r_idx.size)  # convert to standard error

                if iIns == 0:
                    all_frs_l = fr_l
                    all_frs_l_std = fr_l_std
                    all_frs_r = fr_r
                    all_frs_r_std = fr_r_std
                else:
                    all_frs_l = np.r_[all_frs_l, fr_l]
                    all_frs_l_std = np.r_[all_frs_l_std, fr_l_std]
                    all_frs_r = np.r_[all_frs_r, fr_r]
                    all_frs_r_std = np.r_[all_frs_r_std, fr_r_std]


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

        except Exception as err:
            print(f'{pid} errored: {err}')

    # Save data frame
    concat_df = pd.concat(all_df, ignore_index=True)
    save_path = save_data_path(figure=figure)
    print(f'Saving data to {save_path}')
    concat_df.to_csv(save_path.joinpath(f'{figure}_dataframe.csv'))

    if figure != 'fig_taskmodulation':
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
        data = {'all_frs_l': all_frs_l,
                'all_frs_l_std': all_frs_l_std,
                'all_frs_r': all_frs_r,
                'all_frs_r_std': all_frs_r_std,
                'time': t,
                'params': params}

        smoothing = smoothing or 'none'
        norm = norm or 'none'
        np.savez(save_path.joinpath(f'{figure}_data_event_{align_event}_smoothing_{smoothing}_norm_{norm}.npz'), **data)
        return concat_df

def compute_permutation_test(n_permut=20000, qc='pass', n_cores=8):
    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
    if qc == 'pass':
        df_filt = df_filt[df_filt['permute_include'] == 1]
    elif qc != 'all':
        df_filt = df_filt[(df_filt['permute_include'] == 1) | (df_filt[qc] == 1)]

    df_filt_reg = df_filt.groupby('region')
    results = pd.DataFrame()

    for test in tests.keys():
        for reg in BRAIN_REGIONS:

            print("Warning, region reduced")
            if reg != 'CA1':
                continue

            df_reg = df_filt_reg.get_group(reg)
            # vals = df_reg.groupby(['institute', 'subject'])[test].mean()
            # labs = vals.index.get_level_values('institute')
            # subjects = vals.index.get_level_values('subject')
            # data = vals.values
            if test == 'avg_ff_post_move':
                data = df_reg[test].values
            else:
                data = df_reg['mean_fr_diff_{}'.format(test)].values
            labs = df_reg['institute'].values
            subjects = df_reg['subject'].values

            labs = labs[~np.isnan(data)]
            subjects = subjects[~np.isnan(data)]
            data = data[~np.isnan(data)]
            # lab_names, this_n_labs = np.unique(labs, return_counts=True)  # what is this for?

            print(".", end='')
            p = permut_test(data, metric=distribution_dist_approx_max, labels1=labs, plot=True,
                            labels2=subjects, shuffling='labels1_based_on_2', n_cores=n_cores, n_permut=n_permut)

            # print(p)
            # if p > 0.05:
            #     return data, labs, subjects
            results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1],
                                                       data={'test': test, 'region': reg, 'p_value_permut': p})))

    pickle.dump(results.p_value_permut.values, open(save_data_path(figure='fig_taskmodulation').joinpath('p_values'), 'wb'))

def compute_power_analysis(n_permut=50000, n_cores=8):
    p_values = pickle.load(open(save_data_path(figure='fig_taskmodulation').joinpath('p_values'), 'rb'))  # renew by calling plot_panel_permutation
    print(p_values)
    print(np.sum(p_values < 0.01))

    df = load_dataframe()
    df_filt = filter_recordings(df, **filtering_criteria)
    df_filt = df_filt[df_filt['permute_include'] == 1]

    df_filt_reg = df_filt.groupby('region')

    i = -1
    significant_disturbances = np.zeros((len(p_values), 10, 2))
    for test in tests.keys():
        print(test)
        for jj, reg in enumerate(BRAIN_REGIONS):
            i += 1

            df_reg = df_filt_reg.get_group(reg)

            if test == 'avg_ff_post_move':
                data = df_reg[test].values
            else:
                data = df_reg['mean_fr_diff_{}'.format(test)].values
            labs = df_reg['institute'].values
            subjects = df_reg['subject'].values

            labs = labs[~np.isnan(data)]
            subjects = subjects[~np.isnan(data)]
            data = data[~np.isnan(data)]

            if p_values[i] < 0.01:
                print("test already significant {}, {}, {}".format(test, reg, i))
                continue

            for j, manipulate_lab in enumerate(np.unique(labs)):
                lower, higher = find_sig_manipulation(data.copy(), manipulate_lab, labs, subjects, 0.01, 'positive', n_permut=n_permut, n_cores=n_cores)
                significant_disturbances[i, j, 0] = higher
                print("found bound: {}".format(higher))
                lower, higher = find_sig_manipulation(data.copy(), manipulate_lab, labs, subjects, 0.01, 'negative', n_permut=n_permut, n_cores=n_cores)
                significant_disturbances[i, j, 1] = lower
                print("found bound: {}".format(lower))
            pickle.dump(significant_disturbances, open(save_data_path(figure='fig_taskmodulation').joinpath('shifts'), 'wb'))


def find_sig_manipulation(data, lab_to_manip, labs, subjects, p_to_reach, direction='positive', sensitivity=0.01, n_permut=50000, n_cores=8):
    lower_bound = 0 if direction == 'positive' else -1000
    higher_bound = 1000 if direction == 'positive' else 0

    found_bound = False
    bound = 0
    while not found_bound:
        bound += 10 if direction == 'positive' else -10
        p = permut_test(data + (labs == lab_to_manip) * bound, metric=distribution_dist_approx_max, labels1=labs,
                        labels2=subjects, shuffling='labels1_based_on_2', n_cores=n_cores, n_permut=n_permut)
        if p < p_to_reach:
            found_bound = True
            if direction == 'positive':
                higher_bound = bound
            else:
                lower_bound = bound
        else:
            if direction == 'positive':
                lower_bound = bound
            else:
                higher_bound = bound
        if not found_bound:
            if direction == 'positive':
                if bound > data.max() + 20:
                    print("Failed to find")
                    return -np.inf, np.inf
            else:
                if bound < data.min() - 20:
                    print("Failed to find")
                    return -np.inf, np.inf

    while np.abs(lower_bound - higher_bound) > sensitivity:

        test = (lower_bound + higher_bound) / 2
        p = permut_test(data + (labs == lab_to_manip) * test, metric=distribution_dist_approx_max, labels1=labs,
                        labels2=subjects, shuffling='labels1_based_on_2', n_cores=n_cores, n_permut=n_permut)
        if p < p_to_reach:
            if direction == 'positive':
                higher_bound = test
            else:
                lower_bound = test
        else:
            if direction == 'positive':
                lower_bound = test
            else:
                higher_bound = test
    return lower_bound, higher_bound


if __name__ == '__main__':
    print("Filtering criteria: {}".format(filtering_criteria))
    one = ONE()
    one.record_loaded = True
    insertions = get_insertions(level=0, one=one, freeze='release_2022_11', recompute=True)
    prepare_data(insertions, one=one, recompute=True, **default_params)
    save_dataset_info(one, figure='fig_taskmodulation')
    compute_permutation_test(n_permut=50000, n_cores=8)
    compute_power_analysis(n_permut=50000, n_cores=8)
