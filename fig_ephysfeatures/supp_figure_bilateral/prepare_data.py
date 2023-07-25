import numpy as np
import pandas as pd
from os.path import isfile

from one.api import ONE
from ibllib.atlas import AllenAtlas
from ibllib.pipes.ephys_alignment import EphysAlignment
from iblutil.numerical import ismember
from brainbox.processing import compute_cluster_average
from brainbox.io.one import SpikeSortingLoader
import reproducible_ephys_functions
from reproducible_ephys_functions import (get_insertions, combine_regions, BRAIN_REGIONS, save_data_path,
                                          save_dataset_info, filter_recordings, query)
from reproducible_ephys_processing import compute_new_label, compute_psth
from fig_ephysfeatures.ephysfeatures_load_data import load_dataframe


ba = AllenAtlas()

LFP_BAND = [20, 80]


def prepare_data(insertions, one, recompute=False):

    if not recompute:
        data_clust = load_dataframe(df_name='clust', exists_only=True)
        data_chn = load_dataframe(df_name='chns', exists_only=True)
        data_ins = load_dataframe(df_name='ins', exists_only=True)
        if data_clust and data_chn and data_ins:
            df = load_dataframe(df_name='ins')
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                df_clust = load_dataframe(df_name='clust')
                df_chns = load_dataframe(df_name='chns')
                return df_chns, df_clust, df

    all_df_clust = []
    all_df_chns = []
    metrics = pd.DataFrame()

    for iIns, ins in enumerate(insertions):

        print(f'processing {iIns + 1}/{len(insertions)}')
        data_clust = {}
        data_chns = {}

        eid = ins['session']['id']
        lab = ins['session']['lab']
        subject = ins['session']['subject']
        date = ins['session']['start_time'][:10]
        pid = ins['probe_insertion']
        probe = ins['probe_name']

        sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
        spikes, clusters, channels = sl.load_spike_sorting(dataset_types=['clusters.amps', 'clusters.peakToTrough'])

        channels['rawInd'] = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=sl.collection)
        clusters = sl.merge_clusters(spikes, clusters, channels)

        channels['rep_site_acronym'] = combine_regions(channels['acronym'])
        channels['rep_site_acronym_alt'] = np.copy(channels['rep_site_acronym'])
        channels['rep_site_acronym_alt'][channels['rep_site_acronym_alt'] == 'PPC'] = 'VISa'
        channels['rep_site_id'] = ba.regions.acronym2id(channels['rep_site_acronym_alt'])
        clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])

        # Data for cluster dataframe
        cluster_idx = np.where(clusters['label'] == 1)
        data_clust['cluster_ids'] = clusters['cluster_id'][cluster_idx]
        data_clust['region'] = clusters['rep_site_acronym'][cluster_idx]
        # Find spikes that are from the clusterIDs
        spike_idx = np.isin(spikes['clusters'], data_clust['cluster_ids'])

        clu, data_clust['depths'], n_cluster = compute_cluster_average(spikes['clusters'][spike_idx], spikes['depths'][spike_idx])
        assert np.array_equal(clu, data_clust['cluster_ids'])
        clu, data_clust['amps'], _ = compute_cluster_average(spikes['clusters'][spike_idx], spikes['amps'][spike_idx])
        assert np.array_equal(clu, data_clust['cluster_ids'])
        data_clust['fr'] = n_cluster / np.max(spikes.times) - np.min(spikes.times)

        insertion = one.alyx.rest('insertions', 'list', id=pid)[0]
        traj = one.alyx.rest('trajectories', 'list', provenance='Ephys aligned histology track', probe_insertion=pid)[0]
        feature, track = [*traj['json'][insertion['json']['extended_qc']['alignment_stored']]][:2]
        xyz_picks = np.array(insertion['json']['xyz_picks']) / 1e6
        ephysalign = EphysAlignment(xyz_picks, channels['axial_um'], brain_atlas=ba, feature_prev=feature, track_prev=track,
                                    speedy=True)
        data_clust['depths_aligned'] = ephysalign.get_channel_locations(feature, track, data_clust['depths'] / 1e6)[:, 2] * 1e6

        df_clust = pd.DataFrame.from_dict(data_clust)
        df_clust['eid'] = eid
        df_clust['pid'] = pid
        df_clust['subject'] = ins['session']['subject']
        df_clust['probe'] = ins['probe_name']
        df_clust['date'] = ins['session']['start_time'][:10]
        df_clust['lab'] = ins['session']['lab']

        # Data for channel dataframe
        try:
            lfp = one.load_object(eid, 'ephysSpectralDensityLF', collection=f'raw_ephys_data/{probe}')
        except Exception as err:
            print(err)
            print(f'eid: {eid}\n')
        freqs = ((lfp['freqs'] > LFP_BAND[0])
                 & (lfp['freqs'] < LFP_BAND[1]))
        power = lfp['power'][:, channels['rawInd']]
        lfp_power = np.nanmean(10 * np.log(power[freqs]), axis=0)

        data_chns['x'] = channels['x']
        data_chns['y'] = channels['y']
        data_chns['z'] = channels['z']
        data_chns['axial_um'] = channels['axial_um']
        data_chns['lateral_um'] = channels['lateral_um']
        data_chns['lfp'] = lfp_power
        data_chns['region_id'] = channels['atlas_id']
        data_chns['region_id_rep'] = channels['rep_site_id']
        data_chns['region'] = channels['rep_site_acronym']

        df_chns = pd.DataFrame.from_dict(data_chns)
        df_chns['eid'] = eid
        df_chns['pid'] = pid
        df_chns['subject'] = ins['session']['subject']
        df_chns['probe'] = ins['probe_name']
        df_chns['date'] = ins['session']['start_time'][:10]
        df_chns['lab'] = ins['session']['lab']

        all_df_clust.append(df_clust)
        all_df_chns.append(df_chns)

        # Data for insertion dataframe
        try:
            rms_ap = one.load_object(eid, 'ephysTimeRmsAP', collection=f'raw_ephys_data/{probe}',
                                     attribute=['rms'])
        except Exception as err:
            print(err)
            continue
        rms_ap_data = rms_ap['rms'] * 1e6  # convert to uV
        median = np.mean(np.apply_along_axis(lambda x: np.median(x), 1, rms_ap_data))
        rms_ap_data_median = (np.apply_along_axis(lambda x: x - np.median(x), 1, rms_ap_data)
                              + median)


        for region in BRAIN_REGIONS:
            region_clusters = np.where(np.bitwise_and(clusters['rep_site_acronym'] == region,
                                                      clusters['label'] == 1))[0]
            region_chan = channels['rawInd'][np.where(channels['rep_site_acronym'] == region)[0]]

            # Get AP band rms
            rms_ap_region = np.median(rms_ap_data_median[:, region_chan])
            lfp_region = np.median(lfp_power[region_chan])

            if region_clusters.size == 0:
                metrics = pd.concat((metrics, pd.DataFrame(
                    index=[metrics.shape[0] + 1], data={'pid': pid, 'eid': eid, 'probe': probe,
                                                        'lab': lab, 'subject': subject,
                                                        'region': region, 'date': date,
                                                        'median_firing_rate': np.nan,
                                                        'mean_firing_rate': np.nan,
                                                        'spike_amp_mean': np.nan,
                                                        'spike_amp_median': np.nan,
                                                        'spike_amp_90': np.nan,
                                                        'rms_ap': rms_ap_region})))
            else:
                # Get firing rate and spike amplitude
                neuron_fr = np.empty(len(region_clusters))
                spike_amp = np.empty(len(region_clusters))
                for n, neuron_id in enumerate(region_clusters):
                    neuron_fr[n] = np.sum(spikes['clusters'] == neuron_id) / np.max(spikes['times'])
                    spike_amp[n] = np.median(spikes.amps[spikes['clusters'] == neuron_id])

                # Add to dataframe
                metrics = pd.concat((metrics, pd.DataFrame(
                    index=[metrics.shape[0] + 1], data={
                        'pid': pid, 'eid': eid, 'probe': probe,
                        'lab': lab, 'subject': subject,
                        'region': region, 'date': date,
                        'median_firing_rate': np.median(neuron_fr),
                        'mean_firing_rate': np.mean(neuron_fr),
                        'spike_amp_mean': np.nanmean(spike_amp) * 1e6,
                        'spike_amp_median': np.nanmedian(spike_amp),
                        'spike_amp_90': np.percentile(spike_amp, 95),
                        'rms_ap': rms_ap_region,
                        'lfp_power': lfp_region,
                        'yield_per_channel': len(region_clusters) / len(region_chan)})))



    concat_df_clust = pd.concat(all_df_clust, ignore_index=True)
    concat_df_chns = pd.concat(all_df_chns, ignore_index=True)
    save_path = save_data_path(figure='supp_figure_bilateral')
    print(f'Saving data to {save_path}')
    concat_df_clust.to_csv(save_path.joinpath('supp_figure_bilateral_dataframe_clust.csv'))
    concat_df_chns.to_csv(save_path.joinpath('supp_figure_bilateral_dataframe_chns.csv'))
    metrics.to_csv(save_path.joinpath('supp_figure_bilateral_dataframe_ins.csv'))

    return all_df_chns, all_df_clust, metrics


default_params = {'bin_size': 0.06,
                  'align_event': 'move',
                  'event_epoch': [-0.35, 0.22], #[-0.4, 0.22],
                  'base_event': 'stim',
                  'base_epoch': [-0.4, -0.2], #Check (MT)
                  'norm': 'subtract',
                  'smoothing': 'sliding',
                  'slide_kwargs': {'n_win': 5, 'causal': 1},
                  'slide_kwargs_fr': {'n_win': 3, 'causal': 1}}


def prepare_neural_data(insertions, one, recompute=False, **kwargs):

    bin_size = kwargs.get('bin_size', default_params['bin_size'])
    align_event = kwargs.get('align_event', default_params['align_event'])
    event_epoch = kwargs.get('event_epoch', default_params['event_epoch'])
    base_event = kwargs.get('base_event', default_params['base_event'])
    base_epoch = kwargs.get('base_epoch', default_params['base_epoch'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])
    slide_kwargs = kwargs.get('slide_kwargs', default_params['slide_kwargs'])
    slide_kwargs_fr = kwargs.get('slide_kwargs_fr', default_params['slide_kwargs_fr'])

    params = {'bin_size': bin_size,
              'align_event': align_event,
              'event_epoch': event_epoch,
              'base_event': base_event,
              'base_epoch': base_epoch,
              'norm': norm,
              'smoothing': smoothing,
              'slide_kwargs': slide_kwargs,
              'slide_kwargs_fr': slide_kwargs_fr}

    if not recompute:
        print("Not implemented")
        quit()
        # TODO comparison based on the params used
        # data_exists = load_data(event=align_event, norm=norm, smoothing=smoothing, exists_only=True)
        # if data_exists:
        #     df = load_dataframe()
        #     pids = np.array([p['probe_insertion'] for p in insertions])
        #     isin, _ = ismember(pids, df['pid'].unique())
        #     if np.all(isin):
        #         print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
        #         data = load_data()
        #         return df, data

    all_df = []
    for iIns, ins in enumerate(insertions):

        try:
            print(f'processing {iIns + 1}/{len(insertions)}')
            eid = ins['session']['id']
            probe = ins['probe_name']
            pid = ins['probe_insertion']

            data = {}

            # Load in spikesorting
            sl = SpikeSortingLoader(eid=eid, pname=probe, one=one, atlas=ba)
            spikes, clusters, channels = sl.load_spike_sorting()
            clusters = sl.merge_clusters(spikes, clusters, channels)

            clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
            # Find clusters that are in the repeated site brain regions and that have been labelled as good
            cluster_idx = np.sort(np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS),
                                                          clusters['label'] == 1))[0])
            data['cluster_ids'] = clusters['cluster_id'][cluster_idx]

            # Find spikes that are from the clusterIDs
            spike_idx = np.isin(spikes['clusters'], data['cluster_ids'])
            if np.sum(spike_idx) == 0:
                continue

            # Load in trials data
            trials = one.load_object(eid, 'trials', collection='alf')
            # For this computation we use correct, non zero contrast trials
            trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                                       np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))
            # Find nan trials
            nan_trials = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))

            eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nan_trials)]
            eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nan_trials)]

            # Find align events
            if align_event == 'move':
                eventTimes = eventMove
                trial_r_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nan_trials)] == -1)[0]
                trial_l_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nan_trials)] == 1)[0]
                trial_idx = np.concatenate((trial_r_idx, trial_l_idx))
            elif align_event == 'stim':
                eventTimes = eventStim
                trial_r_idx = np.where(trials['contrastRight'][np.bitwise_and(trial_idx, ~nan_trials)] > 0)[0]
                trial_l_idx = np.where(trials['contrastLeft'][np.bitwise_and(trial_idx, ~nan_trials)] > 0)[0]
                trial_idx = np.concatenate((trial_r_idx, trial_l_idx))

            # Find baseline event times
            if base_event == 'move':
                eventBase = eventMove
            elif base_event == 'stim':
                eventBase = eventStim

            # Compute firing rates for left side events
            fr, fr_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                         eventTimes[trial_idx], align_epoch=event_epoch, bin_size=bin_size,
                                         baseline_events=eventBase[trial_idx], base_epoch=base_epoch,
                                         smoothing=smoothing, norm=norm)

            fr_std = fr_std / np.sqrt(trial_idx.size)  # convert to standard error

            # Compute firing rates for right side events
            # fr_r, fr_r_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
            #                                  eventTimes[trial_r_idx], align_epoch=event_epoch, bin_size=bin_size,
            #                                  baseline_events=eventBase[trial_r_idx], base_epoch=base_epoch,
            #                                  smoothing=smoothing, norm=norm)
            # fr_r_std = fr_r_std / np.sqrt(trial_r_idx.size)  # convert to standard error

            # Add other cluster information
            data['region'] = clusters['rep_site_acronym'][cluster_idx]

            df = pd.DataFrame.from_dict(data)
            df['eid'] = eid
            df['pid'] = pid
            df['subject'] = ins['session']['subject']
            df['probe'] = ins['probe_name']
            df['date'] = ins['session']['start_time'][:10]
            df['lab'] = ins['session']['lab']

            all_df.append(df)

            if iIns == 0:
                all_frs = fr
                all_frs_std = fr_std
                # all_frs_r = fr_r
                # all_frs_r_std = fr_r_std
            else:
                all_frs = np.r_[all_frs, fr]
                all_frs_std = np.r_[all_frs_std, fr_std]
                # all_frs_r = np.r_[all_frs_r, fr_r]
                # all_frs_r_std = np.r_[all_frs_r_std, fr_r_std]

        except Exception as err:
            print(f'{pid} errored: {err}')

    concat_df = pd.concat(all_df, ignore_index=True)
    data = {'all_frs': all_frs,
            'all_frs_std': all_frs_std,
            # 'all_frs_r': all_frs_r,
            # 'all_frs_r_std': all_frs_r_std,
            'time': t,
            'params': params}

    save_path = save_data_path(figure='supp_figure_bilateral')
    print(f'Saving data to {save_path}')
    concat_df.to_csv(save_path.joinpath('supp_figure_bilateral_dataframe_neural.csv'))
    smoothing = smoothing or 'none'
    norm = norm or 'none'
    np.savez(save_path.joinpath(f'supp_figure_bilateral_data_event_{align_event}_smoothing_{smoothing}_norm_{norm}.npz'), **data)

    return concat_df, data


if __name__ == '__main__':
    one = ONE()
    one.record_loaded = True

    # Query bilateral insertions in the right hemisphere
    insertions = query(min_regions=0, n_trials=0, behavior=False, exclude_critical=True, one=one,
                       as_dataframe=False, bilateral=True)
    reproducible_ephys_functions.compute_metrics(insertions, one=one, bilateral=True)
    prepare_neural_data(insertions, recompute=True, one=one)
    all_df_chns, all_df_clust, metrics = prepare_data(insertions, recompute=True, one=one)
    save_dataset_info(one, figure='supp_figure_bilateral')
