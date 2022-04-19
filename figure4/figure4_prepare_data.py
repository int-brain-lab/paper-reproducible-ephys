import pandas as pd
import numpy as np
import logging

from one.api import ONE, One
from brainbox.io.one import SpikeSortingLoader
from iblutil.numerical import ismember
from ibllib.atlas import AllenAtlas

from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS, get_insertions, save_data_path, save_dataset_info
from reproducible_ephys_processing import compute_psth
from figure4.figure4_load_data import load_data, load_dataframe


logger = logging.getLogger('paper_repro_ephys')
ba = AllenAtlas()

# Defaults parameters for psth computation
default_params = {'bin_size': 0.01,
                  'align_event': 'move',
                  'event_epoch': [-0.4, 0.2],
                  'base_event': 'stim',
                  'base_epoch': [-0.4, -0.2],
                  'norm': 'subtract',
                  'smoothing': 'kernel',
                  'slide_kwargs': {'n_win': 5, 'causal': 1}}


def prepare_data(insertions, one, recompute=False, **kwargs):

    bin_size = kwargs.get('bin_size', default_params['bin_size'])
    align_event = kwargs.get('align_event', default_params['align_event'])
    event_epoch = kwargs.get('event_epoch', default_params['event_epoch'])
    base_event = kwargs.get('base_event', default_params['base_event'])
    base_epoch = kwargs.get('base_epoch', default_params['base_epoch'])
    norm = kwargs.get('norm', default_params['norm'])
    smoothing = kwargs.get('smoothing', default_params['smoothing'])
    slide_kwargs = kwargs.get('slide_kwargs', default_params['slide_kwargs'])

    params = {'bin_size': bin_size,
              'align_event': align_event,
              'event_epoch': event_epoch,
              'base_event': base_event,
              'base_epoch': base_epoch,
              'norm': norm,
              'smoothing': smoothing,
              'slide_kwargs': slide_kwargs}

    if not recompute:
        # TODO comparison based on the params used
        data_exists = load_data(event=align_event, norm=norm, smoothing=smoothing, exists_only=True)
        if data_exists:
            df = load_dataframe()
            pids = np.array([p['probe_insertion'] for p in insertions])
            isin, _ = ismember(pids, df['pid'].unique())
            if np.all(isin):
                print('Already computed data for set of insertions. Will load in data. To recompute set recompute=True')
                data = load_data()
                return df, data

    all_df = []
    for iIns, ins in enumerate(insertions):

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
            trial_l_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nan_trials)] == 1)[0]
            trial_r_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nan_trials)] == -1)[0]
        elif align_event == 'stim':
            eventTimes = eventStim
            trial_l_idx = np.where(trials['contrastLeft'][np.bitwise_and(trial_idx, ~nan_trials)] > 0)[0]
            trial_r_idx = np.where(trials['contrastRight'][np.bitwise_and(trial_idx, ~nan_trials)] > 0)[0]

        # Find baseline event times
        if base_event == 'move':
            eventBase = eventMove
        elif base_event == 'stim':
            eventBase = eventStim

        # Compute firing rates for left side events
        fr_l, fr_l_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                         eventTimes[trial_l_idx], align_epoch=event_epoch, bin_size=bin_size,
                                         baseline_events=eventBase[trial_l_idx], base_epoch=base_epoch,
                                         smoothing=smoothing, norm=norm)
        fr_l_std = fr_l_std / np.sqrt(trial_l_idx.size)  # convert to standard error

        # Compute firing rates for right side events
        fr_r, fr_r_std, t = compute_psth(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                                         eventTimes[trial_r_idx], align_epoch=event_epoch, bin_size=bin_size,
                                         baseline_events=eventBase[trial_r_idx], base_epoch=base_epoch,
                                         smoothing=smoothing, norm=norm)
        fr_r_std = fr_r_std / np.sqrt(trial_r_idx.size)  # convert to standard error

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
            all_frs_l = fr_l
            all_frs_l_std = fr_l_std
            all_frs_r = fr_r
            all_frs_r_std = fr_r_std
        else:
            all_frs_l = np.r_[all_frs_l, fr_l]
            all_frs_l_std = np.r_[all_frs_l_std, fr_l_std]
            all_frs_r = np.r_[all_frs_r, fr_r]
            all_frs_r_std = np.r_[all_frs_r_std, fr_r_std]

    concat_df = pd.concat(all_df, ignore_index=True)
    data = {'all_frs_l': all_frs_l,
            'all_frs_l_std': all_frs_l_std,
            'all_frs_r': all_frs_r,
            'all_frs_r_std': all_frs_r_std,
            'time': t,
            'params': params}

    save_path = save_data_path(figure='figure4')
    concat_df.to_csv(save_path.joinpath('figure4_dataframe.csv'))
    smoothing = smoothing or 'none'
    norm = norm or 'none'
    np.savez(save_path.joinpath(f'figure4_data_event_{align_event}_smoothing_{smoothing}_norm_{norm}.npz'), **data)

    return concat_df, data


if __name__ == '__main__':
    one = ONE()
    one.load_recorded = True
    insertions = get_insertions(level=2, one=one)
    prepare_data(insertions, one=one, **default_params)
    save_dataset_info(one, figure='figure4')
