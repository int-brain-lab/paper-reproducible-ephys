import pandas as pd
import numpy as np
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
from one.api import ONE, One
from pathlib import Path
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS, labs, get_insertions
from reproducible_ephys_paths import DATA_PATH
from reproducible_ephys_processing import bin_spikes2D, normalise_fr, smoothing_kernel


one_online = ONE()
one_local = One()
ba = AllenAtlas()
lab_number_map, institution_map, lab_colors = labs()

insertions = get_insertions(level=2)
all_df = []
event = 'move'
norm_method = 'subtract'
smoothing = 'kernel'  # kernel or None

for iIns, ins in enumerate(insertions):

    print(f'processing {iIns + 1}/{len(insertions)}')
    eid = ins['session']['id']
    probe = ins['probe_name']
    pid = ins['probe_insertion']

    data = {}

    # Load in spikesorting
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one_online, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
    # Find clusters that are in the repeated site brain regions and that have been labelled as good
    cluster_idx = np.sort(np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1))[0])
    data['cluster_ids'] = clusters['cluster_id'][cluster_idx]

    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], data['cluster_ids'])

    # Load in trials data
    trials = one_online.load_object(eid, 'trials', collection='alf')
    # For this computation we use correct, non zero contrast trials
    trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                               np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))
    # Find nan trials
    nan_trials = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))

    eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~nan_trials)]
    eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~nan_trials)]

    if event == 'move':
        eventTimes = eventMove
        trial_l_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nan_trials)] == 1)[0]
        trial_r_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~nan_trials)] == -1)[0]
        pre_time, post_time = 0.4, 0.2
    elif event == 'stim':
        eventTimes = eventStim
        trial_l_idx = np.where(trials['contrastLeft'][np.bitwise_and(trial_idx, ~nan_trials)] > 0)[0]
        trial_r_idx = np.where(trials['contrastRight'][np.bitwise_and(trial_idx, ~nan_trials)] > 0)[0]
        pre_time, post_time = 0.2, 0.4

    bin_size = 0.01

    # Movement firing rate
    bins, t = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                            eventMove, pre_time, post_time, bin_size)
    # Baseline firing rate
    bins_base, t_base = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'], eventStim,
                                     0.4, -0.2, bin_size)
    if norm_method == 'subtract':
        bins = bins / bin_size
        bins_base = bins_base / bin_size

    # Mean firing rate across trials for each neuron
    fr_l = np.mean(bins[trial_l_idx, :, :], axis=0)
    fr_l_std = np.std(bins[trial_l_idx, :, :], axis=0)
    fr_r = np.mean(bins[trial_r_idx, :, :], axis=0)
    fr_r_std = np.std(bins[trial_r_idx, :, :], axis=0)

    # Mean baseline firing rate across trials for each neuron
    fr_base_l = np.mean(bins_base[trial_l_idx, :, :], axis=0)
    fr_base_r = np.mean(bins_base[trial_r_idx, :, :], axis=0)

    # Normalise the firing rates
    fr_l_norm = normalise_fr(fr_l, fr_base_l, method=norm_method)
    fr_r_norm = normalise_fr(fr_r, fr_base_r, method=norm_method)

    fr_r_smooth = smoothing_kernel(fr_l_norm)
    fr_l_smooth = smoothing_kernel(fr_r_norm)

    # Add other cluster information
    data['region'] = clusters['rep_site_acronym'][cluster_idx]

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
    if iIns == 0:
        all_frs_l = fr_l_smooth
        all_frs_r = fr_r_smooth
    else:
        all_frs_l = np.r_[all_frs_l, fr_l_smooth]
        all_frs_r = np.r_[all_frs_r, fr_r_smooth]

concat_df = pd.concat(all_df, ignore_index=True)

save_path = Path(DATA_PATH).joinpath('figure4')
save_path.mkdir(exist_ok=True, parents=True)
concat_df.to_csv(save_path.joinpath('figure4_dataframe.csv'))
np.savez(save_path.joinpath(f'figure4_data_event_{event}.npy'), all_frs_l=all_frs_l, all_frs_r=all_frs_r, time=t)

import matplotlib.pyplot as plt
concat_df_reg = concat_df.groupby('region')
fig, ax = plt.subplots(1, len(BRAIN_REGIONS))
for iR, reg in enumerate(BRAIN_REGIONS):
    df_reg = concat_df_reg.get_group(reg)
    df_reg_subj = df_reg.groupby('subject')
    for subj in df_reg_subj.groups.keys():
        df_subj = df_reg_subj.get_group(subj)
        subj_idx = df_reg_subj.groups[subj]
        frs_subj = all_frs_l[subj_idx, :]
        ax[iR].plot(np.mean(frs_subj, axis=0), c=lab_colors[df_subj.iloc[0]['institute']])

