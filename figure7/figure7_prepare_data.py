import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from brainbox.io.one import SpikeSortingLoader
from ibllib.atlas import AllenAtlas
from one.api import ONE, One
from pathlib import Path
from reproducible_ephys_functions import combine_regions, BRAIN_REGIONS, labs, get_insertions
from reproducible_ephys_paths import DATA_PATH
from reproducible_ephys_processing import bin_spikes2D, normalise_fr
from brainbox.population.decode import get_spike_counts_in_bins
from brainbox.task.closed_loop import compute_comparison_statistics

one_online = ONE()
one_local = One()
ba = AllenAtlas()
lab_number_map, institution_map, lab_colors = labs()

insertions = get_insertions(level=2)
all_df = []
split = 'rt'
norm_method = 'z_score'

for iIns, ins in enumerate(insertions):

    print(f'processing {iIns + 1}/{len(insertions)}')
    eid = ins['session']['id']
    probe = ins['probe_name']
    pid = ins['probe_insertion']

    data = {}

    # Load in spikesorting
    sl = SpikeSortingLoader(eid=eid, pname=probe, one=one_local, atlas=ba)
    spikes, clusters, channels = sl.load_spike_sorting()
    clusters = sl.merge_clusters(spikes, clusters, channels)
    clusters['rep_site_acronym'] = combine_regions(clusters['acronym'])
    # Find clusters that are in the repeated site brain regions and that have been labelled as good
    cluster_idx = np.sort(np.where(np.bitwise_and(np.isin(clusters['rep_site_acronym'], BRAIN_REGIONS), clusters['label'] == 1))[0])
    data['cluster_ids'] = clusters['cluster_id'][cluster_idx]
    data['region'] = clusters['rep_site_acronym'][cluster_idx]
    print(len(data['cluster_ids']))

    # Find spikes that are from the clusterIDs
    spike_idx = np.isin(spikes['clusters'], data['cluster_ids'])

    # Load in trials data
    trials = one_local.load_object(eid, 'trials', collection='alf')
    # For this computation we use correct, non zero contrast trials
    trial_idx = np.bitwise_and(trials['feedbackType'] == 1,
                               np.bitwise_or(trials['contrastLeft'] > 0, trials['contrastRight'] > 0))
    # Find nan trials
    nan_trials = np.bitwise_or(np.isnan(trials['stimOn_times']), np.isnan(trials['firstMovement_times']))

    # nan_trials = np.c_[np.isnan(trials['stimOn_times']), np.isnan(trials['feedback_times']), np.isnan(trials['choice']),
    #                    np.isnan(trials['feedbackType']), np.isnan(trials['probabilityLeft'])]
    # nan_trials = np.sum(nan_trials, axis=1) > 0

    # Find trials that are too long
    stim_diff = trials['feedback_times'] - trials['stimOn_times']
    rm_trials = stim_diff > 10
    # Remove these trials from trials object
    rm_trials = np.bitwise_or(rm_trials, nan_trials)

    eventMove = trials['firstMovement_times'][np.bitwise_and(trial_idx, ~rm_trials)]
    eventStim = trials['stimOn_times'][np.bitwise_and(trial_idx, ~rm_trials)]

    if split == 'choice':
        trial_l_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~rm_trials)] == 1)[0]
        trial_r_idx = np.where(trials['choice'][np.bitwise_and(trial_idx, ~rm_trials)] == -1)[0]
    elif split == 'block':
        trial_l_idx = np.where(trials['probabilityLeft'][np.bitwise_and(trial_idx, ~rm_trials)] == 0.2)[0]
        trial_r_idx = np.where(trials['probabilityLeft'][np.bitwise_and(trial_idx, ~rm_trials)] == 0.8)[0]
    elif split == 'rt':
        rt = eventMove - eventStim
        trial_l_idx = np.where(rt < 0.1)[0]
        trial_r_idx = np.where(rt > 0.2)[0]


    pre_time = 0.5
    post_time = 1
    bin_size = 0.02

    # Movement firing rate
    bins, t = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'],
                            eventMove, pre_time, post_time, bin_size)
    # Baseline firing rate
    bins_base, t_base = bin_spikes2D(spikes['times'][spike_idx], spikes['clusters'][spike_idx], data['cluster_ids'], eventMove,
                                     0.5, 0, bin_size)
    if norm_method == 'subtract':
        bins = bins / bin_size
        bins_base = bins_base / bin_size

    # Mean firing rate across trials for each neuron
    fr_l = np.mean(bins[trial_l_idx, :, :], axis=0)
    fr_r = np.mean(bins[trial_r_idx, :, :], axis=0)

    # Mean baseline firing rate across trials for each neuron
    fr_base_l = np.mean(bins_base[trial_l_idx, :, :], axis=0)
    fr_base_r = np.mean(bins_base[trial_r_idx, :, :], axis=0)

    # Normalise the firing rates
    fr_l_norm = normalise_fr(fr_l, fr_base_l, method=norm_method)
    fr_r_norm = normalise_fr(fr_r, fr_base_r, method=norm_method)

    # Combine the firing rate from the two trial conditions
    frs = np.c_[fr_l_norm, fr_r_norm]

    # Find responsive neurons
    # Baseline firing rate
    intervals = np.c_[eventStim - 0.2, eventStim]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    fr_base = counts / (intervals[:, 1] - intervals[:, 0])

    # Post-move firing rate
    intervals = np.c_[eventMove - 0.05, eventMove + 0.2]
    counts, cluster_ids = get_spike_counts_in_bins(spikes.times[spike_idx], spikes.clusters[spike_idx], intervals)
    fr_post_move = counts / (intervals[:, 1] - intervals[:, 0])

    data['responsive'], data['p_responsive'], _ = \
        compute_comparison_statistics(fr_base, fr_post_move, test='signrank')

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
        all_frs = frs
    else:
        all_frs = np.r_[all_frs, frs]

concat_df = pd.concat(all_df, ignore_index=True)

save_path = Path(DATA_PATH).joinpath('figure8')
save_path.mkdir(exist_ok=True, parents=True)
concat_df.to_csv(save_path.joinpath('figure8_dataframe.csv'))
np.save(save_path.joinpath(f'figure8_data_split_{split}.npy'), all_frs)




### TEMP from here down####
# Embedding with 2 PCA components on all responsive units
# Find responsive units
pca = PCA(n_components=2)
pca.fit(all_frs[concat_df['responsive']])
emb = pca.transform(all_frs[concat_df['responsive']])
concat_df['emb1'] = np.nan
concat_df['emb2'] = np.nan
concat_df.loc[concat_df['responsive'], 'emb1'] = emb[:, 0]
concat_df.loc[concat_df['responsive'], 'emb2'] = emb[:, 1]


concat_df_reg = concat_df.groupby('region')
fig, ax = plt.subplots()
for reg in BRAIN_REGIONS:
    df_reg = concat_df_reg.get_group(reg)
    reg_idx = concat_df_reg.groups[reg]
    frs_reg = all_frs[reg_idx, :]
    ax.plot(np.mean(frs_reg, axis=0) / bin_size)

# This happens at the end, where we need to groupby individual regions
    # Compute the PCA and reconstructed trial data from first two components
    pca = PCA()
    pca.fit(frs_reg)
    u, s, vh = np.linalg.svd(frs_reg)
    print('comps:', pca.n_components_, 'features:', pca.n_features_)
    print(pca.explained_variance_ratio_[:3])

    S_star = np.zeros(frs_reg.shape)
    for i in range(2):
        S_star[i, i] = s[i]
    Y_re_star = np.matrix(u) * np.matrix(S_star) * np.matrix(vh)
