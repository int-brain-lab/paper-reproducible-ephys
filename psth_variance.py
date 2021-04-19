from oneibl.one import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import brainbox as bb
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query


regions = ['CA1', 'VISa', 'DG', 'LP', 'PO']


one = ONE()

event = 'Stim'
regions = regions
left_region_activities = {x: [] for x in regions}
right_region_activities = {x: [] for x in regions}
zero_region_activities = {x: [] for x in regions}
region_names = {x: [] for x in regions}
region_counts = {x: -1 for x in regions}
lab_indices = {x: {} for x in regions}
traj = query(behavior=True)
names = []

kernel_len = 10
kernel = np.exp(-np.arange(kernel_len) * 0.75)

# for count, t in enumerate(traj):
#     eid = t['session']['id']
#     probe = t['probe_name']
#
#     # load data
#     try:
#         spikes, clusters, channels = pickle.load(open("../data/data_{}.p".format(eid), "rb"))
#     except FileNotFoundError:
#         try:
#             spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, force=True)
#             spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
#             pickle.dump((spikes, clusters, channels), (open("../data/data_{}.p".format(eid), "wb")))
#         except KeyError:
#             print(eid)
#             continue
#
#     if event == 'Stim':
#         times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
#         base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
#         base_line_times = base_line_times[~np.isnan(base_line_times)]
#         contrast_L, contrast_R = one.load(eid, dataset_types=['trials.contrastLeft', 'trials.contrastRight'])
#         event_times_left = times[contrast_L > 0]
#         event_times_right = times[contrast_R > 0]
#         event_times_0 = times[np.logical_or(contrast_R == 0, contrast_L == 0)]
#         pre_time, post_time = 0.2, 0.4
#     elif event == 'Block':
#         times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
#         base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
#         base_line_times = base_line_times[~np.isnan(base_line_times)]
#         block_prob = one.load(eid, dataset_types=['trials.probabilityLeft'])
#         event_times_left = times[block_prob[0] == 0.8]
#         event_times_right = times[block_prob[0] == 0.2]
#         pre_time, post_time = 0.2, 0.4
#     elif event == 'Move':
#         times = one.load(eid, dataset_types=['trials.firstMovement_times'])[0]
#         base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
#         base_line_times = base_line_times[~np.isnan(base_line_times)]
#         choice = one.load(eid, dataset_types=['trials.choice'])[0]
#         if (~np.isnan(times)).sum() < 300:
#             continue
#         choice = choice[~np.isnan(times)]
#         times = times[~np.isnan(times)]
#         event_times_left = times[choice == -1]
#         event_times_right = times[choice == 1]
#         pre_time, post_time = 0.4, 0.2
#
#     cluster_regions = channels.acronym[clusters.channels]
#
#     for br in regions:
#         mask = np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), br), clusters['metrics']['label'] == 1)
#         # TODO: 0.1 spikes/s criterion, mikroV threshold
#         if mask.sum() < 4:
#             continue
#
#         if t['session']['subject'] not in names:
#             names.append(t['session']['subject'])
#         print("{} good {}, {} units".format(t['session']['subject'], br, mask.sum()))
#         activity_left, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_left, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
#         activity_right, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_right, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
#
#         activity_pre, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], base_line_times, pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01)
#         baseline = np.mean(activity_pre.means, axis=1)
#
#         activity_right_norm = ((activity_right.means.T - baseline) / (1 + baseline)).T
#         activity_left_norm = ((activity_left.means.T - baseline) / (1 + baseline)).T
#
#         left_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=activity_left_norm)[:, kernel_len-1:-kernel_len+1])
#         right_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=activity_right_norm)[:, kernel_len-1:-kernel_len+1])
#         region_names[br].append(t['session']['subject'])
#
#         if event == 'Stim':
#             activity_0, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_0, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
#             activity_zero_norm = ((activity_0.means.T - baseline) / (1 + baseline)).T
#             zero_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=activity_zero_norm)[:, kernel_len-1:-kernel_len+1])
#
#         region_counts[br] += 1
#         if t['session']['lab'] not in lab_indices[br]:
#             lab_indices[br][t['session']['lab']] = [region_counts[br]]
#         else:
#             lab_indices[br][t['session']['lab']].append(region_counts[br])
#
# pickle.dump((left_region_activities, right_region_activities, lab_indices, activity_left), (open("../data/info {}.p".format(event), "wb")))
left_region_activities, right_region_activities, lab_indices, activity_left = pickle.load(open("../data/info {}.p".format(event), "rb"))

for i, br in enumerate(regions):

    all_left_act = np.vstack(left_region_activities[br])
    all_right_act = np.vstack(right_region_activities[br])

    lab_means_left = {}
    lab_means_right = {}
    for key, idx in lab_indices[br].items():
        if len(idx) < 2:
            continue
        lab_means_left[key] = np.mean([np.mean(left_region_activities[br][x], axis=0) for x in idx], axis=0)
        lab_means_right[key] = np.mean([np.mean(right_region_activities[br][x], axis=0) for x in idx], axis=0)
    var_across_left = np.var(list(lab_means_left.values()), axis=0, ddof=0)
    var_across_right = np.var(list(lab_means_right.values()), axis=0, ddof=0)

    lab_vars_left = {}
    lab_vars_right = {}
    for key, idx in lab_indices[br].items():
        if len(idx) < 2:
            continue
        lab_vars_left[key] = np.var([np.mean(left_region_activities[br][x], axis=0) for x in idx], axis=0, ddof=0)
        lab_vars_right[key] = np.var([np.mean(right_region_activities[br][x], axis=0) for x in idx], axis=0, ddof=0)
    var_within_left = np.mean(list(lab_vars_left.values()), axis=0)
    var_within_right = np.mean(list(lab_vars_right.values()), axis=0)


    plt.figure(figsize=(11, 8))
    plt.plot(activity_left.tscale[kernel_len-1:], var_across_left / var_within_left, label='Left')
    plt.plot(activity_left.tscale[kernel_len-1:], var_across_right / var_within_right, label='Right')
    # plt.plot(activity_left.tscale[kernel_len-1:], var_within_right, label='var within Right')
    # plt.plot(activity_left.tscale[kernel_len-1:], var_across_right, label='var across Right')
    plt.legend()

    plt.axvline(0, c='k', alpha=0.5)
    plt.ylabel("Var_across / var_within", size=18)
    plt.xlabel('Time around {} onset [s]'.format(event), size=18)
    plt.title("PSTH variance in {}".format(br), size=18)
    plt.xticks(fontsize=16)
    # plt.ylim(bottom=-0.5, top=4)
    sns.despine()

    plt.tight_layout()
    plt.savefig(FIG_PATH + "psth variance {} {}".format(event, br))
    plt.show()
