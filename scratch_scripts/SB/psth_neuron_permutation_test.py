from oneibl.one import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
from permutation_test import permut_test
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import brainbox as bb
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query
import pandas as pd


def labs_dist(data, labs, mice):
    lab_means = []
    for lab in np.unique(labs):
        lab_means.append(np.mean(data[labs == lab]))
    lab_means = np.array(lab_means)
    return np.sum(np.abs(lab_means - np.mean(lab_means)))


def process_peths(spikes, clusters, event_times, mask, pre_time, post_time):
    activities = []
    for etime in event_times:
        a, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask],
                                             etime, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
        activities.append(a)
    return activities


def normalise_neurons(activities, spikes, clusters, mask, base_line_times):
    activity_pre, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters,
                                                    np.arange(len(clusters.metrics))[mask], base_line_times,
                                                    pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01)
    baseline = np.mean(activity_pre.means, axis=1)

    normed_activities = []
    for a in activities:
        normed_activities.append(((a.means.T - baseline) / (1 + baseline)).T)

    return normed_activities


regions = ['CA1', 'VISa', 'DG', 'LP', 'PO']

one = ONE()

event = 'Stim'
left_region_activities = {x: [] for x in regions}
right_region_activities = {x: [] for x in regions}
zero_region_activities = {x: [] for x in regions}
region_names = {x: [] for x in regions}
region_counts = {x: -1 for x in regions}
lab_indices = {x: {} for x in regions}
lab_names = {x: {} for x in regions}
traj = query(behavior=True)
names = []

kernel_len = 10
kernel = np.exp(-np.arange(kernel_len) * 0.45)

for count, t in enumerate(traj):
    eid = t['session']['id']
    probe = t['probe_name']

    # load data
    try:
        spikes, clusters, channels = pickle.load(open("../data/data_{}.p".format(eid), "rb"))
    except FileNotFoundError:
        try:
            spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, force=True)
            spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
            pickle.dump((spikes, clusters, channels), (open("../data/data_{}.p".format(eid), "wb")))
        except KeyError:
            print(eid)
            continue

    if event == 'Stim':
        times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = base_line_times[~np.isnan(base_line_times)]
        contrast_L, contrast_R = one.load(eid, dataset_types=['trials.contrastLeft', 'trials.contrastRight'])
        contrast_L, contrast_R = contrast_L[~np.isnan(times)], contrast_R[~np.isnan(times)]
        times = times[~np.isnan(times)]
        event_times_left = times[contrast_L > 0]
        event_times_right = times[contrast_R > 0]
        event_times_0 = times[np.logical_or(contrast_R == 0, contrast_L == 0)]
        event_times = [event_times_left, event_times_right, event_times_0]
        pre_time, post_time = 0.2, 0.4
    elif event == 'Block':
        times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = base_line_times[~np.isnan(base_line_times)]
        block_prob = one.load(eid, dataset_types=['trials.probabilityLeft'])
        event_times_left = times[block_prob[0] == 0.8]
        event_times_right = times[block_prob[0] == 0.2]
        event_times = [event_times_left, event_times_right]
        pre_time, post_time = 0.2, 0.4
    elif event == 'Move':
        times = one.load(eid, dataset_types=['trials.firstMovement_times'])[0]
        base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = base_line_times[~np.isnan(base_line_times)]
        choice = one.load(eid, dataset_types=['trials.choice'])[0]
        if (~np.isnan(times)).sum() < 300:
            continue
        choice = choice[~np.isnan(times)]
        times = times[~np.isnan(times)]
        event_times_left = times[choice == -1]
        event_times_right = times[choice == 1]
        event_times = [event_times_left, event_times_right]
        pre_time, post_time = 0.4, 0.2

    assert np.sum(np.isnan(event_times_left)) == 0
    assert np.sum(np.isnan(event_times_right)) == 0
    assert np.sum(np.isnan(base_line_times)) == 0
    cluster_regions = channels.acronym[clusters.channels]

    for br in regions:
        try:
            mask = np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), br), clusters['metrics']['label'] == 1)
        except KeyError:
            continue

        # TODO: 0.1 spikes/s criterion, mikroV threshold
        if mask.sum() < 4:
            continue

        activities = process_peths(spikes, clusters, event_times, mask, pre_time, post_time)
        normed_activities = normalise_neurons(activities, spikes, clusters, mask, base_line_times)

        left_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=normed_activities[0])[:, kernel_len-1:-kernel_len+1])
        right_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=normed_activities[1])[:, kernel_len-1:-kernel_len+1])
        if event == 'Stim':
            zero_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=normed_activities[2])[:, kernel_len-1:-kernel_len+1])

        region_names[br].append(t['session']['subject'])
        if t['session']['subject'] not in names:
            names.append(t['session']['subject'])
        region_counts[br] += 1
        if t['session']['lab'] not in lab_indices[br]:
            lab_indices[br][t['session']['lab']] = [region_counts[br]]
            lab_names[br][t['session']['lab']] = [t['session']['subject']]
        else:
            lab_indices[br][t['session']['lab']].append(region_counts[br])
            lab_names[br][t['session']['lab']].append(t['session']['subject'])

pickle.dump((left_region_activities, right_region_activities, zero_region_activities, lab_indices, lab_names, activities[0].tscale), (open("../data/info {}.p".format(event), "wb")))
left_region_activities, right_region_activities, zero_region_activities, lab_indices, lab_names, tscale = pickle.load(open("../data/info {}.p".format(event), "rb"))
plt.figure(figsize=(16, 9))

# TEMP!!:
print('warning')
regions = ['CA1', 'VISa', 'LP']
region2color = dict(zip(regions, ['b', 'orange', 'green']))

for br in regions:
    labs = list(lab_indices[br].keys())
    rich_labs = []
    lab2num = dict(zip(labs, range(len(labs))))
    n_sess = len(left_region_activities[br])
    sess2lab = {}
    for k, v in lab_indices[br].items():
        for num in v:
            sess2lab[num] = k
        if len(v) > 2:
            rich_labs.append(k)

    all_left_act = np.vstack(left_region_activities[br])
    all_right_act = np.vstack(right_region_activities[br])
    if event == 'Stim':
        all_zero_act = np.vstack(zero_region_activities[br])

    n_times = all_left_act.shape[1]
    left_f_lev_one = np.zeros(n_times)
    left_f_lev_two = np.zeros(n_times)
    right_f_lev_one = np.zeros(n_times)
    right_f_lev_two = np.zeros(n_times)

    lev_one = np.zeros(all_left_act.shape[0], dtype=np.int) - 1
    lev_two = np.zeros(all_left_act.shape[0], dtype=np.int) - 1

    neur_sum = 0
    for j in range(n_sess):
        n_neurons = left_region_activities[br][j].shape[0]
        lev_one[neur_sum:neur_sum + n_neurons] = lab2num[sess2lab[j]]
        lev_two[neur_sum:neur_sum + n_neurons] = j
        neur_sum += n_neurons

    labs = []
    mouse = []
    names = []
    for l in rich_labs:
        labs.append(lab2num[l])
        mouse.append(lab_indices[br][l])
        names.append(lab_names[br][l])
    dist = 0.25
    fs = 22

    p_vals_left = np.zeros(n_times)
    p_vals_right = np.zeros(n_times)
    for index in range(n_times):
        df = pd.DataFrame(np.array([all_left_act[:, index], lev_one, lev_two]).T, columns=['fr', 'lab', 'mouse'])
        df_other = pd.DataFrame(np.array([all_right_act[:, index], lev_one, lev_two]).T, columns=['fr', 'lab', 'mouse'])

        emp_mice_means_left = []
        emp_mice_means_right = []
        lab_array = []
        mouse_array = []
        for i, l in enumerate(labs):
            for j, (m, n) in enumerate(zip(mouse[i], names[i])):
                emp_mice_means_left.append(np.mean(df[df['mouse'] == m]['fr']))
                emp_mice_means_right.append(np.mean(df_other[df_other['mouse'] == m]['fr']))
                lab_array.append(l)
                mouse_array.append(m)
        p_vals_left[index] = permut_test(data=np.array(emp_mice_means_left), metric=labs_dist, labels1=np.array(lab_array), labels2=np.array(mouse_array))
        p_vals_right[index] = permut_test(data=np.array(emp_mice_means_right), metric=labs_dist, labels1=np.array(lab_array), labels2=np.array(mouse_array))

    plt.plot(tscale[kernel_len-1:], p_vals_left, label="{} stim left".format(br), c=region2color[br])
    # plt.plot(activity_left.tscale[kernel_len-1:], p_vals_right, label="{} stim right".format(br), c=region2color[br], ls='--')

#plt.legend(fontsize=fs, frameon=False)
#plt.axhline(0.05, color='r')
plt.axvline(0.235, color='r')
#plt.title("Permutation p values, {} left onset".format(event), size=fs+3)
plt.xlabel("Time in s", size=fs+3)
plt.ylabel("p value", size=fs+3)
plt.yticks(fontsize=fs)
plt.xticks(fontsize=fs)
plt.ylim(bottom=0, top=1)

sns.despine()
plt.tight_layout()
plt.savefig(FIG_PATH + "newer Permutation p values, {} left onset.png".format(event))
plt.show()
