from oneibl.one import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
from brainbox.singlecell import calculate_peths
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query
import pandas as pd


def process_peths(spikes, clusters, event_times, mask, pre_time, post_time):
    """Get spikecounts for specified times, put into right format."""
    activities = []
    for etime in event_times:
        a, _ = calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask],
                               etime, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
        activities.append(a)
    return activities


def normalise_neurons(activities, spikes, clusters, mask, base_line_times):
    """Normalise neurons by computing baseline from specific time, then norm_fr = (fr - bs) / (1 + bs)."""
    activity_pre, _ = calculate_peths(spikes.times, spikes.clusters,
                                      np.arange(len(clusters.metrics))[mask], base_line_times,
                                      pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01)
    baseline = np.mean(activity_pre.means, axis=1)

    normed_activities = []
    for a in activities:
        normed_activities.append(((a.means.T - baseline) / (1 + baseline)).T)

    return normed_activities


regions = ['CA1', 'VISa', 'DG', 'LP', 'PO']
apply_baseline = True
event = 'Move'

kernel_len = 10
kernel = np.exp(-np.arange(kernel_len) * 0.45)
kernel_area = np.sum(kernel)
one = ONE()

left_region_activities = {x: [] for x in regions}
right_region_activities = {x: [] for x in regions}
zero_region_activities = {x: [] for x in regions}
region_names = {x: [] for x in regions}
region_counts = {x: -1 for x in regions}
lab_indices = {x: {} for x in regions}
lab_names = {x: {} for x in regions}
traj = query(behavior=True)
names = []


for count, t in enumerate(traj):
    eid = t['session']['id']
    probe = t['probe_name']
    print(eid)

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
        if apply_baseline:
            normed_activities = normalise_neurons(activities, spikes, clusters, mask, base_line_times)
        else:
            normed_activities = [a.means for a in activities]

        left_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=normed_activities[0])[:, kernel_len-1:-kernel_len+1] / kernel_area)
        right_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=normed_activities[1])[:, kernel_len-1:-kernel_len+1] / kernel_area)
        if event == 'Stim':
            zero_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=normed_activities[2])[:, kernel_len-1:-kernel_len+1] / kernel_area)

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

pickle.dump((left_region_activities, right_region_activities, zero_region_activities, lab_indices, lab_names, activities[0].tscale, names), (open("../data/info {}_baseline_{}.p".format(event, apply_baseline), "wb")))
left_region_activities, right_region_activities, zero_region_activities, lab_indices, lab_names, tscale, names = pickle.load(open("../data/info {}_baseline_{}.p".format(event, apply_baseline), "rb"))


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

    lev_one = np.zeros(all_left_act.shape[0], dtype=np.int32) - 1
    lev_two = np.zeros(all_left_act.shape[0], dtype=np.int32) - 1

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

    for index in range(n_times):
        df = pd.DataFrame(np.array([all_left_act[:, index], lev_one, lev_two]).T, columns=['fr', 'lab', 'mouse'])
        df_other = pd.DataFrame(np.array([all_right_act[:, index], lev_one, lev_two]).T, columns=['fr', 'lab', 'mouse'])

        emp_mice_means = []
        emp_lab_means = []
        mouse_counter = 0
        plt.figure(figsize=(16, 9))
        for i, l in enumerate(labs):
            emp_mice_means.append([])
            for j, (m, n) in enumerate(zip(mouse[i], names[i])):
                emp_mice_means[i].append(np.mean(df[df['mouse'] == m]['fr']))
                emp_mice_mean = np.mean(df[df['mouse'] == m]['fr'])
                label = "Neurons" if j == 0 and i == 0 else None
                # plt.scatter(np.linspace(0, 1, (df_other['mouse'] == m).sum()) * 2 * dist - dist + mouse_counter, np.clip(df_other[df_other['mouse'] == m]['fr'], -2, 8), color='gray', s=45)
                # pos = df[df['mouse'] == m]['fr'] > 8
                # if np.sum(pos) > 0:
                #     plt.scatter(np.linspace(0, 1, (df['mouse'] == m).sum())[pos] * 2 * dist - dist + mouse_counter, np.clip(df[df['mouse'] == m]['fr'], -2, 8)[pos], color='r', s=70)
                # plt.scatter(np.linspace(0, 1, (df['mouse'] == m).sum()) * 2 * dist - dist + mouse_counter, np.clip(df[df['mouse'] == m]['fr'], -2, 8), color='k', label=label)
                plt.scatter(np.linspace(0, 1, (df['mouse'] == m).sum()) * 2 * dist - dist + mouse_counter, df[df['mouse'] == m]['fr'], color='k', label=label)
                label = "Mice means" if j == 0 and i == 0 else None

                plt.plot([mouse_counter - dist, mouse_counter + dist], [emp_mice_mean, emp_mice_mean], 'b', label=label)
                plt.annotate(n, (mouse_counter - 0.3, -1.8), fontsize=fs-7)
                mouse_counter += 1
            label = "Lab means" if i == 0 else None
            emp_lab_means.append(np.mean(emp_mice_means[i]))
            plt.plot([mouse_counter - len(mouse[i]), mouse_counter - 1], [emp_lab_means[-1], emp_lab_means[-1]], c='r', label=label)
        emp_intercept = np.mean(emp_lab_means)
        plt.axhline(emp_intercept, c='k', label="Mean", zorder=-1)

        plt.legend(fontsize=fs, frameon=False)
        plt.title("{}, {:.3f} seconds after {} left onset".format(br, tscale[kernel_len-1+index], event), size=fs)
        plt.xlabel("Neurons, Mice, Labs", size=fs)
        plt.xticks([])
        plt.ylabel("Smoothed normalised firing rate", size=fs)

        # plt.gca().set_yticklabels([-2, 0, 2, 4, 6, '>=8'])

        plt.yticks(fontsize=fs-2)
        plt.ylim(bottom=-2)

        sns.despine()
        plt.tight_layout()
        plt.savefig(FIG_PATH + "{}, {:.3f} seconds after {} left onset_baseline_{}.png".format(br, tscale[kernel_len-1+index], event, apply_baseline))
        plt.close()
