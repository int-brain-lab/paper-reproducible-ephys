from oneibl.one import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import brainbox as bb
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query
import pandas as pd


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
lab_names = {x: {} for x in regions}
traj = query(behavior=True)
names = []

kernel_len = 10
kernel = np.exp(-np.arange(kernel_len) * 0.75)

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
        event_times_left = times[contrast_L > 0]
        event_times_right = times[contrast_R > 0]
        event_times_0 = times[np.logical_or(contrast_R == 0, contrast_L == 0)]
        pre_time, post_time = 0.2, 0.4
    elif event == 'Block':
        times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = one.load(eid, dataset_types=['trials.stimOn_times'])[0]
        base_line_times = base_line_times[~np.isnan(base_line_times)]
        block_prob = one.load(eid, dataset_types=['trials.probabilityLeft'])
        event_times_left = times[block_prob[0] == 0.8]
        event_times_right = times[block_prob[0] == 0.2]
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
        pre_time, post_time = 0.4, 0.2

    cluster_regions = channels.acronym[clusters.channels]

    for br in regions:
        try:
            mask = np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), br), clusters['metrics']['label'] == 1)
        except KeyError:
            continue
        # TODO: 0.1 spikes/s criterion, mikroV threshold
        if mask.sum() < 4:
            continue

        if t['session']['subject'] not in names:
            names.append(t['session']['subject'])
        print("{} good {}, {} units".format(t['session']['subject'], br, mask.sum()))
        activity_left, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_left, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
        activity_right, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_right, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)

        activity_pre, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], base_line_times, pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01)
        baseline = np.mean(activity_pre.means, axis=1)

        activity_right_norm = ((activity_right.means.T - baseline) / (1 + baseline)).T
        activity_left_norm = ((activity_left.means.T - baseline) / (1 + baseline)).T

        left_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=activity_left_norm)[:, kernel_len-1:-kernel_len+1])
        right_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=activity_right_norm)[:, kernel_len-1:-kernel_len+1])
        region_names[br].append(t['session']['subject'])

        if event == 'Stim':
            activity_0, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_0, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
            activity_zero_norm = ((activity_0.means.T - baseline) / (1 + baseline)).T
            zero_region_activities[br].append(np.apply_along_axis(lambda m: np.convolve(m, kernel), axis=1, arr=activity_zero_norm)[:, kernel_len-1:-kernel_len+1])

        region_counts[br] += 1
        if t['session']['lab'] not in lab_indices[br]:
            lab_indices[br][t['session']['lab']] = [region_counts[br]]
            lab_names[br][t['session']['lab']] = [t['session']['subject']]
        else:
            lab_indices[br][t['session']['lab']].append(region_counts[br])
            lab_names[br][t['session']['lab']].append(t['session']['subject'])

pickle.dump((left_region_activities, right_region_activities, lab_indices, lab_names, activity_left), (open("../data/info {}.p".format(event), "wb")))
left_region_activities, right_region_activities, lab_indices, lab_names, activity_left = pickle.load(open("../data/info {}.p".format(event), "rb"))

for i, br in enumerate(regions):
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

    # for i in range(n_times):
    #     result = anova2nested(all_left_act[:, i], lev_one, lev_two, equal_var=True)  # code enforces equal_var, is this appropriate for us?

    # labs = [2.0, 4.0, 5.0]
    # mouse = [[2.0, 3.0, 5.0, 6.0, 11.0, 14.0], [7.0, 8.0, 9.0, 12.0, 17.0], [10.0, 13.0, 16.0]]
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
        # df.to_csv("../neural_test_{}.csv".format(j))

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
                plt.scatter(np.linspace(0, 1, (df['mouse'] == m).sum()) * 2 * dist - dist + mouse_counter, df[df['mouse'] == m]['fr'], color='k', label=label)
                label = "Mice means" if j == 0 and i == 0 else None
                plt.plot([mouse_counter - dist, mouse_counter + dist], [emp_mice_mean, emp_mice_mean], 'b', label=label)
                plt.annotate(n, (mouse_counter - 0.3, -1.5), fontsize=fs-13)
                mouse_counter += 1
            label = "Lab means" if i == 0 else None
            emp_lab_means.append(np.mean(emp_mice_means[i]))
            plt.plot([mouse_counter - len(mouse[i]), mouse_counter - 1], [emp_lab_means[-1], emp_lab_means[-1]], c='r', label=label)
        emp_intercept = np.mean(emp_lab_means)
        plt.axhline(emp_intercept, c='k', label="Mean")

        plt.legend(fontsize=fs, frameon=False)
        plt.title("{}, {:.3f} seconds after {} onset".format(br, activity_left.tscale[kernel_len-1+index], event), size=fs)
        plt.xlabel("Neurons, Mice, Labs", size=fs)
        plt.xticks([])
        plt.ylabel("Smoothed normalised firing rate", size=fs)
        plt.yticks(fontsize=fs-2)
        plt.ylim(bottom=-2, top=8)

        sns.despine()
        plt.tight_layout()
        plt.savefig(FIG_PATH + "{}, {:.3f} seconds after {} onset.png".format(br, activity_left.tscale[kernel_len-1+index], event))
        plt.close()
