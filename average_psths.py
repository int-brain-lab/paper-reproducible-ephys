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
traj = query(behavior=True)
names = []

kernel_len = 10
kernel = np.exp(-np.arange(kernel_len) * 0.45)
colors = plt.get_cmap('gist_ncar')
errbar_kwargs={'color': 'blue', 'alpha': 0.5}

for count, t in enumerate(traj):
    eid = t['session']['id']
    probe = t['probe_name']

    if t['session']['subject'] == 'SWC_060':
        continue



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
        mask = np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), br), clusters['metrics']['label'] == 1)
        # TODO: 0.1 spikes/s criterion, mikroV threshold
        if mask.sum() < 4:
            continue

        if t['session']['subject'] not in names:
            names.append(t['session']['subject'])
        print("{} good {}, {} units".format(t['session']['subject'], br, mask.sum()))
        activity_left, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_left, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
        bars_left = activity_left.stds
        activity_right, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_right, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)

        activity_pre, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], base_line_times, pre_time=0.4, post_time=-0.2, smoothing=0, bin_size=0.01)
        baseline = np.mean(activity_pre.means, axis=1)

        activity_right_norm = ((activity_right.means.T - baseline) / (1 + baseline)).T
        activity_left_norm = ((activity_left.means.T - baseline) / (1 + baseline)).T
        bars_left = (bars_left.T / (1 + baseline)).T

        left_region_activities[br].append(activity_left_norm)
        right_region_activities[br].append(activity_right_norm)
        region_names[br].append(t['session']['subject'])

        if event == 'Stim':
            activity_0, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, np.arange(len(clusters.metrics))[mask], event_times_0, pre_time=pre_time, post_time=post_time, smoothing=0, bin_size=0.01)
            activity_zero_norm = ((activity_0.means.T - baseline) / (1 + baseline)).T
            zero_region_activities[br].append(activity_zero_norm)

        # if t['session']['subject'] == "DY_009" and br == 'LP':
        #     for n in range(activity_left_norm.shape[0]):
        #         plt.figure(figsize=(11, 8))
        #
        #         # plt.plot(activity_left.tscale[kernel_len-1:], activity_left_norm[n][kernel_len-1:], c='b', label='Left Stim unsmoothed')
        #         plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_left_norm[n])[kernel_len-1:-kernel_len+1], label='Left Stim')
        #         plt.gca().fill_between(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_left_norm[n])[kernel_len-1:-kernel_len+1] - bars_left[n][kernel_len-1:],
        #                                np.convolve(kernel, activity_left_norm[n])[kernel_len-1:-kernel_len+1] + bars_left[n][kernel_len-1:], **errbar_kwargs)
        #         plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_right_norm[n])[kernel_len-1:-kernel_len+1], label='Right Stim')
        #         plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_zero_norm[n])[kernel_len-1:-kernel_len+1], label='Zero Stim')
        #
        #         plt.axvline(0, c='k', alpha=0.5)
        #         plt.ylabel("Normalised firing rate", size=25)
        #         plt.xlabel('Time around {} onset in s'.format(event), size=25)
        #         plt.title("{} neuron in {}".format(t['session']['subject'], br), size=25)
        #         plt.xticks(fontsize=20)
        #         plt.legend(frameon=False, fontsize=22)
        #         plt.ylim(bottom=-1, top=7.5)
        #         sns.despine()
        #         if n == 3:
        #             quit()
        #
        #         plt.tight_layout()
        #         plt.show()

name2colors = dict(zip(names, [colors(i / len(names)) for i in range(len(names))]))
name2line = dict(zip(names, ['-' if i % 2 == 0 else '--' for i in range(len(names))]))
# markerstyles = ['s', 'P', '*', 'D', 'v', '^', '<', '>', 'X', '|']

# for i, br in enumerate(regions):
#
#     left_act = np.vstack(left_region_activities[br])
#     right_act = np.vstack(right_region_activities[br])
#
#     for marker, left_act, right_act in zip(markerstyles, left_region_activities[br], right_region_activities[br]):
#         for j, (l, r) in enumerate(zip(left_act, right_act)):
#             plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, l)[kernel_len-1:-kernel_len+1], c='b', alpha=0.25)
#             plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, r)[kernel_len-1:-kernel_len+1], c='orange', alpha=0.25)
#             plt.plot(activity_left.tscale[kernel_len-1::10], np.convolve(kernel, l)[kernel_len-1:-kernel_len+1:10], c='b', alpha=0.25, marker=marker, ls='', markersize=8)
#             plt.plot(activity_left.tscale[kernel_len-1::10], np.convolve(kernel, r)[kernel_len-1:-kernel_len+1:10], c='orange', alpha=0.25, marker=marker, ls='', markersize=8)
#             plt.axvline(0, c='k', alpha=0.5)
#             plt.axvline(0.2, c='k', alpha=0.25)
#             plt.ylabel("Normalised firing rate", size=18)
#             plt.xlabel('Time around stimulus onset', size=18)
#             plt.title("Mice averages in {}, neuron {}".format(br, j), size=18)
#             plt.xticks(fontsize=16)
#             plt.ylim(bottom=0, top=4)
#             sns.despine()
#
#             plt.tight_layout()
#             plt.savefig("./neuron_figures/Single neuron {} baselined, region {}, neuron {}".format(event, br, j))
#             plt.close()

region_mice = {}
region_mice['LP'] = ['SWC_054', 'SWC_052', 'SWC_058', 'CSHL058', 'CSHL059', 'CSHL045', 'CSHL052', 'DY_018', 'DY_009', 'DY_016']
region_mice['CA1'] = ['CSH_ZAD_029', 'CSH_ZAD_026', 'CSH_ZAD_019', 'CSHL058', 'CSHL059', 'CSHL045', 'CSHL049', 'CSHL051', 'CSHL052', 'DY_018', 'DY_009', 'DY_016', 'DY_013', 'DY_020']
region_mice['VISa'] = ['CSHL058', 'CSHL059', 'CSHL045', 'CSHL051', 'CSHL049', 'DY_018', 'DY_020', 'DY_016']

region_colors = {}
region_colors['LP'] = ['r', 'r', 'r', 'b', 'b', 'b', 'b', 'g', 'g', 'g']
region_colors['CA1'] = ['m', 'm', 'm', 'b', 'b', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'g', 'g']
region_colors['VISa'] = ['b', 'b', 'b', 'b', 'b', 'g', 'g', 'g']

region_styles = {}
region_styles['LP'] = ['-', '--', '-.', '-', '--', '-.', ':', '-', '--', '-.']
region_styles['CA1'] = ['-', '--', '-.', '-', '--', '-.', ':', '-', '-', '-', '--', '-.', ':', '-']
region_styles['VISa'] = ['-', '--', '-.', ':', '-', '-', '--', '-.']

name2num = {}
name2num['LP'] = dict(zip(region_mice['LP'], range(len(region_mice['LP']))))
name2num['CA1'] = dict(zip(region_mice['CA1'], range(len(region_mice['CA1']))))
name2num['VISa'] = dict(zip(region_mice['VISa'], range(len(region_mice['VISa']))))

# TEMP:
print('warning, limited regions')
regions = ['LP', 'CA1', 'VISa']

for i, br in enumerate(regions):

    all_left_act = np.vstack(left_region_activities[br])
    all_right_act = np.vstack(right_region_activities[br])
    plt.figure(figsize=(11, 8))
    # plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, np.mean(all_left_act, axis=0))[kernel_len-1:-kernel_len+1], label="{} left".format(event), c='b', lw=3.5)
    # plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, np.mean(all_right_act, axis=0))[kernel_len-1:-kernel_len+1], label="{} right".format(event), c='orange', lw=3.5)
    if event == 'Stim':
        all_0_act = np.vstack(zero_region_activities[br])
        # plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, np.mean(all_0_act, axis=0))[kernel_len-1:-kernel_len+1], label="{} 0".format(event), c='k', lw=3.5)

    for name, left_act, right_act in zip(region_names[br], left_region_activities[br], right_region_activities[br]):
        if name not in region_mice[br]:
            continue
        plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, np.mean(left_act, axis=0))[kernel_len-1:-kernel_len+1], c=region_colors[br][name2num[br][name]], linestyle=region_styles[br][name2num[br][name]], label=name)
        # plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, np.mean(right_act, axis=0))[kernel_len-1:-kernel_len+1], c=region_colors[br][name2num[br][name]], linestyle=region_styles[br][name2num[br][name]])
        #plt.plot(activity_left.tscale[kernel_len-1::10], np.convolve(kernel, np.mean(left_act, axis=0))[kernel_len-1:-kernel_len+1:10], c='b', alpha=0.25, marker=marker, ls='', markersize=8)
        #plt.plot(activity_left.tscale[kernel_len-1::10], np.convolve(kernel, np.mean(right_act, axis=0))[kernel_len-1:-kernel_len+1:10], c='orange', alpha=0.25, marker=marker, ls='', markersize=8)
    if event == 'Stim':
        for name, zero_act in zip(region_names[br], zero_region_activities[br]):
            if name not in region_mice[br]:
                continue
            # plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, np.mean(zero_act, axis=0))[kernel_len-1:-kernel_len+1], c=region_colors[br][name2num[br][name]], linestyle=region_styles[br][name2num[br][name]])

    plt.axvline(0, c='k', alpha=0.5)
    plt.axvline(0.125, color='r')
    plt.annotate('Left stim-\nulus onset', (0.01, 3.0), alpha=0.7, fontsize=25)
    # plt.axvline(0.2, c='k', alpha=0.25)
    plt.ylabel("Normalised firing rate", size=25)
    plt.xlabel('Time (seconds)'.format(event), size=25)
    plt.title("Mice averages in {}".format(br), size=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.gca().legend(handles, labels, frameon=False, fontsize=17)
    plt.ylim(bottom=-0.5, top=4)
    sns.despine()

    plt.tight_layout()
    plt.savefig(FIG_PATH + "PSTHS_{}_baselined_region_{}".format(event, br))
    plt.show()
