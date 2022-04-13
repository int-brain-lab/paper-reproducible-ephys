from one.api import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
from brainbox.plot import peri_event_time_histogram
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import brainbox as bb
from reproducible_ephys_functions import query


one = ONE()
pre_time, post_time = 0.1, 0.35

traj = query(behavior=True)
boundary_width = 0.01
base_grey = 0.15
fs = 21

names = ['DY_018']  # , 'DY_009', 'DY_016', 'CSHL059', 'CSHL045', 'CSHL052']

for count, t in enumerate(traj):
    eid = t['session']['id']
    probe = t['probe_name']
    if t['session']['subject'] not in names:
        continue

    # load data
    try:
        spikes, clusters, channels = pickle.load(open("../data/data_{}_sorting_1.p".format(eid), "rb"))
    except FileNotFoundError:
        try:
            spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, force=True)
            spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
            pickle.dump((spikes, clusters, channels), (open("../data/data_{}.p".format(eid), "wb")))
        except KeyError:
            print(eid)
            continue

    times = one.load_object(eid, 'trials', attribute=['stimOn_times'])['stimOn_times']
    contrasts = one.load_object(eid, 'trials',  attribute=['contrastLeft', 'contrastRight'])
    contrast_L, contrast_R = contrasts['contrastLeft'], contrasts['contrastRight']
    contrast_L, contrast_R = contrast_L[~np.isnan(times)], contrast_R[~np.isnan(times)]
    times = times[~np.isnan(times)]
    event_times_left = times[np.logical_or(contrast_L >= 0, contrast_R == 0.)]  # !!! I changed this to greater OR EQUAL

    left_contrasts = contrast_L[np.logical_or(contrast_L >= 0, contrast_R == 0.)]
    left_contrasts[np.isnan(left_contrasts)] = 0.

    cluster_regions = channels.acronym[clusters.channels]
    neurons = np.where(np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), 'VIS'), clusters['metrics']['label'] == 1))[0]

    for j, neuron in enumerate(neurons):
        print("Warning, code is limited")
        if neuron != 335:
            continue
        # peri_event_time_histogram(spikes.times, spikes.clusters, event_times_left, neuron, include_raster=True)
        # plt.title(str(neuron) + " " + t['session']['subject'])
        # plt.show()

        plt.figure(figsize=(9, 9))

        plt.subplot(2, 1, 1)
        counter = 0
        contrast_count_list = [0]
        for c in [1., 0.25, 0.125, 0.0625, 0.]:
            temp = left_contrasts == c
            print("{}, count {}".format(c, np.sum(temp)))

            clu_spks = spikes.times[spikes.clusters == neuron]
            for i, time in enumerate(event_times_left[temp]):
                idx = np.bitwise_and(clu_spks >= time - pre_time, clu_spks <= time + post_time)
                event_spks = clu_spks[idx]
                plt.vlines(event_spks - time, counter - i, counter - i - 1)
            counter -= np.sum(temp)
            contrast_count_list.append(counter)
        ylabel_pos = []
        for i, c in enumerate([1., 0.25, 0.125, 0.0625, 0.]):
            top = contrast_count_list[i]
            bottom = contrast_count_list[i + 1]
            plt.fill_between([-pre_time, -pre_time + boundary_width], [top, top], [bottom, bottom],
                             zorder=3, color=str(1 - (base_grey + c * (1 - base_grey))))
            ylabel_pos.append((top - bottom) / 2 + bottom)

        plt.yticks(ylabel_pos, [1., 0.25, 0.125, 0.0625, 0.], size=fs)
        plt.axvline(0, color='k', ls='--')
        plt.xlim(left=-pre_time, right=post_time)
        plt.ylim(top=0, bottom=counter)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.tick_params(left=False, right=False, labelbottom=False, bottom=False)
        plt.title("Contrast", loc='left', size=fs+3)

        plt.subplot(2, 1, 2)
        for c in [1., 0.25, 0.125, 0.0625, 0.]:
            mask = left_contrasts == c
            psths, _ = bb.singlecell.calculate_peths(spikes.times, spikes.clusters, neurons, event_times_left[mask],
                                                     pre_time=pre_time, post_time=post_time, smoothing=0.01, bin_size=0.01)
            plt.plot(psths.tscale, psths.means[j], c=str(1 - (base_grey + c * (1 - base_grey))))
            plt.fill_between(psths.tscale,
                             psths.means[j] + psths.stds[j] / np.sqrt(np.sum(mask)),
                             psths.means[j] - psths.stds[j] / np.sqrt(np.sum(mask)),
                             color=str(1 - (base_grey + c * (1 - base_grey))), alpha=0.3)
        plt.axvline(0, color='k', ls='--')
        plt.xlim(left=-pre_time, right=post_time)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.yticks([0, 25, 50, 75], [0, 25, 50, 75], size=fs)
        plt.xticks([0., 0.1, 0.2, 0.3], [0., 0.1, 0.2, 0.3], size=fs)
        plt.ylabel("Firing rate (sp/s)", size=fs+3)
        plt.xlabel("Time from stim onset (s)", size=fs+3)
        plt.savefig("{}, {}".format(t['session']['subject'], neuron))
        plt.close()
