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


regions = ['CA1', 'VISa', 'DG', 'LP', 'PO']
apply_baseline = True
event = 'Stim'
average = ['Mean', 'Median'][1]

kernel_len = 10
kernel = np.exp(-np.arange(kernel_len) * 0.45)
kernel_area = np.sum(kernel)
one = ONE()

        # if t['session']['subject'] == "DY_009" and br == 'LP':
        #     for n in range(activity_left_norm.shape[0]):
        #         plt.figure(figsize=(11, 8))
        #
        #         # plt.plot(activity_left.tscale[kernel_len-1:], activity_left_norm[n][kernel_len-1:], c='b', label='Left Stim unsmoothed')
        #         plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_left_norm[n])[kernel_len-1:-kernel_len+1], label='Left Stim')
        #         plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_right_norm[n])[kernel_len-1:-kernel_len+1], label='Right Stim')
        #         plt.plot(activity_left.tscale[kernel_len-1:], np.convolve(kernel, activity_zero_norm[n])[kernel_len-1:-kernel_len+1], label='Zero Stim')
        #
        #         plt.axvline(0, c='k', alpha=0.5)
        #         plt.ylabel("Normalised firing rate", size=25)
        #         plt.xlabel('Time around {} onset in s'.format(event), size=25)
        #         plt.title("{} neuron {} in {}".format(t['session']['subject'], np.where(mask)[0][n], br), size=25)
        #         plt.xticks(fontsize=20)
        #         plt.legend(frameon=False, fontsize=22)
        #         # plt.ylim(bottom=-1, top=7.5)
        #         sns.despine()
        #
        #         plt.tight_layout()
        #         plt.show()
        #     quit()

left_region_activities, right_region_activities, zero_region_activities, lab_indices, lab_names, tscale, names = pickle.load(open("../data/info {}_baseline_{}.p".format(event, apply_baseline), "rb"))
colors = plt.get_cmap('gist_ncar')
name2colors = dict(zip(names, [colors(i / len(names)) for i in range(len(names))]))
name2line = dict(zip(names, ['-' if i % 2 == 0 else '--' for i in range(len(names))]))
# markerstyles = ['s', 'P', '*', 'D', 'v', '^', '<', '>', 'X', '|']

region_mice = {}
region_mice['LP'] = ['SWC_054', 'SWC_052', 'SWC_058', 'ZFM-01592', 'ZM_2241', 'ZFM-01936', 'CSHL058', 'CSHL059', 'CSHL045', 'CSHL052', 'DY_018', 'DY_009', 'DY_016']
region_mice['CA1'] = ['CSH_ZAD_029', 'CSH_ZAD_026', 'CSH_ZAD_019', 'CSHL058', 'CSHL059', 'CSHL045', 'CSHL049', 'CSHL051', 'CSHL052', 'DY_018', 'DY_009', 'DY_016', 'DY_013', 'DY_020']
region_mice['VISa'] = ['CSHL058', 'CSHL059', 'CSHL045', 'CSHL051', 'CSHL049', 'DY_018', 'DY_020', 'DY_016']

region_colors = {}
region_colors['LP'] = ['r', 'r', 'r', 'k', 'k', 'k', 'b', 'b', 'b', 'b', 'g', 'g', 'g']
region_colors['CA1'] = ['m', 'm', 'm', 'b', 'b', 'b', 'b', 'b', 'b', 'g', 'g', 'g', 'g', 'g']
region_colors['VISa'] = ['b', 'b', 'b', 'b', 'b', 'g', 'g', 'g']

region_styles = {}
region_styles['LP'] = ['-', '--', '-.', '-', '--', '-.', '-', '--', '-.', ':', '-', '--', '-.']
region_styles['CA1'] = ['-', '--', '-.', '-', '--', '-.', ':', '-', '-', '-', '--', '-.', ':', '-']
region_styles['VISa'] = ['-', '--', '-.', ':', '-', '-', '--', '-.']

name2num = {}
name2num['LP'] = dict(zip(region_mice['LP'], range(len(region_mice['LP']))))
name2num['CA1'] = dict(zip(region_mice['CA1'], range(len(region_mice['CA1']))))
name2num['VISa'] = dict(zip(region_mice['VISa'], range(len(region_mice['VISa']))))

# TEMP:
print('warning, limited regions')
regions = ['LP', 'CA1', 'VISa']

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

    plt.figure(figsize=(16, 9))
    for i, l in enumerate(labs):
        for j, (m, n) in enumerate(zip(mouse[i], names[i])):
            if average == 'Mean':
                averages = np.mean(all_left_act[lev_two == m], axis=0)
            elif average == 'Median':
                averages = np.median(all_left_act[lev_two == m], axis=0)
            plt.plot(tscale[kernel_len-1:], averages, c=region_colors[br][name2num[br][n]], linestyle=region_styles[br][name2num[br][n]], label=n)
            ebar = np.std(all_left_act[lev_two == m], axis=0, ddof=1) / np.sqrt(np.sum(lev_two == m))  # TODO: what is n? number of neurons or number of trials?
            #plt.fill_between(tscale[kernel_len-1:], mean + ebar, mean - ebar, color=region_colors[br][name2num[br][n]], alpha=0.1)

    plt.axvline(0, c='k', alpha=0.5)
    # plt.annotate('Left stim-\nulus onset', (0.01, 3.0), alpha=0.7, fontsize=25)
    ylabel = "Normalised firing rate" if apply_baseline else "Firing rate"
    plt.ylabel(ylabel, size=25)
    plt.xlabel('Time (seconds)'.format(event), size=25)
    plt.title("Mice averages around {} in {} (baseline={}, {})".format(event, br, apply_baseline, average), size=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(frameon=False, fontsize=22)
    handles, labels = plt.gca().get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    plt.gca().legend(handles, labels, frameon=False, fontsize=17)
    # plt.ylim(bottom=-0.5, top=4)
    sns.despine()

    plt.tight_layout()
    plt.savefig(FIG_PATH + "PSTHS_{}_region_{}_baseline_{}_{}".format(event, br, apply_baseline, average))
    plt.show()
