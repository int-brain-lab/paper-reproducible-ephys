from one.api import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
from brainbox.singlecell import calculate_peths
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query
import pandas as pd


regions = ['LP', 'CA1', 'VISa', 'DG', 'PO']
spike_sorting = "1"

one = ONE()

traj = query(behavior=True)
names = []

probe_mins = []
missing_stuff = []
for count, t in enumerate(traj):
    eid = t['session']['id']
    probe = t['probe_name']
    probe_mins.append(1)

    # load data
    try:
        spikes, clusters, channels = pickle.load(open("../data/data_{}_sorting_{}.p".format(eid, spike_sorting), "rb"))
    except FileNotFoundError:
        try:
            if spike_sorting == '1':
                spk, clus, chn = load_spike_sorting_with_channel(eid, one=one)
            elif spike_sorting == '2':
                spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, spike_sorter='ks2_preproc_tests')
            spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
            if spikes is None or clusters is None or channels is None:
                missing_stuff.append(eid)
                print("no spikes or clusters or channels")
                continue
            pickle.dump((spikes, clusters, channels), (open("../data/data_{}_sorting_{}.p".format(eid, spike_sorting), "wb")))
        except KeyError:
            print("Keyerror")
            print(eid)
            continue
        except Exception as e:
            print(e)
            print(eid)
            continue

    cluster_regions = clusters.acronym  # == channels.acronym[clusters.channels]

    for br in regions:
        if spike_sorting == '1':
            mask = np.logical_and(np.chararray.startswith(cluster_regions.astype('U9'), br), clusters['metrics']['label'] == 1)
        elif spike_sorting == '2':
            print("Not implemented")
            quit()
        neurons_in_region = np.sum(mask)
        channels_in_region = np.sum(np.chararray.startswith(channels.acronym.astype('U9'), br))
        if channels_in_region != 0:
            probe_mins[-1] = min(probe_mins[-1], neurons_in_region / channels_in_region)
            print("{} neurons/channel ({}, {}) in {}".format(neurons_in_region / channels_in_region, neurons_in_region, channels_in_region, br))
    print()

def cumulative_and_position(l):
    """Take a list, find out how many things there are at specific values."""
    pos = [min(l)]
    vals = [0]
    for x in sorted(l):
        if pos[-1] == x:
            vals[-1] += 1
        else:
            pos.append(x)
            vals.append(vals[-1] + 1)
    return pos, vals

print(probe_mins)
pos, vals = cumulative_and_position(probe_mins)

plt.step(pos, vals)
plt.ylim(bottom=0)
plt.ylabel("Recordings lost to criterion", size=18)
plt.xlabel("Minimum yield per channel criterion", size=18)
plt.show()
