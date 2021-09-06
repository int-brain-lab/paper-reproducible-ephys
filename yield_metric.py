from one.api import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query
from ibllib.atlas import BrainRegions


regions = ["CTX", "CNU", "TH", "HY", "MB", "HB", "CB"]
ids = [688, 623, 549, 1097, 313, 1065, 512]
br = BrainRegions()
region2top_level = {}
for r, id in zip(regions, ids):
    children = br.descendants(id).acronym
    for c in children:
        region2top_level[c] = r


def default_dict(x, d):
    if x in d:
        return d[x]
        return None


def vec_translate(a, d):
    return np.vectorize(default_dict)(a, d)


spike_sorting = "1"

one = ONE()

traj = query(behavior=True)
names = []

probe_mins = []
region_mins = {}
for br in regions:
    region_mins[br] = []
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

    cluster_regions = vec_translate(clusters.acronym, region2top_level)  # == channels.acronym[clusters.channels]
    channel_regions = vec_translate(channels.acronym, region2top_level)

    for br in regions:
        if spike_sorting == '1':

            mask = np.logical_and(cluster_regions == br, clusters['metrics']['label'] == 1)
        elif spike_sorting == '2':
            print("Not implemented")
            quit()
        neurons_in_region = np.sum(mask)
        channels_in_region = np.sum(channel_regions == br)
        if channels_in_region >= 10:
            probe_mins[-1] = min(probe_mins[-1], neurons_in_region / channels_in_region)
            region_mins[br].append(neurons_in_region / channels_in_region)
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

plt.step(pos, vals, where='post', label='Probe')
for br in regions:
    pos, vals = cumulative_and_position(region_mins[br])
    plt.step(pos, vals, where='post', label=br)
plt.ylim(bottom=0)
plt.ylabel("Recordings lost to criterion", size=18)
plt.xlabel("Minimum yield per channel criterion", size=18)
plt.show()
