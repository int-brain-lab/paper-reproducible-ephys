from one.api import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query
from iblatlas.regions import BrainRegions
from brainbox.metrics.single_units import quick_unit_metrics
from brainbox.io.one import load_spike_sorting_fast


regions = ["HPF", "Isocortex", "CNU", "TH", "HY", "MB", "HB", "CB"]
ids = [1089, 315, 623, 549, 1097, 313, 1065, 512]
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

traj = query()
names = []

probe_mins = []
region_mins = {}
mice = {}
for br in regions:
    region_mins[br] = []
missing_stuff = []
for count, t in enumerate(reversed(traj)):
    eid = t['session']['id']
    pid = t['probe_insertion']
    eid2, pname = one.pid2eid(pid)
    assert eid == eid2

    # load data
    # try:
    #     spikes, clusters, channels = pickle.load(open("../data/data_{}_sorting_{}.p".format(eid, spike_sorting), "rb"))
    # except FileNotFoundError:
    #     try:
    #         if spike_sorting == '1':
    #             spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, dataset_types=['spikes.amps', 'spikes.depths'])
    #         elif spike_sorting == '2':
    #             spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, dataset_types=['spikes.amps', 'spikes.depths'], spike_sorter='ks2_preproc_tests')
    #         spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
    #         if spikes is None or clusters is None or channels is None:
    #             missing_stuff.append(eid)
    #             print("no spikes or clusters or channels")
    #             continue
    #         pickle.dump((spikes, clusters, channels), (open("../data/data_{}_sorting_{}.p".format(eid, spike_sorting), "wb")))
    #     except KeyError:
    #         print("Keyerror")
    #         print(eid)
    #         continue
    #     except Exception as e:
    #         print(e)
    #         print(eid)
    #         continue

    if eid in ['d2832a38-27f6-452d-91d6-af72d794136c', '4b7fbad4-f6de-43b4-9b15-c7c7ef44db4b',
               'c51f34d8-42f6-4c9c-bb5b-669fd9c42cd9', '7af49c00-63dd-4fed-b2e0-1b3bd945b20b',
               '0802ced5-33a3-405e-8336-b65ebc5cb07c']:
        continue
    if t['session']['subject'] == 'ibl_witten_32':
        continue

    try:
        spikes, clusters, channels = load_spike_sorting_with_channel(eid=eid, one=one, probe=pname, spike_sorter='pykilosort', dataset_types=['spikes.amps', 'spikes.depths'])
        spikes, clusters, channels = spikes[pname], clusters[pname], channels[pname]
        print("{} {} works".format(eid, pname))
    except BaseException as e:
        print("{} doesnt work {} {}".format(str(e), eid, pname))
        spk, clus, chn = load_spike_sorting_with_channel(eid, one=one, dataset_types=['spikes.amps', 'spikes.depths'])
        spikes, clusters, channels = spk[pname], clus[pname], chn[pname]

    if 'acronym' not in clusters:
        print('clusters not assigned to region ' + t['session']['subject'])
        continue
    if len(clusters.acronym) == 0:
        print('clusters not assigned to region ' + t['session']['subject'])
        continue

    cluster_regions = vec_translate(clusters['acronym'], region2top_level)  # == channels.acronym[clusters.channels]
    channel_regions = vec_translate(channels['acronym'], region2top_level)

    if 'metrics' not in clusters or clusters.metrics.shape[0] != cluster_regions.shape[0]:
        print('computing metrics')
        r = quick_unit_metrics(spikes.clusters, spikes.times, spikes.amps, spikes.depths, cluster_ids=np.arange(clusters.channels.size))
    else:
        r = clusters['metrics']

    mice[t['session']['subject']] = {}
    print(t['session']['subject'])
    probe_mins.append(1)
    for br in regions:

        mask = np.logical_and(cluster_regions == br, r['label'] == 1)

        neurons_in_region = np.sum(mask)
        channels_in_region = np.sum(channel_regions == br)
        if channels_in_region >= 10:
            probe_mins[-1] = min(probe_mins[-1], neurons_in_region / channels_in_region)
            region_mins[br].append(neurons_in_region / channels_in_region)
            print("{} neurons/channel ({}, {}) in {}".format(neurons_in_region / channels_in_region, neurons_in_region, channels_in_region, br))
            mice[t['session']['subject']][br] = (float(neurons_in_region / channels_in_region), int(neurons_in_region), int(channels_in_region))
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
    if region_mins[br] != []:
        pos, vals = cumulative_and_position(region_mins[br])
        plt.step(pos, vals, where='post', label=br)
plt.legend(fontsize=22, frameon=False)
plt.ylim(bottom=0)
plt.ylabel("Recordings lost to criterion", size=22)
plt.xlabel("Minimum yield per channel criterion", size=22)
plt.xlim(left=0, right=0.6)
plt.show()
quit()
import json
json.dump(mice, open("yield_metric.json", 'w'))
