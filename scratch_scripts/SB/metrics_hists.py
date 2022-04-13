# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 11:37:05 2020

@author: Sebastian Bruijns, Noam Roth
"""

from oneibl.one import ONE
import numpy as np
from brainbox.io.one import load_spike_sorting_with_channel
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import iblapps
from iblapps.launch_phy.metrics import *
from reproducible_ephys_paths import FIG_PATH
"""
TODO:
- takes too long to run - only download data once and run all metrics then
- make plotting separate function?
- what's up with eids that throw errors?
- deal with outliers that shift bins too much?
- put everything in function
"""
FIG_PATH = 

regions = ['LP', 'DG-mo', 'Eth', 'PO', 'CA1']
all_labs = ['danlab', 'churchlandlab', 'zadorlab', 'angelakilab', 'wittenlab',
            'mrsicflogellab', 'hoferlab', 'cortexlab', 'mainenlab']
labs = 'all'


def firing_rates(spikes, clusters, mask):
    """Return list of firing rates given mask."""
    return list(clusters.metrics.firing_rate[mask].values)

def amplitude()


metric = firing_rates
metric_name = "Firing rate"

one = ONE()
exclude_eids = ['a66f1593-dafd-4982-9b66-f9554b6c86b5',
'ee40aece-cffd-4edb-a4b6-155f158c666a',
                'ebe090af-5922-4fcd-8fc6-17b8ba7bad6d',
'266a0360-ea0a-4580-8f6a-fe5bad9ed17c',
                '61e11a11-ab65-48fb-ae08-3cb80662e5d6',
'064a7252-8e10-4ad6-b3fd-7a88a2db5463',
                'b01df337-2d31-4bcc-a1fe-7112afd50c50',
'c7248e09-8c0d-40f2-9eb4-700a8973d8c8',
                '28cd1b10-a722-459c-9539-8aac60f3da16',
'46f6bf36-6d5a-42f7-b627-51f3357fbf03',
                '2e43faf9-2aba-488f-b7b7-e5193032b25b',
'dda5fc59-f09a-4256-9fb5-66c67667a466',
                '03cf52f6-fba6-4743-a42e-dd1ac3072343',
'f312aaec-3b6f-44b3-86b4-3a0c119c0438',
                '7f6b86f9-879a-4ea2-8531-294a221af5d0',
'57fd2325-67f4-4d45-9907-29e77d3043d7' ]
#%%
def metric_matrix(regions, metric, metric_name, labs='all', one=None):
    """Plot matrix of metric histogram by region and lab.
    
    Metric takes spikes, clusters and a mask to subset neurons, then
returns a value or a list of values.
    """
    pass


traj = one.alyx.rest('trajectories', 'list', provenance='Planned',
                     x=-2243, y=-2000,  # repeated site coordinate
                     project='ibl_neuropixel_brainwide_01')

labs = all_labs if labs == 'all' else labs
values = np.frompyfunc(list, 0, 1)(np.empty((len(regions), len(labs)),
dtype=object))
lab2num = dict(zip(labs, range(len(labs))))
#%%
for count, t in enumerate(traj):
    print(count)
    eid = t['session']['id']
    if t['session']['lab'] not in labs or eid in exclude_eids:
        continue

    # download data
    probe = t['probe_name']
    try:
        spikes, clusters, channels =pickle.load(open("./data_{}.p".format(eid), "rb"))
    except FileNotFoundError:
        spk, clus, chn = load_spike_sorting_with_channel(eid, one=one)
        spikes, clusters, channels = spk[probe], clus[probe], chn[probe]
        pickle.dump((spikes, clusters, channels),(open("./data_{}.p".format(eid), "wb")))

    cluster_regions = channels.acronym[clusters.channels]

    # subset to brain regions
    for i, br in enumerate(regions):
        mask = cluster_regions == br
        temp = metric(spikes, clusters, mask)
        if type(temp) == list:
            values[i, lab2num[t['session']['lab']]] += temp
        else:
            values[i, lab2num[t['session']['lab']]].append(temp)
#%%
# find max and min of values under histogram
bins = np.histogram_bin_edges(np.sum(values))
max_count = 0
for x in values.flat:
    temp, _ = np.histogram(x, bins)
    max_count = max(max_count, temp.max())

# plot histograms of all the values
for (i, j), x in np.ndenumerate(values):
    plt.subplot(len(regions), len(labs), i * len(labs) + j + 1)
    plt.hist(x, bins)
    plt.ylim(bottom=0, top=max_count+1)
    sns.despine()
    if j == 0:
        plt.ylabel(regions[i])
    else:
        plt.yticks([])
    if i == 0:
        plt.title(labs[j])
    if i == len(regions) - 1:
        plt.xlabel(metric_name)
    else:
        plt.xticks([])
plt.tight_layout()
plt.show()