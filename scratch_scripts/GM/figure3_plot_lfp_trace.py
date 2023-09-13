#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:17 2020

@author: guido
"""

from os import mkdir
import numpy as np
import pandas as pd
from os.path import join, isdir
from pathlib import Path
import glob
import brainbox.io.one as bbone
from ibllib.io import spikeglx
from brainbox.io.one import SpikeSortingLoader
import matplotlib.pyplot as plt
from reproducible_ephys_functions import query, save_figure_path, combine_regions
from one.api import ONE
from iblatlas.atlas import AllenAtlas
ba = AllenAtlas()
one = ONE()

# Settings
PLOT_SEC = 3
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']
REGION_PLOT = np.linspace(0.002, 0, 5)
FIG_PATH = save_figure_path()

# Query repeated site trajectories
traj = query()

# Initialize dataframe
rep_site = pd.DataFrame()

# %% Loop through repeated site recordings
metrics = pd.DataFrame()
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))

    # Get subject data
    pid = traj[i]['probe_insertion']
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    lab = traj[i]['session']['lab']
    nickname = traj[i]['session']['subject']

    # Load in LFP
    lf_paths = one.load_datasets(eid, datasets=['_spikeglx_ephysData_g*_t0.imec*.lf.cbin',
                                                '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
                                                '_spikeglx_ephysData_g*_t0.imec*.lf.ch'],
                                 collections=[f'raw_ephys_data/{probe}']*3, download_only=True)
    raw = spikeglx.Reader(lf_paths[0][0])
    signal = raw.read(nsel=slice(10000, 10000 + 2500 * PLOT_SEC, None),
                      csel=slice(None, None, None))[0]
    signal = signal.T

    # Load in channel data
    sl = SpikeSortingLoader(pid=pid, one=one, atlas=ba)
    _, _, channels = sl.load_spike_sorting()

    # Load in channels
    collections = one.list_collections(eid)
    if f'alf/{probe}/pykilosort' in collections:
        collection = f'alf/{probe}/pykilosort'
    else:
        collection = f'alf/{probe}'
    chn_inds = one.load_dataset(eid, dataset='channels.rawInd.npy', collection=collection)

    # Plot a trace per region
    f, ax1 = plt.subplots(1, 1, figsize=(20, 10), dpi=150)
    for k, region in enumerate(REGIONS):

        # Plot a random channel from each region
        region_chan = chn_inds[[x for x, y
                                in enumerate(combine_regions(channels['acronym']))
                                if y == region]]
        if len(region_chan) > 0:
            ax1.plot(np.linspace(0, PLOT_SEC, signal.shape[1]),
                     signal[np.random.choice(region_chan), :] + REGION_PLOT[k], label=region, lw=2)
    ax1.set(title='%s' % nickname, xlabel='Time (s)')
    ax1.legend()
    if not isdir(join(FIG_PATH, 'lfp-traces')):
        mkdir(join(FIG_PATH, 'lfp-traces'))
    plt.savefig(join(FIG_PATH, 'lfp-traces', 'lfp-traces_%s' % nickname))
    plt.close(f)
