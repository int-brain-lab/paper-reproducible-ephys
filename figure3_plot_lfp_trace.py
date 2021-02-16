#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:22:17 2020

@author: guido
"""

import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
import glob
import brainbox.io.one as bbone
from ibllib.io import spikeglx
import matplotlib.pyplot as plt
from reproducible_ephys_functions import query, data_path, combine_regions
from reproducible_ephys_paths import FIG_PATH
from oneibl.one import ONE
one = ONE()

# Settings
PLOT_SEC = 3
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']
REGION_PLOT = np.linspace(0.002, 0, 5)
DATA_DIR = data_path()

# Query repeated site trajectories
traj = query()

# Initialize dataframe
rep_site = pd.DataFrame()

# %% Loop through repeated site recordings
metrics = pd.DataFrame()
for i in range(len(traj)):
    print('Processing repeated site recording %d of %d' % (i+1, len(traj)))

    # Get subject data
    eid = traj[i]['session']['id']
    probe = traj[i]['probe_name']
    lab = traj[i]['session']['lab']
    nickname = traj[i]['session']['subject']

    try:
        # Load in LFP
        lf_paths = one.load(eid, dataset_types=['ephysData.raw.lf', 'ephysData.raw.meta',
                                                'channels.rawInd', 'ephysData.raw.ch'],
                            download_only=True)
        raw = spikeglx.Reader(lf_paths[int(probe[-1])])
        signal = raw.read(nsel=slice(10000, 10000 + 2500 * PLOT_SEC, None),
                          csel=slice(None, None, None))[0]
        signal = signal.T

        # Load in channel data
        _, _, channels = bbone.load_spike_sorting_with_channel(eid, aligned=True, one=one)
        alf_path = one.path_from_eid(eid).joinpath('alf', probe)
        chn_inds = np.load(Path(join(alf_path, 'channels.rawInd.npy')))

    except Exception as error_message:
        print(error_message)
        continue

    # Plot a trace per region
    f, ax1 = plt.subplots(1, 1, figsize=(20, 10), dpi=150)
    for k, region in enumerate(REGIONS):

        # Plot a random channel from each region
        region_chan = chn_inds[[x for x, y
                                in enumerate(combine_regions(channels[probe]['acronym']))
                                if y == region]]
        if len(region_chan) > 0:
            ax1.plot(np.linspace(0, PLOT_SEC, signal.shape[1]),
                     signal[np.random.choice(region_chan), :] + REGION_PLOT[k], label=region, lw=2)
    ax1.set(title='%s' % nickname, xlabel='Time (s)')
    ax1.legend()
    plt.savefig(join(FIG_PATH, 'lfp-traces_%s' % nickname))
    plt.close(f)
