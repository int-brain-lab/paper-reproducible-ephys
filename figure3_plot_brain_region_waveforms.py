#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:48:00 2020

@author: guido
"""

from os.path import join
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from reproducible_ephys_functions import data_path, labs
from reproducible_ephys_paths import FIG_PATH

# Settings
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']

# Load in data
waveforms_df = pd.read_pickle(join(data_path(), 'figure3_brain_region_waveforms.p'))

# Get lab info
lab_number_map, institution_map, lab_colors = labs()

# %% Plot
sns.set(style='ticks', context='paper', font_scale=1.8)
f, axs = plt.subplots(len(REGIONS), waveforms_df['lab'].unique().shape[0],
                      figsize=(20, 15), dpi=150, sharey=True, sharex=True)
for i, region in enumerate(REGIONS):
    for j, lab in enumerate(waveforms_df['lab'].unique()):
        for k in waveforms_df[(waveforms_df['lab'] == lab)
                              & (waveforms_df['region'] == region)].index:
            if waveforms_df.loc[k, 'waveforms'].shape[1] != 0:
                axs[i, j].plot(np.linspace(0, (82 / 30000) * 1000, 82),
                               waveforms_df.loc[k, 'waveforms'], color=[0.6, 0.6, 0.6])
            axs[i, j].set_frame_on(False)
            if i == 0:
                axs[i, j].set(title='%s' % lab)
                axs[i, j].axes.get_xaxis().set_visible(False)
            elif i == len(REGIONS)-1:
                axs[i, j].set(xlabel='ms')
            else:
                axs[i, j].axes.get_xaxis().set_visible(False)

            if j == 0:
                axs[i, j].set_ylabel(region, rotation=0, size='large')
            else:
                axs[i, j].axes.get_yaxis().set_visible(False)

sns.despine()
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_brain_region_waveforms'))




