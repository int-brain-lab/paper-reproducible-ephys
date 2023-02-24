# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:56:43 2021
By: Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import save_data_path

# Settings
SW_CUTOFF = 0.3
DATA_DIR = save_data_path()
FIG_PATH = join(FIG_PATH, 'waveform_clustering')

# Load in waveform data
waveforms_df = pd.read_pickle(join(DATA_DIR, 'waveform_metrics.p'))

# Perform cutoff
waveforms_df.loc[waveforms_df['spike_width'] < SW_CUTOFF, 'type'] = 'NS'
waveforms_df.loc[waveforms_df['spike_width'] >= SW_CUTOFF, 'type'] = 'RS'
waveforms_df.loc[waveforms_df['pt_subtract'] >= 0.025, 'type'] = 'Pos.'

# Print results
perc_reg = (np.sum(waveforms_df["type"] == "RS") / waveforms_df.shape[0]) * 100
perc_fast = (np.sum(waveforms_df["type"] == "NS") / waveforms_df.shape[0]) * 100
print(f'{perc_fast:.2f}% fast spiking')
print(f'{perc_reg:.2f}% regular spiking')


def fix_hist_step_vertical_line_at_end(ax):
    axpolygons = [poly for poly in ax.get_children() if isinstance(poly, mpl.patches.Polygon)]
    for poly in axpolygons:
        poly.set_xy(poly.get_xy()[:-1])


# Save result
waveforms_df.to_csv(join(DATA_DIR, 'neuron_type.csv'))

# %% Plot
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 2), dpi=300)

hst = sns.histplot(data=waveforms_df, x='spike_width', hue='type', hue_order=['NS', 'RS'],
                   legend=None, multiple='stack', bins=70, ax=ax1)
ax1.set(xlim=[0, 1], ylabel='Neuron count', xlabel='Spike width (ms)')

ax2.hist(waveforms_df.loc[waveforms_df['type'] == 'NS', 'firing_rate'], histtype='step',
         density=True, bins=100, cumulative=True, label='Narrow spiking (NS)')
ax2.hist(waveforms_df.loc[waveforms_df['type'] == 'RS', 'firing_rate'], histtype='step',
         density=True, bins=100, cumulative=True, label='Regular spiking (RS)')
ax2.set(xlabel='Firing rate (spks/s)', ylabel='Density', xticks=[0, 25, 50], xlim=[0, 50], yticks=[0, 1])
fix_hist_step_vertical_line_at_end(ax2)


sns.despine(trim=True)
plt.tight_layout()
