#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:43:50 2020

@author: guido
"""

from reproducible_ephys_functions import query, labs
from features_2D import plot_2D_features
from oneibl.one import ONE
import numpy as np
from os.path import join
from reproducible_ephys_paths import FIG_PATH
import matplotlib.pyplot as plt
from ibllib.atlas import atlas
brain_atlas = atlas.AllenAtlas(25)
one = ONE()

# Query repeated site trajectories
traj = query(as_dataframe=True)
traj = traj.sort_values(by=['institution', 'eid'], ignore_index=True).reset_index(drop=True)

# Get lab info
lab_number_map, institution_map, lab_colors = labs()
rec_per_lab = traj.groupby('institution').size()
traj['recording'] =  np.concatenate([np.arange(i) for i in rec_per_lab.values])
traj['lab_position'] = np.linspace(0.12, 0.865, traj.shape[0])
plot_titles = traj.groupby('institution').mean()

# Plot
PLOTS = ['fr', 'psd', 'rms_ap', 'rms_lf', 'fr_alt', 'amp', 'fr_line', 'amp_line']

# %% Plotting
for p, plot_name in enumerate(PLOTS):
    print('Generating %s plot' % plot_name)
    f, axs = plot_2D_features(traj['subjects'], traj['dates'], traj['probes'], one=one,
                              brain_atlas=brain_atlas, plot_type=plot_name)
    for i, ax in enumerate(axs):
        ax.set_title(traj.loc[i, 'recording'] + 1, color=lab_colors[traj.loc[i, 'institution']],
                     fontsize=20)
        if i == 0:
            ax.tick_params(axis='y', labelsize=16)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.set_ylabel('Depth relative to Bregma (\u03BCm)', fontsize=20)
        else:
            ax.set_axis_off()
        ax.set(xticks=[], ylim=[-5000, 0])
    f.colorbar(ax.collections[0])

    for i, inst in enumerate(plot_titles.index.values):
        plt.figtext(plot_titles.loc[inst, 'lab_position'], 0.94, inst, color=lab_colors[inst],
                    fontsize=20)
    plt.savefig(join(FIG_PATH, 'figure3_probe_%s' % plot_name))
