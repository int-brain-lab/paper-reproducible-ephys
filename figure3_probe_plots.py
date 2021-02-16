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
from os.path import join, isdir
from os import mkdir
from reproducible_ephys_paths import FIG_PATH
import matplotlib.pyplot as plt
from ibllib.atlas import atlas
brain_atlas = atlas.AllenAtlas(25)
one = ONE()

# Plot settings
PLOTS = ['fr', 'psd', 'rms_ap', 'rms_lf', 'fr_alt', 'amp', 'fr_line', 'amp_line']
LABELS = ['Firing rate (spks/s)', 'Power spectral density', 'AP band RMS', 'LFP band RMS',
          'Firing rate (spks/s)', 'Spike amplitude', '', '']
#PLOTS = ['rms_ap']
#LABELS = ['AP band RMS']
NICKNAMES = True

# Query repeated site trajectories
traj = query(as_dataframe=True)
traj = traj.sort_values(by=['subjects'], ignore_index=True).reset_index(drop=True)

# Get lab info
lab_number_map, institution_map, lab_colors = labs()
rec_per_lab = traj.groupby('institution').size()
traj['recording'] =  np.concatenate([np.arange(i) for i in rec_per_lab.values])
traj['lab_position'] = np.linspace(0.12, 0.865, traj.shape[0])
plot_titles = traj.groupby('institution').mean()


# %% Plotting
for p, plot_name in enumerate(PLOTS):
    print('Generating %s plot' % plot_name)
    f, axs = plot_2D_features(traj['subjects'], traj['dates'], traj['probes'], one=one,
                              brain_atlas=brain_atlas, plot_type=plot_name,
                              boundary_align='VIS-HPF')
    for i, subject in enumerate(traj['subjects']):
        if NICKNAMES:
            axs[i].set_title(subject, color='k', rotation=30, ha='left', fontsize=16)
        else:
            axs[i].set_title(traj.loc[i, 'recording'] + 1,
                         color=lab_colors[traj.loc[i, 'institution']], fontsize=20)

        if i == 0:
            axs[i].tick_params(axis='y', labelsize=16)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].set_ylabel('Depth relative to cortex-hippocampus boundary (\u03BCm)',
                              fontsize=20)
        else:
            axs[i].set_axis_off()
        axs[i].set(xticks=[], ylim=[-3000, 1000])

    if plot_name[-4:] != 'line':
        cbar = axs[-1].images[-1].colorbar
        cbar.set_label(LABELS[p], rotation=270, labelpad=-8, fontsize=16)
        cbar.ax.tick_params(labelsize=12)

    if not NICKNAMES:
        for i, inst in enumerate(plot_titles.index.values):
            plt.figtext(plot_titles.loc[inst, 'lab_position'], 0.94, inst, color=lab_colors[inst],
                        fontsize=20)
    if not isdir(join(FIG_PATH, 'probe_plots')):
        mkdir(join(FIG_PATH, 'probe_plots'))

    plt.savefig(join(FIG_PATH, 'probe_plots', 'figure3_probe_%s' % plot_name))
