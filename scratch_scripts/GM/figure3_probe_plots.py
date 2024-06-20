# %% !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 09:43:50 2020

@author: guido
"""

import pandas as pd
from reproducible_ephys_functions import labs, data_path, exclude_recordings, figure_style
from scratch_scripts.MF.features_2D import plot_2D_features
from one.api import ONE
import numpy as np
from os.path import join, isdir
from os import mkdir
from reproducible_ephys_paths import FIG_PATH
import matplotlib.pyplot as plt
from iblatlas import atlas
brain_atlas = atlas.AllenAtlas(25)
one = ONE()

# Plot settings
#BOUNDARY = None
BOUNDARY = 'DG-TH'
# BOUNDARY = 'VIS-HPF'
MIN_REC_PER_LAB = 1
"""
PLOTS = ['fr', 'psd', 'rms_ap', 'rms_lf', 'fr_alt', 'amp', 'regions_line',
         'distance', 'amp_scatter']
LABELS = ['Firing rate (spks/s)', 'Power spectral density', 'AP band RMS', 'LFP band RMS',
          '', '', 'Histology Regions',
          'Distance from Repeated Site', 'Firing rate (spks/s)']
"""
"""
PLOTS = ['amp_scatter', 'psd', 'rms_ap']
LABELS = ['Firing rate (spks/s)', 'Power spectral density', 'AP band RMS']
"""
PLOTS = ['amp_scatter']
LABELS = ['Firing rate (spks/s)']
"""
PLOTS = ['psd']
LABELS = ['Power spectral density']
"""
NICKNAMES = False
YLIM = [-2500, 2500]
FIG_SIZE = (7, 3.5)

# Load in recordings
data = pd.read_csv(join(data_path(), 'metrics_region.csv'))

# Exclude recordings
data, excluded = exclude_recordings(data, return_excluded=True)

# Reformat dataframe
lab_number_map, institution_map, lab_colors = labs()
data['institution'] = data.lab.map(institution_map)
data = data.drop_duplicates(subset='subject')

# Exclude labs with too few recordings done
data = data.groupby('institution').filter(
    lambda s : s['eid'].unique().shape[0] >= MIN_REC_PER_LAB)
data['lab_number'] = data.lab.map(lab_number_map)
data = data.sort_values(by=['lab_number', 'subject']).reset_index(drop=True)

# Get lab info
rec_per_lab = data.groupby('lab_number').size()
data['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])
data['lab_position'] = np.linspace(0.18, 0.91, data.shape[0])
plot_titles = data.groupby('institution').mean()

##
# %% Plotting
figure_style()
for p, plot_name in enumerate(PLOTS):
    print('Generating %s plot' % plot_name)
    if plot_name == 'amp_scatter':
        f, axs, cbar = plot_2D_features(data['subject'], data['date'], data['probe'], one=one,
                                        brain_atlas=brain_atlas, plot_type=plot_name,
                                        boundary_align=BOUNDARY, show_regions=True, figsize=FIG_SIZE)
    else:

        f, axs, cbar = plot_2D_features(data['subject'], data['date'], data['probe'], one=one,
                                        brain_atlas=brain_atlas, plot_type=plot_name,
                                        freq_range=[20, 80], show_regions=False,
                                        boundary_align=BOUNDARY, figsize=FIG_SIZE)

    for i, subject in enumerate(data['subject']):
        if NICKNAMES:
            axs[i].set_title(subject, rotation=30, ha='left',
                             color=lab_colors[data.loc[i, 'institution']])
        else:
            axs[i].set_title(data.loc[i, 'recording'] + 1,
                             color=lab_colors[data.loc[i, 'institution']])

        if i == 0:
            axs[i].tick_params(axis='y')
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].set_ylabel('Depth relative to DG-Thalamus boundary (\u03BCm)')
        else:
            axs[i].set_axis_off()
        axs[i].set(xticks=[], ylim=YLIM)

    if plot_name[-4:] != 'line':
        # cbar = axs[-1].images[-1].colorbar
        cbar.set_label(LABELS[p], rotation=270, labelpad=-8)
        cbar.ax.tick_params()

    for i, inst in enumerate(plot_titles.index.values):
        if not NICKNAMES:
            plt.figtext((plot_titles.loc[inst, 'lab_position'] - 0.06) * 1.02, 0.94, inst,
                        color=lab_colors[inst], ha='left')
            # plt.figtext(plot_titles.loc[inst, 'lab_position'], 0.94, inst,
            #             color=lab_colors[inst], ha='center')
    if not isdir(join(FIG_PATH, 'probe_plots')):
        mkdir(join(FIG_PATH, 'probe_plots'))

    plt.savefig(join(FIG_PATH, 'probe_plots', 'figure3_probe_%s.png' % plot_name), dpi=300)
    plt.savefig(join(FIG_PATH, 'probe_plots', 'figure3_probe_%s.pdf' % plot_name))


