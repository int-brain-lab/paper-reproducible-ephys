#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 15:05:50 2021
By: Guido Meijer
"""

import pandas as pd
from reproducible_ephys_functions import labs, data_path, exclude_recordings
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

def probe_plot(ax, plot_name, boundary='VIS-HPF', nicknames=False, ylim=[-4000, 1000], ylabel=''):

    # Load in recordings
    data = pd.read_csv(join(data_path(), 'figure3_brain_regions.csv'))

    # Exclude recordings
    data = exclude_recordings(data)

    # Reformat dataframe
    lab_number_map, institution_map, lab_colors = labs()
    data['institution'] = data.lab.map(institution_map)
    data = data.drop_duplicates(subset='subject')
    data = data.sort_values(by=['institution', 'subject']).reset_index(drop=True)

    # Get lab info
    rec_per_lab = data.groupby('institution').size()
    data['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])
    data['lab_position'] = np.linspace(0.18, 0.9, data.shape[0])
    plot_titles = data.groupby('institution').mean()

    if plot_name == 'amp_scatter':
        f, axs = plot_2D_features(data['subject'], data['date'], data['probe'], one=one,
                                  brain_atlas=brain_atlas, plot_type=plot_name,
                                  boundary_align=boundary, show_regions=True)
    else:
        f, axs = plot_2D_features(data['subject'], data['date'], data['probe'], one=one,
                                  brain_atlas=brain_atlas, plot_type=plot_name,
                                  boundary_align=boundary)

    for i, subject in enumerate(data['subject']):
        if nicknames:
            axs[i].set_title(subject, rotation=30, ha='left',
                                color=lab_colors[data.loc[i, 'institution']], fontsize=16)
        else:
            axs[i].set_title(data.loc[i, 'recording'] + 1,
                            color=lab_colors[data.loc[i, 'institution']], fontsize=20)

        if i == 0:
            axs[i].tick_params(axis='y', labelsize=16)
            axs[i].spines["right"].set_visible(False)
            axs[i].spines["bottom"].set_visible(False)
            axs[i].spines["top"].set_visible(False)
            axs[i].set_ylabel('Depth relative to DG-Thalamus boundary (\u03BCm)',
                                fontsize=20)
        else:
            axs[i].set_axis_off()
        axs[i].set(xticks=[], ylim=ylim)

    if plot_name[-4:] != 'line':
        cbar = axs[-1].images[-1].colorbar
        cbar.set_label(ylabel, rotation=270, labelpad=-8, fontsize=16)
        cbar.ax.tick_params(labelsize=12)

    for i, inst in enumerate(plot_titles.index.values):
        if nicknames:
            plt.figtext(plot_titles.loc[inst, 'lab_position'], 0.978, inst, color=lab_colors[inst],
                            fontsize=20, ha='center')
        elif ((inst == 'CSHL (Z)') | (inst == 'NYU') | (inst == 'Princeton')):
            plt.figtext((plot_titles.loc[inst, 'lab_position'] - 0.06) * 1.05, 0.925, inst,
                        color=lab_colors[inst], fontsize=20, rotation=30, ha='left')
        else:
            plt.figtext((plot_titles.loc[inst, 'lab_position'] - 0.06) * 1.05, 0.925, inst,
                        color=lab_colors[inst], fontsize=20, ha='left')
