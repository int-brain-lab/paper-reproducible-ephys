#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from os.path import join
from figure3_functions import (plots_data, panel_probe_lfp, panel_probe_neurons, panel_example,
                               panel_permutation, panel_sankey)
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style
from reproducible_ephys_paths import FIG_PATH
from one.api import ONE
one = ONE()

# Settings
MIN_REC_PER_LAB = 4
MIN_REC_PER_REGION = 3
BOUNDARY = 'DG-TH'
EXAMPLE_REGION = 'LP'
EXAMPLE_METRIC = 'lfp_ratio'
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_ratio',
           'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP ratio',
          'AP band RMS', 'Spike amp.']
N_PERMUT = 100000  # Amount of shuffles for permutation testing
DPI = 150  # if the figure is too big on your screen, lower this number

# Get amount of probe plots
data, _ = plots_data(MIN_REC_PER_LAB)
n_columns = len(data['subject'].unique())

# Set up figure
figure_style()
fig = plt.figure(figsize=(7, 9), dpi=DPI)  # full width figure is 7 inches
# ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0, 0.18]),
#       'panel_B': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.25, 0.5],
#                                        dim=[1, n_columns + 1], wspace=0.3),
#       'panel_C': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.55, 0.8],
#                                        dim=[1, n_columns + 1], wspace=0.3),
#       'panel_D': fg.place_axes_on_grid(fig, xspan=[0.2, 0.45], yspan=[0.85, 1]),
#       'panel_E': fg.place_axes_on_grid(fig, xspan=[0.55, 1], yspan=[0.85, 1])}
#
#TODO remove GC print
ax = {'panel_C': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.55, 0.8],
                                       dim=[1, n_columns + 1], wspace=0.3)}

# Add subplot labels
labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'b', 'xpos':0, 'ypos':0.2, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'c', 'xpos':0, 'ypos':0.55, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'d', 'xpos':0, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'},
          {'label_text':'e', 'xpos':0.5, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
           'ha': 'right', 'va': 'bottom'}]
fg.add_labels(fig, labels)

# Call functions to plot panels
# TODO GC uncomment
# panel_sankey(fig, ax['panel_A'])
# panel_probe_lfp(fig, ax['panel_B'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY, one=one)
panel_probe_neurons(fig, ax['panel_C'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY, one=one)
# panel_example(ax['panel_D'], n_rec_per_lab=MIN_REC_PER_LAB, example_region=EXAMPLE_REGION,
#               example_metric=EXAMPLE_METRIC, ylim=[0, 4])
# p_values = panel_permutation(ax['panel_E'], METRICS, REGIONS, LABELS, n_permut=N_PERMUT,
#                              n_rec_per_lab=MIN_REC_PER_LAB)
#
# # Save figure
# plt.savefig(join(FIG_PATH, 'figure3.png'))
# plt.savefig(join(FIG_PATH, 'figure3.pdf'))
