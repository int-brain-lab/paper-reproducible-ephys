#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from os.path import join
from figure3_functions import plots_data, panel_a, panel_b, panel_c, panel_d
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style
from reproducible_ephys_paths import FIG_PATH
from one.api import ONE
one = ONE()

# Settings
MIN_REC_PER_LAB = 4
BOUNDARY = 'DG-TH'
EXAMPLE_REGION = 'CA1'
EXAMPLE_METRIC = 'lfp_power_high'
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power_high', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
N_PERMUT = 1000000  # Amount of shuffles for permutation testing
DPI = 400  # if the figure is too big on your screen, lower this number

# Get amount of probe plots
data, _ = plots_data(MIN_REC_PER_LAB)
n_columns = len(data['subject'].unique())

# Set up figure
figure_style()
fig = plt.figure(figsize=(7, 7), dpi=DPI)  # full width figure is 7 inches
ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.075, 0.6], yspan=[0.05, 0.45],
                                       dim=[1, n_columns], wspace=0.3),
      'panel_B': fg.place_axes_on_grid(fig, xspan=[0.075, 0.6], yspan=[0.57, 1],
                                       dim=[1, n_columns], wspace=0.3),
      'panel_C': fg.place_axes_on_grid(fig, xspan=[0.8, 1], yspan=[0.05, 0.38]),
      'panel_D': fg.place_axes_on_grid(fig, xspan=[0.8, 1], yspan=[0.5, 1])}

# Call functions to plot panels
panel_a(fig, ax['panel_A'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY, one=one)
panel_b(fig, ax['panel_B'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY, one=one)
panel_c(ax['panel_C'], n_rec_per_lab=MIN_REC_PER_LAB, example_region=EXAMPLE_REGION,
        example_metric=EXAMPLE_METRIC)
panel_d(ax['panel_D'], METRICS, REGIONS, LABELS, n_permut=N_PERMUT, n_rec_per_lab=MIN_REC_PER_LAB)

# Add subplot labels
labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'b', 'xpos':0, 'ypos':0.55, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'c', 'xpos':0.72, 'ypos':0, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'d', 'xpos':0.72, 'ypos':0.55, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]
fg.add_labels(fig, labels)

# Save figure
plt.savefig(join(FIG_PATH, 'figure3.png'))
plt.savefig(join(FIG_PATH, 'figure3.pdf'))
