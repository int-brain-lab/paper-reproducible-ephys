#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from os.path import join
from figure3_supp1_functions import plots_data, panel_a, panel_b
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style
from reproducible_ephys_paths import FIG_PATH
from one.api import ONE
one = ONE()

# Settings
#INCL_LABS = ['CCU', 'CSHL (C)', 'NYU', 'SWC', 'Berkeley', 'Princeton']
#INCL_LABS = ['Princeton']
INCL_LABS = None
BOUNDARY = 'DG-TH'
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power_high', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
N_PERMUT = 1000  # Amount of shuffles for permutation testing
#N_PERMUT = 10  # Amount of shuffles for permutation testing
DPI = 300  # if the figure is too big on your screen, lower this number

# Get amount of probe plots
data, _ = plots_data()
n_columns = len(data['subject'].unique())

# Set up figure
figure_style()
fig = plt.figure(figsize=(8, 6), dpi=DPI)  # full width figure is 8 inches
ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.05, 1], yspan=[0.1, 0.5],
                                       dim=[1, n_columns], wspace=0.3),
      'panel_B': fg.place_axes_on_grid(fig, xspan=[0.05, 1], yspan=[0.6, 1],
                                       dim=[1, n_columns], wspace=0.3)}

# Call functions to plot panels
panel_a(fig, ax['panel_A'], incl_labs=INCL_LABS, boundary_align=BOUNDARY, one=one)
panel_b(fig, ax['panel_B'], incl_labs=INCL_LABS, boundary_align=BOUNDARY, one=one)
#panel_c(ax['panel_C'], METRICS, REGIONS, LABELS, n_permut=N_PERMUT, incl_labs=INCL_LABS)

# Add subplot labels
labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'b', 'xpos':0, 'ypos':0.6, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]
fg.add_labels(fig, labels)

# Save figure
plt.savefig(join(FIG_PATH, 'figure3_supp1.png'))
plt.savefig(join(FIG_PATH, 'figure3_supp1.pdf'))
