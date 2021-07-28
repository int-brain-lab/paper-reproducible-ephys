#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from figure3_functions import probe_plots_data, panel_a
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style
from one.api import ONE
one = ONE()

# Settings
MIN_REC_PER_LAB = 4
BOUNDARY = 'DG-TH'

# Get amount of probe plots
data, _ = probe_plots_data(MIN_REC_PER_LAB)
n_columns = len(data['subject'].unique())

# Set up figure
figure_style()
fig = plt.figure(figsize=(7, 7), dpi=400)  # full width figure is 7 inches
ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.075, 0.7], yspan=[0.05, 0.45],
                                       dim=[1, n_columns], wspace=0.3),
      'panel_B': fg.place_axes_on_grid(fig, xspan=[0.075, 0.7], yspan=[0.57, 1],
                                       dim=[1, n_columns], wspace=0.3),
      'panel_C': fg.place_axes_on_grid(fig, xspan=[0.8, 1], yspan=[0.05, 0.45]),
      'panel_D': fg.place_axes_on_grid(fig, xspan=[0.8, 1], yspan=[0.57, 1])}

# Call functions to plot panels
panel_a(ax['panel_A'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY, one=one)

# Add subplot labels
labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'b', 'xpos':0, 'ypos':0.5, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'c', 'xpos':0.72, 'ypos':0, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
          {'label_text':'d', 'xpos':0.72, 'ypos':0.5, 'fontsize':10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]
fg.add_labels(fig, labels)