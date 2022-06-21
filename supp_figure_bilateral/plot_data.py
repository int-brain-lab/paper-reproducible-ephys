#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from supp_figure_bilateral.fig_bilateral_plot_functions import (panel_probe_lfp, panel_probe_neurons,
                                                                panel_example)
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, filter_recordings, save_figure_path, query
from one.api import ONE


def plot_main_figure(one=None):

    one = one or ONE()

    # Settings
    BOUNDARY = 'DG-TH'
    REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
    METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power',
               'rms_ap', 'spike_amp_mean']
    LABELS = ['Neuron yield', 'Firing rate', 'LFP power',
              'AP band RMS', 'Spike amp.']
    DPI = 150  # if the figure is too big on your screen, lower this number

    # Get amount of probe plots
    n_columns = len(query(min_regions=0, n_trials=0, behavior=False, exclude_critical=True, one=one,
                          as_dataframe=False, bilateral=True))

    # Set up figure
    figure_style()
    fig = plt.figure(figsize=(7, 9), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.05, 0.3],
                                           dim=[1, n_columns + 1], wspace=0.3),
         'panel_B': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.35, 0.6],
                                          dim=[1, n_columns + 1], wspace=0.3),
         'panel_D': fg.place_axes_on_grid(fig, xspan=[0.1, 0.32], yspan=[0.85, 1]),
         'panel_E': fg.place_axes_on_grid(fig, xspan=[0.42, 0.62], yspan=[0.85, 1]),
         'panel_F': fg.place_axes_on_grid(fig, xspan=[0.7, 1], yspan=[0.85, 1])}

    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.35, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'c', 'xpos':0, 'ypos':0.55, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'d', 'xpos':0, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'e', 'xpos':0.32, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'f', 'xpos':0.67, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    fg.add_labels(fig, labels)

    # Call functions to plot panels
    panel_probe_lfp(fig, ax['panel_A'], boundary_align=BOUNDARY)
    panel_probe_neurons(fig, ax['panel_B'], boundary_align=BOUNDARY)



    # Save figure
    save_path = save_figure_path(figure='supp_fig_bilateral')
    print(f'Saving figures to {save_path}')
    plt.savefig(save_path.joinpath('supp_figure_bilateral.png'))
    plt.savefig(save_path.joinpath('supp_figure_bilateral.pdf'))



if __name__ == '__main__':
    one = ONE()
    plot_main_figure(one=one)
