#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from figure3.figure3_plot_functions import (panel_probe_lfp, panel_probe_neurons, panel_example,
                                            panel_permutation, panel_sankey)
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, filter_recordings, save_figure_path
from one.api import ONE


def plot_main_figure(one=None):

    one = one or ONE()

    # Settings
    MIN_REC_PER_LAB = 0  # for plotting of probe plots
    MIN_REC_PER_REGION = 3  # for permutation testing
    BOUNDARY = 'DG-TH'
    REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
    METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power',
               'rms_ap', 'spike_amp_mean']
    LABELS = ['Neuron yield', 'Firing rate', 'LFP power',
              'AP band RMS', 'Spike amp.']
    N_PERMUT = 10000  # Amount of shuffles for permutation testing
    DPI = 150  # if the figure is too big on your screen, lower this number

    # Get amount of probe plots
    data = filter_recordings(min_rec_lab=MIN_REC_PER_LAB)
    data = data[data['lab_include'] == 1]
    n_columns = len(data['subject'].unique())

    # Set up figure
    figure_style()
    fig = plt.figure(figsize=(7, 9), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0, 0.18]),
         'panel_B': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.25, 0.5],
                                          dim=[1, n_columns + 1], wspace=0.3),
         'panel_C': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.55, 0.8],
                                          dim=[1, n_columns + 1], wspace=0.3),
         'panel_D': fg.place_axes_on_grid(fig, xspan=[0.1, 0.35], yspan=[0.85, 1]),
         'panel_E': fg.place_axes_on_grid(fig, xspan=[0.42, 0.67], yspan=[0.85, 1]),
         'panel_F': fg.place_axes_on_grid(fig, xspan=[0.7, 1], yspan=[0.85, 1])}
   
    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.2, 'fontsize':10, 'weight': 'bold',
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
    panel_sankey(fig, ax['panel_A'], one)
    panel_probe_lfp(fig, ax['panel_B'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY)
    panel_probe_neurons(fig, ax['panel_C'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY)
    panel_example(ax['panel_D'], n_rec_per_lab=MIN_REC_PER_LAB, example_region='CA1',
                  example_metric='spike_amp_mean', ylim=[75, 225], ylabel='Spike amplitude in CA1 (uV)',
                  yticks=[75, 125, 175, 225])
    panel_example(ax['panel_E'], n_rec_per_lab=MIN_REC_PER_LAB, example_region='LP',
                  example_metric='median_firing_rate', ylim=[0, 10], ylabel='Firing rate in LP (spks/s)',
                  yticks=[0, 2, 4, 6, 8, 10])
    p_values = panel_permutation(ax['panel_F'], METRICS, REGIONS, LABELS, n_permut=N_PERMUT,
                                 n_rec_per_lab=MIN_REC_PER_LAB, n_rec_per_region=MIN_REC_PER_REGION)

    # Save figure
    save_path = save_figure_path(figure='figure3')
    print(f'Saving figures to {save_path}')
    plt.savefig(save_path.joinpath('figure3.png'))
    plt.savefig(save_path.joinpath('figure3.pdf'))

    return p_values


if __name__ == '__main__':
    one = ONE()
    plot_main_figure(one=one)
