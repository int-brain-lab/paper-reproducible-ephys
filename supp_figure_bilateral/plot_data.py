#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from supp_figure_bilateral.fig_bilateral_plot_functions import (panel_probe_lfp, panel_probe_neurons,
                                                                panel_boxplot, panel_summary,
                                                                panel_distribution)
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, filter_recordings, save_figure_path, query
from one.api import ONE


def plot_main_figure(one=None):

    one = one or ONE()

    # Settings
    BOUNDARY = 'DG-TH'
    REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
    METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean']
    LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
    DPI = 300  # if the figure is too big on your screen, lower this number

    # Get amount of probe plots
    n_columns = len(query(min_regions=0, n_trials=0, behavior=False, exclude_critical=True, one=one,
                          as_dataframe=False, bilateral=True)) - 6

    # Set up figure
    figure_style()
    fig = plt.figure(figsize=(7, 7), dpi=DPI)  # full width figure is 7 inches
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.05, 0.4],
                                           dim=[1, n_columns + 1], wspace=0.3),
         'panel_B': fg.place_axes_on_grid(fig, xspan=[0.1, 1], yspan=[0.45, 0.75],
                                          dim=[1, n_columns + 1], wspace=0.3),
         'panel_C': fg.place_axes_on_grid(fig, xspan=[0.1, 0.2], yspan=[0.8, 1]),
         'panel_D': fg.place_axes_on_grid(fig, xspan=[0.3, 0.4], yspan=[0.8, 1]),
         'panel_E': fg.place_axes_on_grid(fig, xspan=[0.53, 0.63], yspan=[0.8, 1]),
         'panel_F': fg.place_axes_on_grid(fig, xspan=[0.7, 1], yspan=[0.8, 1])}

    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.45, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'c', 'xpos':0, 'ypos':0.8, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'d', 'xpos':0.2, 'ypos':0.8, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'e', 'xpos':0.42, 'ypos':0.8, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'f', 'xpos':0.65, 'ypos':0.8, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    fg.add_labels(fig, labels)

    # Call functions to plot panels
    panel_probe_lfp(fig, ax['panel_A'], boundary_align=BOUNDARY)
    panel_probe_neurons(fig, ax['panel_B'], boundary_align=BOUNDARY)
    panel_distribution(ax['panel_C'], example_region='CA1', example_metric='lfp_power',
                       ylabel='LFP power diff. in CA1 (db)', yticks=[0, 30, 60])
    panel_distribution(ax['panel_D'], example_region='PPC', example_metric='median_firing_rate',
                       ylabel='Firing rate diff. in PPC (spks/s)', yticks=[0, 5, 10])
    panel_distribution(ax['panel_E'], example_region='PPC', example_metric='spike_amp_median',
                       ylabel='Spike amp. diff. in PPC (uV)', yticks=[0, 0.0003])
    #panel_distribution(ax['panel_E'], example_region='PPC', example_metric='rms_ap',
    #                   ylabel='Spike amp. diff. in PPC (uV)')
    #panel_boxplot(ax['panel_C'], example_region='DG', example_metric='lfp_power', yticks=[0, 5, 10, 15])
    #panel_boxplot(ax['panel_D'], example_region='PPC', example_metric='median_firing_rate',
    #              ylabel='Firing rate diff. in PPC (spks/s)', yticks=[0, 5])
    #panel_boxplot(ax['panel_E'], example_region='PPC', example_metric='spike_amp_median',
    #              ylabel='Spike amp. diff. in PPC (uV)', yticks=[0, 0.0003])
    panel_summary(ax['panel_F'])



    # Save figure
    save_path = save_figure_path(figure='supp_fig_bilateral')
    #print(f'Saving figures to {save_path}')
    plt.savefig(save_path.joinpath('supp_figure_bilateral.png'))
    plt.savefig(save_path.joinpath('supp_figure_bilateral.pdf'))



if __name__ == '__main__':
    one = ONE()
    plot_main_figure(one=one)
