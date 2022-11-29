#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from figure3.figure3_plot_functions import (panel_probe_lfp, panel_probe_neurons, panel_example,
                                            panel_permutation, panel_sankey, panel_decoding)
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, filter_recordings, save_figure_path
from one.api import ONE
import numpy as np


def plot_main_figure(one=None):

    one = one or ONE()

    # Settings
    MIN_REC_PER_LAB = 3  # for plotting of probe plots
    MIN_REC_PER_REGION = 3  # for permutation testing
    BOUNDARY = 'DG-TH'
    REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
    METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_theta_power',
               'rms_ap', 'spike_amp_mean']
    LABELS = ['Neuron yield', 'Firing rate', 'LFP power',
              'AP band RMS', 'Spike amp.']
    #N_PERMUT = 50000  # Amount of shuffles for permutation testing
    N_PERMUT = 500  # Amount of shuffles for permutation testing
    DPI = 200  # if the figure is too big on your screen, lower this number
    np.random.seed(42)  # fix the random seed for reproducible permutatation results

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
         'panel_D': fg.place_axes_on_grid(fig, xspan=[0.1, 0.25], yspan=[0.85, 1]),
         'panel_E': fg.place_axes_on_grid(fig, xspan=[0.32, 0.55], yspan=[0.85, 1]),
         'panel_F': fg.place_axes_on_grid(fig, xspan=[0.74, 1], yspan=[0.85, 1])}

    # Add subplot labels
    labels = [{'label_text':'a', 'xpos':0, 'ypos':0, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'b', 'xpos':0, 'ypos':0.2, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'c', 'xpos':0, 'ypos':0.55, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'d', 'xpos':0, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'e', 'xpos':0.27, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text':'f', 'xpos':0.64, 'ypos':0.85, 'fontsize':10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}]
    fg.add_labels(fig, labels)

    # Call functions to plot panels
    panel_sankey(fig, ax['panel_A'], one)
    panel_probe_lfp(fig, ax['panel_B'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY)
    panel_probe_neurons(fig, ax['panel_C'], n_rec_per_lab=MIN_REC_PER_LAB, boundary_align=BOUNDARY)
    """
    panel_example(ax['panel_D'], n_rec_per_lab=MIN_REC_PER_LAB, example_region='DG',
                  example_metric='lfp_power', ylim=[-170, -140], ylabel='LFP power in DG (db)',
                  despine=True, yticks=[-170, -160, -150, -140])
    panel_example(ax['panel_D'], n_rec_per_lab=MIN_REC_PER_LAB, example_region='LP',
                  example_metric='lfp_power', ylim=[-190, -160], ylabel='LFP power in LP (db)',
                  despine=True, yticks=[-190, -180, -170, -160])
    panel_example(ax['panel_D'], n_rec_per_lab=MIN_REC_PER_LAB, example_region='PPC',
                  example_metric='median_firing_rate', ylim=[0, 12], ylabel='Firing rate in PPC (spks/s)',
                  despine=True, yticks=[0, 6, 12])
    """
    panel_example(ax['panel_D'], n_rec_per_lab=MIN_REC_PER_LAB, example_region='CA1',
                  example_metric='lfp_theta_power', ylim=[-160, -100], ylabel='LFP power in CA1 (db)',
                  despine=True, yticks=[-160, -100])

    p_permut = panel_permutation(ax['panel_E'], METRICS, REGIONS, LABELS, n_permut=N_PERMUT,
                                 n_rec_per_lab=MIN_REC_PER_LAB, n_rec_per_region=MIN_REC_PER_REGION)
    p_decoding = panel_decoding(ax['panel_F'], qc=True)

    # Save figure
    save_path = save_figure_path(figure='figure3')
    print(f'Saving figures to {save_path}')
    plt.savefig(save_path.joinpath('figure3.png'))
    plt.savefig(save_path.joinpath('figure3.pdf'))

    return p_decoding, p_permut


if __name__ == '__main__':
    one = ONE()
    p_decoding, p_permut = plot_main_figure(one=one)
