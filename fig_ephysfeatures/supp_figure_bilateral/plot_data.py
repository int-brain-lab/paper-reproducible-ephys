#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from fig_ephysfeatures.supp_figure_bilateral.fig_bilateral_plot_functions import (
    panel_probe_lfp, panel_probe_neurons, panel_boxplot, panel_summary, panel_distribution)
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, filter_recordings, save_figure_path, query, get_row_coord, get_label_pos
from one.api import ONE


def plot_main_figure(one=None):

    one = one or ONE()

    # Settings
    BOUNDARY = 'DG-TH'
    REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
    METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power', 'rms_ap', 'spike_amp_mean']
    LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
    DPI = 150  # if the figure is too big on your screen, lower this number

    # Get amount of probe plots
    # n_columns = len(query(min_regions=0, n_trials=0, behavior=False, exclude_critical=True, one=one,
    #                       as_dataframe=False, bilateral=True))
    n_columns = 14

    # Set up figure
    figure_style()
    width = 7
    height = 7
    fig = plt.figure(figsize=(width, height), dpi=DPI)  # full width figure is 7 inches

    xspans = get_row_coord(width, [1], pad=0.6)
    yspans = get_row_coord(height, [3, 3, 2], hspace=0.8, pad=0.3)
    xspans_row3 = get_row_coord(width, [1, 1, 1, 2], pad=0.6)

    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[0],
                                     dim=[1, n_columns + 1], wspace=0.3),
          'B': fg.place_axes_on_grid(fig, xspan=xspans[0], yspan=yspans[1],
                                           dim=[1, n_columns + 1], wspace=0.3),
          'C': fg.place_axes_on_grid(fig, xspan=xspans_row3[0], yspan=yspans[2]),
          'D': fg.place_axes_on_grid(fig, xspan=xspans_row3[1], yspan=yspans[2]),
          'E': fg.place_axes_on_grid(fig, xspan=xspans_row3[2], yspan=yspans[2]),
          'F': fg.place_axes_on_grid(fig, xspan=xspans_row3[3], yspan=yspans[2]),
          }

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspans[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspans_row3[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspans_row3[1][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspans_row3[2][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width, xspans_row3[3][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10, 'weight': 'bold',
               'ha': 'right', 'va': 'bottom'}
              ]

    fg.add_labels(fig, labels)

    # Call functions to plot panels
    panel_probe_lfp(fig, ax['A'], boundary_align=BOUNDARY)
    panel_probe_neurons(fig, ax['B'], boundary_align=BOUNDARY)
    panel_distribution(ax['C'], example_region='CA1', example_metric='lfp_power',
                       ylabel='LFP power diff. in CA1 (dB)')#, yticks=[0, 30, 60])
    panel_distribution(ax['D'], example_region='CA1', example_metric='median_firing_rate',
                       ylabel='Firing rate diff. in CA1 (spikes/s)')#, yticks=[0, 5, 10])
    panel_distribution(ax['E'], example_region='CA1', example_metric='spike_amp_median',
                       ylabel='Spike amp. diff. in CA1 (\u03bcV)')#, yticks=[0, 0.0003])
    #panel_distribution(ax['panel_E'], example_region='PPC', example_metric='rms_ap',
    #                   ylabel='Spike amp. diff. in PPC (uV)')
    #panel_boxplot(ax['panel_C'], example_region='DG', example_metric='lfp_power', yticks=[0, 5, 10, 15])
    #panel_boxplot(ax['panel_D'], example_region='PPC', example_metric='median_firing_rate',
    #              ylabel='Firing rate diff. in PPC (spks/s)', yticks=[0, 5])
    #panel_boxplot(ax['panel_E'], example_region='PPC', example_metric='spike_amp_median',
    #              ylabel='Spike amp. diff. in PPC (uV)', yticks=[0, 0.0003])
    panel_summary(ax['F'])

    # Save figure
    save_path = save_figure_path(figure='fig_ephysfeatures')
    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + 0.3)/height, left=(adjust)/width, right=1-(adjust + 0.3)/width)
    #print(f'Saving figures to {save_path}')
    plt.savefig(save_path.joinpath('figure3_supp3_bilateral.svg'))
    plt.savefig(save_path.joinpath('figure3_supp3_bilateral.pdf'))



if __name__ == '__main__':
    one = ONE()
    plot_main_figure(one=one)
