#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from fig_ephysfeatures.ephysfeatures_supp_plot_functions import panel_probe_lfp, panel_probe_neurons, panel_probe_lfp_unaligned
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, save_figure_path, filter_recordings, get_row_coord, get_label_pos


def plot_figure_supp1():
    # Settings
    BOUNDARY = 'DG-TH'
    DPI = 300  # if the figure is too big on your screen, lower this number

    # Get amount of probe plots

    data = filter_recordings()
    n_columns = len(data['subject'].unique())

    # Set up figure
    figure_style()
    width = 7
    height = 8
    fig = plt.figure(figsize=(width, height), dpi=DPI)  # full width figure is 8 inches

    xspan = get_row_coord(width, [1])
    yspan = get_row_coord(height, [1, 1, 1], hspace=0.8, pad=0.3)

    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspan[0], yspan=yspan[0],
                                     dim=[1, n_columns], wspace=0.3),
          'B': fg.place_axes_on_grid(fig, xspan=xspan[0], yspan=yspan[1],
                                     dim=[1, n_columns], wspace=0.3),
          'C': fg.place_axes_on_grid(fig, xspan=xspan[0], yspan=yspan[2],
                                     dim=[1, n_columns], wspace=0.3)
          }

    # Call functions to plot panels
    panel_probe_lfp_unaligned(fig, ax['C'], boundary_align=BOUNDARY)
    panel_probe_lfp(fig, ax['A'], boundary_align=BOUNDARY)
    panel_probe_neurons(fig, ax['B'], boundary_align=BOUNDARY)

    # Add subplot labels
    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspan[0][0]), 'ypos': get_label_pos(height, yspan[0][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan[0][0]), 'ypos': get_label_pos(height, yspan[1][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspan[0][0]),
               'ypos': get_label_pos(height, yspan[2][0], pad=0.3),
               'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]
    fg.add_labels(fig, labels)

    # Save figure
    save_path = save_figure_path(figure='fig_ephysfeatures')
    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=adjust/height, left=(adjust + 0.2)/width, right=1-(adjust + 0.2)/width)
    plt.savefig(save_path.joinpath('figure3_supp1.png'))
    plt.savefig(save_path.joinpath('figure3_supp1.pdf'))


if __name__ == '__main__':
    plot_figure_supp1()
