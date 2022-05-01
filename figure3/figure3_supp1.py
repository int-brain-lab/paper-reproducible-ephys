#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:44:22 2021
By: Guido Meijer
"""

import figrid as fg
from figure3.figure3_supp1_functions import panel_a, panel_b
import matplotlib.pyplot as plt
from reproducible_ephys_functions import figure_style, save_figure_path, filter_recordings


def plot_figure_supp1():
    # Settings
    # INCL_LABS = ['CCU', 'CSHL (C)', 'NYU', 'SWC', 'Berkeley', 'Princeton']
    # INCL_LABS = ['Princeton']
    INCL_LABS = None
    BOUNDARY = 'DG-TH'
    DPI = 300  # if the figure is too big on your screen, lower this number

    # Get amount of probe plots

    data = filter_recordings()
    n_columns = len(data['subject'].unique())

    # Set up figure
    figure_style()
    fig = plt.figure(figsize=(8, 6), dpi=DPI)  # full width figure is 8 inches
    ax = {'panel_A': fg.place_axes_on_grid(fig, xspan=[0.05, 1], yspan=[0.1, 0.5],
                                           dim=[1, n_columns], wspace=0.3),
          'panel_B': fg.place_axes_on_grid(fig, xspan=[0.05, 1], yspan=[0.6, 1],
                                           dim=[1, n_columns], wspace=0.3)}

    # Call functions to plot panels
    panel_a(fig, ax['panel_A'], incl_labs=INCL_LABS, boundary_align=BOUNDARY)
    panel_b(fig, ax['panel_B'], incl_labs=INCL_LABS, boundary_align=BOUNDARY)

    # Add subplot labels
    labels = [{'label_text': 'a', 'xpos': 0, 'ypos': 0, 'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': 0, 'ypos': 0.6, 'fontsize': 10, 'weight': 'bold', 'ha': 'right', 'va': 'bottom'}]
    fg.add_labels(fig, labels)

    # Save figure
    save_path = save_figure_path(figure='figure3')
    plt.savefig(save_path.joinpath('figure3_supp1.png'))
    plt.savefig(save_path.joinpath('figure3_supp1.pdf'))


if __name__ == '__main__':
    plot_figure_supp1()
