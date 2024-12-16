#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure containing all subject histology plots
coronal/sagittal through the repeated site histology data, and plots the
probe channels onto this histology data.

@author: sjwest
"""


import svgutils.compose as sc  # compose figure with svgutils
import svgutils.transform as sg
from fig_hist.figure_hist_plot_probe_angle_MLxAP import plot_probe_angle_histology_panel, plot_probe_angle_histology_all_lab, plot_probe_angle_histology
from fig_hist.figure_hist_plot_probe_surf_coord_MLxAP import (plot_probe_surf_coord_histology_panel,
                                                             plot_probe_surf_coord_micro_panel, plot_probe_distance_all_lab, plot_probe_surf_coord)
from fig_hist.figure_hist_plot_probe_trajs_ccf import plot_trajs
from fig_hist.figure_hist_plot_channels_all_subjs import plot_channels_n3, plot_channels_n2, plot_channels_n1
from reproducible_ephys_functions import save_figure_path, repo_path, figure_style, get_label_pos, get_row_coord, remove_frame

import figrid as fg
import matplotlib.pyplot as plt

def plot_hist_figure_new(perform_permutation_test=False):
    
    figure_style()
    width = 7
    height = 8
    fig = plt.figure(figsize=(width, height))

    xspan_row1 = get_row_coord(width, [3, 2, 2], pad=0.6)
    xspans_row = get_row_coord(width, [1, 1, 1], hspace=0.8, pad=0.6)
    yspans = get_row_coord(height, [2, 2, 3], hspace=[0.6, 1])


    yspan_inset = get_row_coord(height, [1, 5], pad=0, hspace=0.1, span=yspans[2])

    ax = {'A': fg.place_axes_on_grid(fig, xspan=xspan_row1[0], yspan=yspans[0]),
          'B': fg.place_axes_on_grid(fig, xspan=xspan_row1[1], yspan=yspans[0]),
          'C': fg.place_axes_on_grid(fig, xspan=xspan_row1[2], yspan=yspans[0], dim=[1, 2], wspace=0.05),
          'D': fg.place_axes_on_grid(fig, xspan=xspans_row[0], yspan=yspans[1]),
          'E': fg.place_axes_on_grid(fig, xspan=xspans_row[1], yspan=yspans[1]),
          'F': fg.place_axes_on_grid(fig, xspan=xspans_row[2], yspan=yspans[1]),
          'G_1': fg.place_axes_on_grid(fig, xspan=xspans_row[0], yspan=yspan_inset[0]),
          'G_2': fg.place_axes_on_grid(fig, xspan=xspans_row[0], yspan=yspan_inset[1]),
          'H_1': fg.place_axes_on_grid(fig, xspan=xspans_row[1], yspan=yspan_inset[0]),
          'H_2': fg.place_axes_on_grid(fig, xspan=xspans_row[1], yspan=yspan_inset[1]),
          'I_1': fg.place_axes_on_grid(fig, xspan=xspans_row[2], yspan=yspan_inset[0]),
          'I_2': fg.place_axes_on_grid(fig, xspan=xspans_row[2], yspan=yspan_inset[1]),
          }

    ax['A'].set_axis_off()
    ax['B'].set_axis_off()

    plot_trajs(ax1=ax['C'][0], ax2=ax['C'][1], save=False)

    min_rec_per_lab = 4

    plot_probe_surf_coord(traj='micro', min_rec_per_lab=min_rec_per_lab, ax1=ax['D'], save=False)
    plot_probe_surf_coord(traj='hist', min_rec_per_lab=min_rec_per_lab, ax1=ax['E'], save=False)
    plot_probe_angle_histology(min_rec_per_lab=min_rec_per_lab, ax1=ax['F'], save=False)

    plot_probe_distance_all_lab(traj='micro', min_rec_per_lab=min_rec_per_lab,
                                perform_permutation_test=perform_permutation_test, axs=[ax['G_1'], ax['G_2']], save=False)

    plot_probe_distance_all_lab(traj='hist', min_rec_per_lab=min_rec_per_lab,
                                perform_permutation_test=perform_permutation_test, axs=[ax['H_1'], ax['H_2']], save=False)

    plot_probe_angle_histology_all_lab(min_rec_per_lab=min_rec_per_lab,
                                       perform_permutation_test=perform_permutation_test, axs=[ax['I_1'], ax['I_2']],
                                       save=False)

    labels = [{'label_text': 'a', 'xpos': get_label_pos(width, xspan_row1[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.1),
               'fontsize': 10, 'weight': 'heavy', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'b', 'xpos': get_label_pos(width, xspan_row1[1][0], pad=0.2),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.1), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'c', 'xpos': get_label_pos(width, xspan_row1[2][0], pad=0.2),
               'ypos': get_label_pos(height, yspans[0][0], pad=0.1), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'd', 'xpos': get_label_pos(width, xspans_row[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'e', 'xpos': get_label_pos(width, xspans_row[1][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'f', 'xpos': get_label_pos(width, xspans_row[2][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[1][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'g', 'xpos': get_label_pos(width, xspans_row[0][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'h', 'xpos': get_label_pos(width, xspans_row[1][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'},
              {'label_text': 'i', 'xpos': get_label_pos(width, xspans_row[2][0], pad=0.6),
               'ypos': get_label_pos(height, yspans[2][0], pad=0.3), 'fontsize': 10,
               'weight': 'bold', 'ha': 'right', 'va': 'bottom'}
              ]

    fig_path = save_figure_path(figure='fig_hist')
    print(f'Saving figures to {fig_path}')

    fg.add_labels(fig, labels)

    adjust = 0.3
    fig.subplots_adjust(top=1-adjust/height, bottom=(adjust + 0.2)/height, left=(adjust)/width, right=1-adjust/width)
    plt.savefig(fig_path.joinpath('fig_hist.svg'))
    plt.savefig(fig_path.joinpath('fig_hist.png'))
    plt.savefig(fig_path.joinpath('fig_hist.pdf'))
    plt.close()


def plot_hist_figure(raw_histology=False, perform_permutation_test=False):

    if raw_histology:
        plot_channels_n1()
    else:
        print(f'Raw histology not yet available, will use pregenerated figure '
              f'{repo_path().joinpath("fig_hist", "fig_hist_schematics", "B_channels_subj1_hist_coronal.svg")}')

    plot_trajs()

    plot_probe_surf_coord_micro_panel(min_rec_per_lab=4,
                                      perform_permutation_test=perform_permutation_test)

    plot_probe_surf_coord_histology_panel(min_rec_per_lab=4,
                                      perform_permutation_test=perform_permutation_test)

    plot_probe_angle_histology_panel(min_rec_per_lab=4,
                                      perform_permutation_test=perform_permutation_test)

    font_size = 3
    fig_path = save_figure_path(figure='fig_hist')
    print(f'Saving figures to {fig_path}')

    panel_Bc = fig_path.joinpath('B_channels_subj1_hist_coronal.svg') if raw_histology else \
        repo_path().joinpath('fig_hist', 'fig_hist_schematics', 'B_channels_subj1_hist_coronal.svg')
    panel_Bs = fig_path.joinpath('B_channels_subj1_hist_sagittal.svg') if raw_histology else \
        repo_path().joinpath('fig_hist', 'fig_hist_schematics', 'B_channels_subj1_hist_sagittal.svg')
    # compose figure with svgutils
    fig = sc.Figure("200mm", "190mm",
                    
                    sc.Panel(
                        sc.SVG(repo_path().joinpath('fig_hist', 'fig_hist_schematics',
                            'A_histology_pipeline_analysis.svg')).scale(0.95).move(4, 1),
                        sc.Text("a", 1, 2.5, size=font_size, weight='bold')),
                    
                    sc.Panel(
                        sc.SVG(panel_Bc).scale(0.3).move(4, 0),
                        sc.SVG(panel_Bs).scale(0.3).move(26, 0),
                        sc.Text("b", 0, 2.5, size=font_size, weight='bold') ).move(98, 0),
                    
                    sc.Panel(
                        sc.SVG(fig_path.joinpath('C_probe_trajs_ccf_coronal.svg')).scale(0.34).move(4, 0),
                        sc.SVG(fig_path.joinpath('C_probe_trajs_ccf_sagittal.svg')).scale(0.34).move(26, 0),
                        sc.Text("c", 0, 2.5, size=font_size, weight='bold') ).move(144, 0),
                    
                    sc.Panel(
                        sc.SVG(fig_path.joinpath('surf_coord_micro_panel.svg')).scale(1.0),
                        sc.Text("d", 1, 2.5, size=font_size, weight='bold') ).move(0, 47),
                    
                    sc.Panel(
                        sc.SVG(fig_path.joinpath('surf_coord_histology_panel.svg')).scale(1.0),
                        sc.Text("e", 0, 2.5, size=font_size, weight='bold'), ).move(67, 47),
                    
                    sc.Panel(
                        sc.SVG(fig_path.joinpath('angle_histology_panel.svg')).scale(1.0),
                        sc.Text("f", 0, 2.5, size=font_size, weight='bold') ).move(133, 47),
                    
                    sc.Panel(
                        sc.Text("g", 1, 2.5, size=font_size, weight='bold') ).move(0, 109),
                    
                    sc.Panel(
                        sc.Text("h", 0, 2.5, size=font_size, weight='bold'), ).move(67, 109),
                    
                    sc.Panel(
                        sc.Text("i", 0, 2.5, size=font_size, weight='bold') ).move(133, 109),)

    fig.save(fig_path.joinpath("fig_hist.svg"))



if __name__ == "__main__":
    plot_hist_figure_new()  # generate the whole figure
    #plot_hist_figure()
