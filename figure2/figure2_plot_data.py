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
from figure2.figure_hist_plot_probe_angle_MLxAP import plot_probe_angle_histology_panel
from figure2.figure_hist_plot_probe_surf_coord_MLxAP import (plot_probe_surf_coord_histology_panel,
                                                             plot_probe_surf_coord_micro_panel)
from figure2.figure_hist_plot_probe_trajs_ccf import plot_trajs
from figure2.figure_hist_plot_channels_all_subjs import plot_channels_n3
from reproducible_ephys_functions import save_figure_path, repo_path


def plot_hist_figure():

    print('')
    print('PLOT HISTOLOGY FIGURE')
    print('=====================')
    print('')

    print(' run plot_channels_n3()')
    plot_channels_n3()
    
    print(' run plot_trajs()')
    plot_trajs()
    
    print(' run plot_probe_surf_coord_micro_panel()')
    plot_probe_surf_coord_micro_panel()
    
    print(' run plot_probe_surf_coord_histology_panel()')
    plot_probe_surf_coord_histology_panel()
    
    print(' run plot_probe_angle_histology_panel()')
    plot_probe_angle_histology_panel()
    
    font_size = 3
    fig_path = save_figure_path(figure='figure2')
    # compose figure with svgutils
    fig = sc.Figure("200mm", "190mm",
        
        sc.Panel(
            sc.Text("a", 1, 2.5, size=font_size, weight='bold'),
            sc.SVG(repo_path().joinpath('figure2', 'A_histology_pipeline_analysis.svg')).scale(0.95).move(3, 0),),
        sc.Panel(
            sc.Text("b", 0, 2.5, size=font_size, weight='bold'),
            sc.SVG(fig_path.joinpath('B_channels_subj3_hist_coronal.svg')).scale(0.3).move(2, 0)).move(90, 0),
        sc.Panel(
            sc.Text("c", 0, 2.5, size=font_size, weight='bold'),
            sc.SVG(fig_path.joinpath('C_probe_trajs_ccf_coronal.svg')).scale(0.34),
            sc.SVG(fig_path.joinpath('C_probe_trajs_ccf_sagittal.svg')).scale(0.34).move(22, 0),).move(153, 0),
        sc.Panel(
            sc.Text("d", 1, 2.5, size=font_size, weight='bold'),
            sc.SVG(fig_path.joinpath('surf_coord_micro_panel.svg')).scale(1.0),).move(0, 47),
        sc.Panel(
            sc.Text("e", 0, 2.5, size=font_size, weight='bold'),
            sc.SVG(fig_path.joinpath('surf_coord_histology_panel.svg')).scale(1.0),).move(67, 47),
        sc.Panel(
            sc.Text("f", 0, 2.5, size=font_size, weight='bold'),
            sc.SVG(fig_path.joinpath('angle_histology_panel.svg')).scale(1.0),).move(133, 47),)
    
    fig.save(fig_path.joinpath("figure2.svg"))
    

if __name__ == "__main__":
    plot_hist_figure()  # generate the whole figure


