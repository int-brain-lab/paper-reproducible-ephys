#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:54 2021

This Python Script generates a figure plotting the PLANNED repeated site
insertion coordinate at the brain surface at [0,0], and then the VECTORS from 
planned surface to actual surface coord of histology tracks.  The points
of the histology track surface coords are coloured based on lab affiliation.

@author: sjwest
"""

from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import svgutils.compose as sc  # layout figure in svgutils

from iblutil.numerical import ismember

from reproducible_ephys_functions import figure_style, save_figure_path, labs, filter_recordings
from figure2.figure2_load_data import load_dataframe
from permutation_test import permut_test, permut_dist

lab_number_map, institution_map, institution_colors = labs()


def plot_probe_surf_coord_micro_panel():
    '''
    Plot the whole probe micro-manipulator panel, consisting of: 
    
    * scatterplot of the PLANNED to MICRO coords at brain surface, 
    
    * horizontal boxplot plus distplot (density plot) of all PLANNED to 
    MICRO surf coord distances (to see total distribution),
    
    * horizontal boxplots of each labs distribution
    
    * heat map of each labs permutation test p-value.?
    
    
    Panel saved to output as: surf_coord_micro_panel.svg

    Returns
    -------
    None.

    '''
    
    # generate scatterplot in first axes
    plot_probe_surf_coord(traj='micro') # saves as SVG to output
    
    # generate histogram/density plot of Euclidean distance at surface from 
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_all_lab(traj='micro')

    fig_path = save_figure_path(figure='figure2')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_surf_coord_micro_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_dist_micro_all_lab.svg')).scale(0.35).move(0, 68)))
    
    fig.save(fig_path.joinpath("surf_coord_micro_panel.svg"))


def plot_probe_surf_coord_micro_panel():
    '''
    Plot the whole probe micro-manipulator panel, consisting of:

    * scatterplot of the PLANNED to MICRO coords at brain surface,

    * horizontal boxplot plus distplot (density plot) of all PLANNED to
    MICRO surf coord distances (to see total distribution),

    * horizontal boxplots of each labs distribution

    * heat map of each labs permutation test p-value.?


    Panel saved to output as: surf_coord_micro_panel.svg

    Returns
    -------
    None.

    '''

    # generate scatterplot in first axes
    plot_probe_surf_coord(traj='micro')  # saves as SVG to output

    # generate histogram/density plot of Euclidean distance at surface from
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_all_lab(traj='micro')

    fig_path = save_figure_path(figure='figure2')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_surf_coord_micro_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_dist_micro_all_lab.svg')).scale(0.35).move(0, 64)))

    fig.save(fig_path.joinpath("surf_coord_micro_panel.svg"))


def plot_probe_surf_coord_histology_panel():
    '''
    Plot the whole probe histology panel, consisting of:

    * scatterplot of the PLANNED to HISTOLOGY coords at brain surface,

    * horizontal boxplot plus distplot (density plot) of all PLANNED to
    HISTOLOGY surf coord distances (to see total distribution),

    * horizontal boxplots of each labs distribution

    * heat map of each labs permutation test p-value.?


    Panel saved to output as: surf_coord_histology_panel.svg

    Returns
    -------
    None.

    '''

    # generate scatterplot in first axes
    plot_probe_surf_coord(traj='hist')  # saves as SVG to output

    # generate histogram/density plot of Euclidean distance at surface from
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_all_lab(traj='hist')

    fig_path = save_figure_path(figure='figure2')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_surf_coord_hist_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_dist_hist_all_lab.svg')).scale(0.35).move(0, 64)))

    fig.save(fig_path.joinpath("surf_coord_histology_panel.svg"))


def plot_probe_surf_coord(traj='micro'):
    '''Plot the PLANNED surface coord at [0,0], VECTORS from planned surface to
    actual surface coord of histology tracks, histology track points coloured
    by lab affiliation.
    '''

    # Load in data
    probe_data = load_dataframe(df_name='traj')

    # use repo-ephys figure style
    figure_style()
    fig1, ax1 = plt.subplots()

    # draw 0,0 lines
    ax1.axvline(x=-2243, color="grey", linestyle="--", linewidth=0.5)
    ax1.axhline(y=-2000, color="grey", linestyle="--", linewidth=0.5)

    for idx, row in probe_data.iterrows():

       ax1.plot([row[f'{traj}_x'], row['planned_x']], [row[f'{traj}_y'], row['planned_y']],
                color=institution_colors[institution_map[row['lab']]], linewidth=0.2, alpha=0.8)

       ax1.plot(row[f'{traj}_x'], row[f'{traj}_y'], color=institution_colors[institution_map[row['lab']]],
                marker="o", markersize=0.5, alpha=0.8, markeredgewidth=0.5)

    # Plot the mean micro coords
    # lab means
    lab_mean_x = probe_data.groupby('lab')[f'{traj}_x'].mean()
    lab_mean_y = probe_data.groupby('lab')[f'{traj}_y'].mean()
    
    for x, y, k in zip(lab_mean_x, lab_mean_y, lab_mean_x.keys()):
        ax1.plot(x, y, color=institution_colors[institution_map[k]], marker="+", markersize=3, alpha=0.5,
                 label=institution_map[k])
    
    # overall mean (mean of labs)
    mean_x = probe_data[f'{traj}_x'].mean()
    mean_y = probe_data[f'{traj}_y'].mean()
    
    ax1.plot(mean_x, mean_y, color='k', marker="+", markersize=6, alpha=0.7, label="MEAN")

    # Compute targeting error at surface of brain
    # TODO to be consistent with other figure this should be 'micro_error_surface_xy'
    # TODO DANLAB IS ALL ZERO? FOR MICRO
    df = filter_recordings(min_neuron_region=0)
    # Find the pids are that are passing the inclusion criteria
    pids = df[df['include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['include'] = isin

    top_mean_all = np.mean(probe_data[f'{traj}_error_surf_xy'].values)
    top_std_all = np.std(probe_data[f'{traj}_error_surf_xy'].values)

    top_mean_include = np.mean(probe_data[f'{traj}_error_surf_xy'][probe_data['include'] == 1].values)
    top_std_include = np.std(probe_data[f'{traj}_error_surf_xy'][probe_data['include'] == 1].values)

    # set x/y axis labels
    if traj == 'micro':
        ax1.set_xlabel('micro-manipulator ML displacement (µm)', fontsize=6)
        ax1.set_ylabel('micro-manipulator AP displacement (µm)', fontsize=6)
        ax1.set_title('MICRO-MANIPULATOR: Mean (SD) distance \n    '
                      'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
                      'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm',
                      fontsize=8)
        # add legend
        ax1.legend(loc='upper right', prop={'size': 3.5})
    else:
        ax1.set_xlabel('histology ML displacement (µm)', fontsize=6)
        ax1.set_ylabel('histology AP displacement (µm)', fontsize=6)
        ax1.set_title('HISTOLOGY: Mean (SD) distance \n    '
                      'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
                      'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm',
                      fontsize=8)

    ax1.set_xlim((-2800, -800))
    ax1.set_ylim((-3000, -1000))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()  # tighten layout around xlabel & ylabel
    fig1.set_size_inches(2.15, 2.15)

    # add a subplot INSIDE the fig1 ax1
    axav = fig1.add_axes([0.66, 0.12, 0.28, 0.28])
    axav.xaxis.tick_top()
    axav.tick_params(axis='both', labelsize=3, pad=1)

    axav.axhline(y=-2000, color="grey", linestyle="--", linewidth=0.5)
    axav.axvline(x=-2243, color="grey", linestyle="--", linewidth=0.5)

    if traj == 'micro':
        axav.set_xlim((-2350, -2000))
        axav.set_ylim((-2100, -1850))
    else:
        axav.set_xlim((-2500, -1650))
        axav.set_ylim((-2400, -1550))

    for x, y, k in zip(lab_mean_x, lab_mean_y, lab_mean_x.keys()):
        axav.plot(x, y, color=institution_colors[institution_map[k]], marker="+", markersize=5, alpha=0.7,
                  label=institution_map[k])

    axav.plot(mean_x, mean_y, color='k', marker="+", markersize=8, alpha=0.7, label="MEAN")

    fig_path = save_figure_path(figure='figure2')
    fig1.savefig(fig_path.joinpath(f'D_probe_surf_coord_{traj}_label.svg'), bbox_inches="tight")


def plot_probe_distance_all_lab(traj='micro', min_rec_per_lab=4):
    '''Plot the DISTANCES from planned to micro displacement, histogram plus
    density plot of ALL distances - to see its distribution shape.
    COMBINED with plot of distances, split by lab
    '''

    # Load in data
    probe_data = load_dataframe(df_name='traj')

    # use repo-ephys figure style
    figure_style()
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2]})

    # add institution col
    probe_data['institute'] = probe_data['lab'].map(institution_map)

    # get the histology distance
    # TODO WHY NOW NOT CONSIDERING Z?

    # create new column to indicate if each row passes advanced query
    df = filter_recordings(min_neuron_region=0)
    # Find the pids are that are passing the inclusion criteria
    pids = df[df['include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['include'] = isin
    probe_data['passed'] = np.full(len(probe_data), 'PASS')
    probe_data.loc[~probe_data['include'], 'passed'] = 'FAIL'
    # probe_data['passed'][~probe_data['include']] = 'FAIL'

    # Find the pids are that are passing the permuation test inclusion criteria
    pids = df[df['permute_include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['permute_include'] = isin

    # Create an array with the colors you want to use
    colors = ["#000000", "#FF0B04"]  # BLACK AND RED
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    sns.histplot(probe_data[f'{traj}_error_surf_xy'], kde=True, color='grey', ax=ax1)
    ax1.set_xlim(0, 1500)
    ax1.set_ylabel('count')
    ax1.set_xlabel(None)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)

    sns.stripplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data, hue='passed', size=1.5, orient="h", ax=ax2)
    
    # plot the mean line
    sns.boxplot(showmeans=True, meanline=True, meanprops={'color': 'gray', 'ls': '-', 'lw': 1}, medianprops={'visible': False},
                whiskerprops={'visible': False}, zorder=10, x=f'{traj}_error_surf_xy', y="institute", data=probe_data,
                showfliers=False, showbox=False, showcaps=False, ax=ax2)
    ax2.set_ylabel(None)
    ax2.set_xlim(0, 1500)
    if traj == 'micro':
        ax2.set_xlabel('Micromanipulator distance (µm)')
        leg = ax2.legend(fontsize=4, title='Advanced \n query', title_fontsize=6, loc='upper right', markerscale=0.2)
        plt.setp(leg.get_title(), multialignment='center')
    else:
        ax2.set_xlabel('Histology distance (µm)')
        ax2.get_legend().remove()

    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.tick_params(axis='x', labelrotation=90)

    # compute permutation testing - ALL DATA
    # For this we need to limit to labs with min_rec_per_lab
    inst_counts = probe_data['institute'].value_counts()
    remove_inst = inst_counts.index[(inst_counts < min_rec_per_lab).values]
    remove_inst = ~probe_data['institute'].isin(remove_inst).values

    # TODO why 3 times?
    probe_data_permute = probe_data[remove_inst]
    p_m1 = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values, metric=permut_dist,
                       labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)
    p_m2 = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values, metric=permut_dist,
                       labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)
    p_m3 = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values, metric=permut_dist,
                       labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)
    p_m = np.mean([p_m1, p_m2, p_m3])

    print("PERMUTATION TEST ALL : ", p_m)

    # TODO why is permutation result so different now?
    # permutation testing - PASS DATA ONLY
    probe_data_permute = probe_data[probe_data['permute_include'] == 1]
    pp_m1 = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values, metric=permut_dist,
                        labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)
    pp_m2 = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values, metric=permut_dist,
                        labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)
    pp_m3 = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values, metric=permut_dist,
                        labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)
    pp_m = np.mean([pp_m1, pp_m2, pp_m3])

    print("PERMUTATION TEST PASS : ", pp_m)
    
    ax1.set_title('Permutation Test p-value: \n    ALL : ' + str(round(p_m, 4)) + '    PASS : ' + str(round(pp_m, 4)))

    plt.tight_layout()  # tighten layout around xlabel & ylabel

    fig.set_size_inches(2.15, 2.8)

    fig_path = save_figure_path(figure='figure2')
    fig.savefig(fig_path.joinpath(f'D_probe_dist_{traj}_all_lab.svg'), bbox_inches="tight")
