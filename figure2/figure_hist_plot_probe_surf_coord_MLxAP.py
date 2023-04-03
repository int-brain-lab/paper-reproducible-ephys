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

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import svgutils.compose as sc  # layout figure in svgutils
import math

from iblutil.numerical import ismember

from reproducible_ephys_functions import figure_style, save_figure_path, labs, filter_recordings
from figure2.figure2_load_data import load_dataframe
from permutation_test import permut_test, distribution_dist_approx_max

lab_number_map, institution_map, institution_colors = labs()


def plot_probe_surf_coord_micro_panel(min_rec_per_lab=4, perform_permutation_test=True):
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
    plot_probe_surf_coord(traj='micro', min_rec_per_lab=min_rec_per_lab)  # saves as SVG to output

    # generate histogram/density plot of Euclidean distance at surface from
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_all_lab(traj='micro', min_rec_per_lab=min_rec_per_lab,
                                perform_permutation_test=perform_permutation_test)

    fig_path = save_figure_path(figure='figure2')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_surf_coord_micro_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_dist_micro_all_lab.svg')).scale(0.35).move(0, 64)))

    fig.save(fig_path.joinpath("surf_coord_micro_panel.svg"))


def plot_probe_surf_coord_histology_panel(min_rec_per_lab=4, perform_permutation_test=True):
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
    plot_probe_surf_coord(traj='hist', min_rec_per_lab=min_rec_per_lab)  # saves as SVG to output

    # generate histogram/density plot of Euclidean distance at surface from
    # planned to actual for all trajectories AND dotplots by lab
    plot_probe_distance_all_lab(traj='hist', min_rec_per_lab=min_rec_per_lab,
                                perform_permutation_test=perform_permutation_test)

    fig_path = save_figure_path(figure='figure2')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_surf_coord_hist_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_dist_hist_all_lab.svg')).scale(0.35).move(0, 64)))

    fig.save(fig_path.joinpath("surf_coord_histology_panel.svg"))


def plot_probe_surf_coord(traj='micro', min_rec_per_lab=4):
    '''Plot the PLANNED surface coord at [0,0], VECTORS from planned surface to
    actual surface coord of histology tracks, histology track points coloured
    by lab affiliation.
    '''

    # Load in data
    probe_data = load_dataframe(df_name='traj')

    # map labs to institutions
    probe_data['institution'] = probe_data['lab'].map(institution_map)

    # for micro-manipulator EXCLUDE wheremicro-manipulator data was not recorded - planned == micro
    if traj == 'micro':
        # exclude data where planned xyz == micro xyz
        xPM = probe_data['planned_x'] == probe_data['micro_x']
        yPM = probe_data['planned_y'] == probe_data['micro_y']
        zPM = probe_data['planned_z'] == probe_data['micro_z']
        probe_data = probe_data[ (xPM & yPM & zPM) == False ].reset_index()

    # use repo-ephys figure style
    figure_style()
    fig1, ax1 = plt.subplots()

    # draw 0,0 lines
    ax1.axvline(x=-2243, color="grey", linestyle="--", linewidth=0.5)
    ax1.axhline(y=-2000, color="grey", linestyle="--", linewidth=0.5)

    for idx, row in probe_data.iterrows():

        ax1.plot([row[f'{traj}_x'], row['planned_x']], [row[f'{traj}_y'], row['planned_y']],
                 color=institution_colors[institution_map[row['lab']]], 
                 linewidth=0.2, alpha=0.8)

        ax1.plot(row[f'{traj}_x'], row[f'{traj}_y'], 
                 color=institution_colors[institution_map[row['lab']]],
                 marker="o", markersize=0.5, alpha=0.8, markeredgewidth=0.5)

    # Plot the mean micro coords - institution means
    inst_mean_x = probe_data.groupby('institution')[f'{traj}_x'].mean()
    inst_mean_y = probe_data.groupby('institution')[f'{traj}_y'].mean()

    for x, y, k in zip(inst_mean_x, inst_mean_y, inst_mean_x.keys()):
        ax1.plot(x, y, 
                 color=institution_colors[k], 
                 label=k, 
                 marker="+", markersize=3, alpha=0.5)

    # overall mean (mean of institutions)
    mean_x = probe_data[f'{traj}_x'].mean()
    mean_y = probe_data[f'{traj}_y'].mean()

    ax1.plot(mean_x, mean_y, color='k', marker="+", markersize=6, alpha=0.7, label="MEAN")
    ax1.tick_params(axis='both', which='major', labelsize=5)

    # Compute targeting error at surface of brain
    df = filter_recordings(by_anatomy_only=True, min_neuron_region=0)
    #df = filter_recordings(by_anatomy_only=True, min_rec_lab=min_rec_per_lab)
    
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
        ax1.set_xlabel('micro-manipulator ML displacement (µm)', fontsize=7)
        ax1.set_ylabel('micro-manipulator AP displacement (µm)', fontsize=7)
        ax1.set_title('Targeting variability')
        #ax1.set_title('MICRO-MANIPULATOR: Mean (SD) distance \n    '
        #              'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
        #              'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm',
        #              fontsize=8)
        
        # print average values
        print('MICRO-MANIPULATOR: Mean (SD) distance \n    '
              'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
              'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm')
        # add legend
        ax1.legend(loc='upper right', prop={'size': 3.5})
    else:
        ax1.set_xlabel('histology ML displacement (µm)', fontsize=7)
        ax1.set_ylabel('histology AP displacement (µm)', fontsize=7)
        ax1.set_title('Geometrical variability')
        #ax1.set_title('HISTOLOGY: Mean (SD) distance \n    '
        #              'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
        #              'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm',
        #              fontsize=8)
        # print average values
        print('HISTOLOGY: Mean (SD) distance \n    '
              'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
              'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm')

    if traj == 'micro':
        ax1.set_xlim((-2500, -1800))
        ax1.set_ylim((-2500, -1500))
    else:
        ax1.set_xlim((-3000, -1000))
        ax1.set_ylim((-3000, -1000))
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

    plt.tight_layout()  # tighten layout around xlabel & ylabel
    fig1.set_size_inches(2.15, 2.15)

    # add a subplot INSIDE the fig1 ax1
    #axav = fig1.add_axes([0.66, 0.12, 0.28, 0.28])
    #axav.xaxis.tick_top()
    #axav.tick_params(axis='both', labelsize=3, pad=1)

    #axav.axhline(y=-2000, color="grey", linestyle="--", linewidth=0.5)
    #axav.axvline(x=-2243, color="grey", linestyle="--", linewidth=0.5)

    #if traj == 'micro':
    #    axav.set_xlim((-2350, -2000))
    #    axav.set_ylim((-2100, -1850))
    #else:
    #    axav.set_xlim((-2500, -1650))
    #    axav.set_ylim((-2400, -1550))

    #for x, y, k in zip(inst_mean_x, inst_mean_y, inst_mean_x.keys()):
    #    axav.plot(x, y, color=institution_colors[institution_map[k]], marker="+", markersize=5, alpha=0.7,
    #              label=institution_map[k])

    #axav.plot(mean_x, mean_y, color='k', marker="+", markersize=8, alpha=0.7, label="MEAN")

    fig_path = save_figure_path(figure='figure2')
    fig1.savefig(fig_path.joinpath(f'D_probe_surf_coord_{traj}_label.svg'), bbox_inches="tight")


def plot_probe_distance_all_lab(traj='micro', min_rec_per_lab=4, perform_permutation_test=True):
    '''Plot the DISTANCES from planned to micro displacement, histogram plus
    density plot of ALL distances - to see its distribution shape.
    COMBINED with plot of distances, split by lab
    '''
    
    # Load data
    probe_data = load_dataframe(df_name='traj')

    # for micro-manipulator EXCLUDE wheremicro-manipulator data was not recorded - planned == micro
    if traj == 'micro':
        # exclude data where planned xyz == micro xyz
        xPM = probe_data['planned_x'] == probe_data['micro_x']
        yPM = probe_data['planned_y'] == probe_data['micro_y']
        zPM = probe_data['planned_z'] == probe_data['micro_z']
        probe_data = probe_data[ (xPM & yPM & zPM) == False ].reset_index()

    # add institution col
    probe_data['institute'] = probe_data['lab'].map(institution_map)

    # create new column to indicate if each row passes anatomy exclusions
    df = filter_recordings(by_anatomy_only=True, min_rec_lab=min_rec_per_lab)

    # Find the pids are that are passing the inclusion criteria
    pids = df[df['include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['include'] = isin
    probe_data['passed'] = np.full(len(probe_data), 'PASS')
    probe_data.loc[~probe_data['include'], 'passed'] = 'FAIL'
    # probe_data['passed'][~probe_data['include']] = 'FAIL'

    # Find the pids are that are passing the permutation test inclusion criteria
    pids = df[df['permute_include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['permute_include'] = isin

    # remove any institutes which have N less than min_rec_per_lab
    pd_inst_counts = probe_data['institute'].value_counts()
    # keys returns the value counts names (ie. institute strings!)
    inst_ex = pd_inst_counts.keys()[pd_inst_counts < min_rec_per_lab]
    # remove each institute that has too few n
    for ie in inst_ex:
        print('excluding institute from ALL-PROBE analysis (below min_rec_per_lab): ', ie)
        probe_data = probe_data[probe_data['institute'] != ie]

    # get pass-only data & plot data (adding negative value distances to ensure institution is still plotted)
    probe_data_pass = probe_data[ probe_data['passed'] == 'PASS']
    probe_data_pass = probe_data_pass.reset_index()
    probe_data_pass_plot = probe_data[ probe_data['passed'] == 'PASS']
    probe_data_pass_plot = probe_data_pass_plot.reset_index()
    probe_data_pass_plot_count = probe_data[ probe_data['passed'] == 'PASS']
    probe_data_pass_plot_count = probe_data_pass_plot_count.reset_index()

    # remove any institutes which have N less than min_rec_per_lab
    pd_inst_counts = probe_data_pass['institute'].value_counts()
    # keys returns the value counts names (ie. institute strings!)
    inst_ex = pd_inst_counts.keys()[pd_inst_counts < min_rec_per_lab]
    # remove each institute that has too few n
    for ie in inst_ex:
        print('excluding institute from PASS-ONLY analysis (below min_rec_per_lab): ', ie)
        # remove all institute entries
        probe_data_pass = probe_data_pass[probe_data_pass['institute'] != ie]
        
        # for plot data
        # get copy of first row of institution ie
        s = probe_data_pass_plot['institute'] == ie
        si = s[s].index[0]
        sir = probe_data_pass_plot.iloc[si].copy()
        # modify this to set distance to -100
        sir[f'{traj}_error_surf_xy'] = -100.0
        # remove all institute entries
        probe_data_pass_plot = probe_data_pass_plot[probe_data_pass_plot['institute'] != ie]
        probe_data_pass_plot_count = probe_data_pass_plot_count[probe_data_pass_plot_count['institute'] != ie]
        # now re-add sir with angle set to -1.0
        probe_data_pass_plot = probe_data_pass_plot.append(sir, ignore_index=True)

    # list pids for figure: probe_data_pass_plot_count['pid'].reset_index()['pid']
    # probe_data['pid'].reset_index()['pid']

    # use repo-ephys figure style
    figure_style()

    # generate 2x2 subplots with 1:9 height ratios
    widths = [1, 1]
    heights = [1, 9]
    gs_kw = dict(width_ratios=widths, height_ratios=heights)
    fig, fig_axes = plt.subplots(ncols=2, nrows=2, constrained_layout=True,
        gridspec_kw=gs_kw)
    axr0c0 = fig_axes[0][0]
    axr1c0 = fig_axes[1][0]
    axr0c1 = fig_axes[0][1]
    axr1c1 = fig_axes[1][1]

    # Create an array with colors to use
    colors_pts = ["#0BFF0B", "#FF0B0B"]  # GREEN AND RED FOR PASS/FAIL

    # Plot ALL kdeplot + boxplot/stripplot
    #sns.histplot(probe_data[f'{traj}_error_surf_xy'], kde=True, color='grey', ax=ax1)
    #sns.boxplot(y='passed', x=f'{traj}_error_surf_xy', data=probe_data, hue='passed', orient="h", fliersize=2,
    #            order = ['PASS', 'FAIL'], ax=ax1)
    #ax1.legend_.remove()
    sns.kdeplot( x=f'{traj}_error_surf_xy', data=probe_data, color='#000000', fill=True, ax=axr0c0)
    axr0c0.tick_params(axis='both', which='major', labelsize=5)
    # round up to nearest hundred from maximum xy surface error for histoloy
    max_distance = int(math.ceil( max(probe_data[f'{traj}_error_surf_xy']) / 100.0)) * 100
    axr0c0.set_xlim(0, max_distance)
    axr0c0.set_ylabel('density', fontsize=6)
    axr0c0.set_xlabel(None)
    axr0c0.set(xticklabels=[])
    axr0c0.tick_params(bottom=False)

    # compute order metrics
    dec_med = probe_data.groupby(by=["institute"])[f'{traj}_error_surf_xy'].mean().sort_values().index
    asc_med = probe_data.groupby(by=["institute"])[f'{traj}_error_surf_xy'].mean().sort_values().iloc[::-1].index
    asc_passed = probe_data['institute'][probe_data['passed'] == "PASS"].value_counts().sort_values().index

    # order by descending number of passed recordings
    order_by = asc_med

    # and get the correct color order for institutions
    order_colors=[]
    for ob in order_by:
        order_colors.append(institution_colors[ob])

    # plot boxplot
    #sns.boxplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data, orient="h", showfliers = False, 
    #            order=order_by, ax=axr1c0)
    sns.boxplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data, 
                palette=order_colors, orient="h", showfliers=False, linewidth=0.5, 
                order=order_by, ax=axr1c0)

    # overlay points
    axx = sns.stripplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data, hue='passed', 
                        size=1.5,  orient="h", palette=colors_pts, order=order_by, ax=axr1c0)

    # plot overall mean distance (mean of labs)
    mean_error_surf_xy = probe_data[f'{traj}_error_surf_xy'].mean()
    axr1c0.axvline(x=mean_error_surf_xy, linestyle='--', linewidth=0.5, color='gray')

    axr1c0.tick_params(axis='both', which='major', labelsize=5)
    axr1c0.set_ylabel(None)
    axr1c0.set_xlim(0, max_distance)
    axr1c0.set_xlabel(None)
    axr1c0.xaxis.set_major_locator(plt.MaxNLocator(5))
    axr1c0.tick_params(axis='x', labelrotation=90)
    axr1c0.get_legend().remove()

    # Plot PASS-ONLY kdeplot + boxplot/stripplot
    #sns.histplot(probe_data[f'{traj}_error_surf_xy'], kde=True, color='grey', ax=ax1)
    #sns.boxplot(y='passed', x=f'{traj}_error_surf_xy', data=probe_data, hue='passed', orient="h", fliersize=2,
    #            order = ['PASS', 'FAIL'], ax=ax1)
    #ax1.legend_.remove()
    sns.kdeplot( x=f'{traj}_error_surf_xy', data=probe_data_pass_plot, color='#0BFF0B', fill=True, ax=axr0c1)
    # round up to nearest hundred from maximum xy surface error for histoloy
    axr0c1.tick_params(axis='both', which='major', labelsize=5)
    max_distance = int(math.ceil( max(probe_data[f'{traj}_error_surf_xy']) / 100.0)) * 100
    axr0c1.set_xlim(0, max_distance)
    axr0c1.set_ylabel(None)
    axr0c1.set_xlabel(None)
    axr0c1.set(xticklabels=[])
    axr0c1.tick_params(bottom=False)
    axr0c1.set(yticklabels=[])
    axr0c1.tick_params(left=False)

    # plot boxplot
    #sns.boxplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data_pass_plot, orient="h", showfliers = False, 
    #            order=order_by, ax=axr1c1)
    sns.boxplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data_pass_plot, 
                palette=order_colors, orient="h", showfliers=False, linewidth=0.5, 
                order=order_by, ax=axr1c1)

    # overlay points
    axx = sns.stripplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data_pass_plot, hue='passed', 
                        size=1.5,  orient="h", palette=colors_pts, order=order_by, ax=axr1c1)

    # plot overall mean distance (mean of labs)
    mean_error_surf_xy_pass = probe_data_pass_plot[f'{traj}_error_surf_xy'].mean()
    axr1c1.axvline(x=mean_error_surf_xy_pass, linestyle='--', linewidth=0.5, color='#B4FFB4')

    axr1c1.tick_params(axis='both', which='major', labelsize=5)
    axr1c1.set_ylabel(None)
    axr1c1.set_xlim(0, max_distance)
    axr1c1.set_xlabel(None)
    axr1c1.xaxis.set_major_locator(plt.MaxNLocator(5))
    axr1c1.tick_params(axis='x', labelrotation=90)
    axr1c1.get_legend().remove()
    axr1c1.set(yticklabels=[])
    axr1c1.tick_params(left=False)

    if traj == 'micro':
        # add legend to micro group plot only
        handles, labels = axr1c0.get_legend_handles_labels()
        axr1c0.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.98, 0.05), loc=4, 
               borderaxespad=0., prop={'size': 4}, markerscale=0.2)
        fig.suptitle('Micromanipulator-to-planned distance', fontsize=7)
        fig.supxlabel('Micromanipulator distance (µm)', fontsize=7)
    else:
        fig.suptitle('Histology-to-planned distance', fontsize=7)
        fig.supxlabel('Histology distance (µm)', fontsize=7)


    plt.tight_layout()  # tighten layout around xlabel & ylabel
    fig.set_size_inches(2.15, 2.8)

    fig_path = save_figure_path(figure='figure2')
    fig.savefig(fig_path.joinpath(f'D_probe_dist_{traj}_all_lab.svg'), bbox_inches="tight")

    if perform_permutation_test == True:
        
        # compute permutation testing - ALL DATA
        probe_data_permute = probe_data
        p_m = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values,
                          metric=distribution_dist_approx_max,
                          labels1=probe_data_permute['lab'].values,
                          labels2=probe_data_permute['subject'].values,
                          n_cores=8, n_permut=500000)
        
        if traj == 'micro':
            print('\nMicro-Manipulator brain surface coordinate')
        else:
            print('\nHistology brain surface coordinate')
        
        print("PERMUTATION TEST ALL : ", p_m)
    
        # permutation testing - PASS DATA ONLY
        probe_data_permute = probe_data_pass[probe_data_pass['permute_include'] == 1]
        pp_m = permut_test(probe_data_permute[f'{traj}_error_surf_xy'].values,
                           metric=distribution_dist_approx_max,
                           labels1=probe_data_permute['lab'].values,
                           labels2=probe_data_permute['subject'].values,
                           n_cores=8, n_permut=500000)
        
        print("PERMUTATION TEST PASS : ", pp_m)


