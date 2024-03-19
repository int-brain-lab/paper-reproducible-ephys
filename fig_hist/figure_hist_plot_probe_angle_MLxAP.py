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
from fig_hist.fig_hist_load_data import load_dataframe
from permutation_test import permut_test, distribution_dist_approx_max

lab_number_map, institution_map, institution_colors = labs()


def plot_probe_angle_histology_panel(min_rec_per_lab=4, perform_permutation_test=True):
    """
    Plot the whole probe histology panel, consisting of:

    * scatterplot of the PLANNED to HISTOLOGY angles at brain surface,

    * horizontal boxplot plus distplot (density plot) of all PLANNED to
    HISTOLOGY angle values (to see total distribution),

    * horizontal boxplots of each labs distribution

    * heat map of each labs permutation test p-value.?

    Panel saved to output as: angle_histology_panel.svg

    Returns
    -------
    None.

    """

    sns.set(font="serif") # stop error findfont: Font family 'Arial' not found.
    # generate scatterplot in first axes
    plot_probe_angle_histology(min_rec_per_lab=min_rec_per_lab)  # saves as SVG to output

    # generate histogram/density plot of Euclidean distance at surface from
    # planned to actual for all trajectories
    # AND plot by lab
    plot_probe_angle_histology_all_lab(min_rec_per_lab=min_rec_per_lab, 
                                       perform_permutation_test=perform_permutation_test)

    fig_path = save_figure_path(figure='fig_hist')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('E_probe_angle_hist_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('E_probe_angle_hist_all_lab.svg')).scale(0.35).move(0, 64)))
    fig.save(fig_path.joinpath("angle_histology_panel.svg"))


def plot_probe_angle_histology(traj='hist', min_rec_per_lab=4):
    """
    Plot the PLANNED probe angle at [0,0], VECTORS from planned angle to
    actual angle of histology tracks, histology track points coloured
    by lab affiliation.
    """

    probe_data = load_dataframe(df_name='traj')

    # manually exclude (for now) unresolved PIDs + one odd?(68d) looking insertion..?
    pids = [
        '2ff92e61-c2af-4dbf-8862-bd50b344762b',
        '8f1d5aad-8c1f-4e81-869a-5a1ab1bf53b2',
        'b53cc868-008a-4d20-a33a-ea3101be2d34',
        'f06d6cd9-a6b8-49a4-90d1-7905d04c2f8b',
        '84fd7fa3-6c2d-4233-b265-46a427d3d68d']

    for p in pids:
        probe_data = probe_data[ probe_data['pid'] != p]

    figure_style()
    fig1, ax1 = plt.subplots()

    # draw 0,0 lines
    ax1.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    ax1.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)

    for idx, row in probe_data.iterrows():

        ax1.plot([row['angle_ml'], 0], [row['angle_ap'], 0],
                 color=institution_colors[institution_map[row['lab']]], linewidth=0.15, alpha=0.8)
        ax1.plot(row['angle_ml'], row['angle_ap'], color=institution_colors[institution_map[row['lab']]],
                 marker="o", markersize=0.5, alpha=0.8, markeredgewidth=0.5)

    # Plot the mean micro coords
    # lab means
    lab_mean_ml = probe_data.groupby('lab')['angle_ml'].mean()
    lab_mean_ap = probe_data.groupby('lab')['angle_ap'].mean()

    for ml, ap, k in zip(lab_mean_ml, lab_mean_ap, lab_mean_ml.keys()):
        ax1.plot(ml, ap, color=institution_colors[institution_map[k]], marker="+", markersize=3, alpha=0.5,
                 label=institution_map[k])

    # overall mean (mean of labs)
    mean_ml = probe_data['angle_ml'].mean()
    mean_ap = probe_data['angle_ap'].mean()

    ax1.plot(mean_ml, mean_ap, color='k', marker="+", markersize=6, alpha=0.7, label="MEAN")
    ax1.tick_params(axis='both', which='major', labelsize=5)

    df = filter_recordings(by_anatomy_only=True, min_neuron_region=0)
    #df = filter_recordings(by_anatomy_only=True, min_rec_lab=min_rec_per_lab)
    # Find the pids are that are passing the inclusion criteria
    pids = df[df['include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['include'] = isin

    angle_mean_all = np.mean(probe_data['angle'].values)
    angle_std_all = np.std(probe_data['angle'].values)

    angle_mean_include = np.mean(probe_data['angle'][probe_data['include'] == 1].values)
    angle_std_include = np.std(probe_data['angle'][probe_data['include'] == 1].values)

    # set x/y axis labels
    ax1.set_xlabel('histology ML angle (degrees)', fontsize=7)
    ax1.set_ylabel('histology AP angle (degrees)', fontsize=7)
    ax1.set_title('Angle variability')
    # add mean trageting error distance to title
    print('Mean (SD) angle \n' +
                  'ALL : ' + str(np.around(angle_mean_all, 1)) + ' (' + str(np.around(angle_std_all, 2)) + ')' + ' degrees \n' +
                  'PASS : ' + str(np.around(angle_mean_include, 1)) + ' (' + str(np.around(angle_std_include, 2)) + ')'
                  + ' degrees')

    ax1.xaxis.set_major_locator(plt.MaxNLocator(7))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(7))

    ax1.set_ylim(-20, 10)
    ax1.set_xlim(-20, 10)

    # plt.tight_layout()  # tighten layout around xlabel & ylabel

    # add a subplot INSIDE the fig1 ax1
    #axav = fig1.add_axes([0.1, 0.12, 0.28, 0.28])
    #axav.xaxis.tick_top()
    #axav.yaxis.tick_right()
    #axav.tick_params(axis='both', labelsize=3, pad=1)
    #axav.axhline(y=0, color="grey", linestyle="--", linewidth=0.5)
    #axav.axvline(x=0, color="grey", linestyle="--", linewidth=0.5)
    #axav.set_xlim((-10, 5))
    #axav.set_ylim((-10, 5))

    #for ml, ap, k in zip(lab_mean_ml, lab_mean_ap, lab_mean_ml.keys()):
    #    axav.plot(ml, ap, color=institution_colors[institution_map[k]], marker="+", markersize=5, alpha=0.7,
    #              label=institution_map[k])
    #axav.plot(mean_ml, mean_ap, color='k', marker="+", markersize=8, alpha=0.7, label="MEAN")

    plt.tight_layout()
    fig1.set_size_inches(2.15, 2.15)

    fig_path = save_figure_path(figure='fig_hist')
    fig1.savefig(fig_path.joinpath('E_probe_angle_hist_label.svg'), bbox_inches="tight")


def plot_probe_angle_histology_all_lab(traj='hist', min_rec_per_lab=4, perform_permutation_test=True):
    '''Plot the DISTANCES from planned to histology angles, histology track
    boxplot of ALL angles - to see its distribution shape.
    '''

    # Load data
    probe_data = load_dataframe(df_name='traj')

    # manually exclude (for now) unresolved PIDs + one odd?(68d) looking insertion..?
    pids = [
        '2ff92e61-c2af-4dbf-8862-bd50b344762b',
        '8f1d5aad-8c1f-4e81-869a-5a1ab1bf53b2',
        'b53cc868-008a-4d20-a33a-ea3101be2d34',
        'f06d6cd9-a6b8-49a4-90d1-7905d04c2f8b',
        '84fd7fa3-6c2d-4233-b265-46a427d3d68d']

    for p in pids:
        probe_data = probe_data[ probe_data['pid'] != p]

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

    # get pass-only data
    probe_data_pass = probe_data[ probe_data['passed'] == 'PASS']
    probe_data_pass = probe_data_pass.reset_index()
    probe_data_pass_plot = probe_data[ probe_data['passed'] == 'PASS']
    probe_data_pass_plot = probe_data_pass_plot.reset_index()

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
        # modify this to set angle to -1
        sir['angle'] = -1.0
        # remove all institute entries
        probe_data_pass_plot = probe_data_pass_plot[probe_data_pass_plot['institute'] != ie]
        # now re-add sir with angle set to -1.0
        probe_data_pass_plot = probe_data_pass_plot.append(sir, ignore_index=True)

    # use repo-ephys figure style
    figure_style()
    
    # generate 2x2 subplots with 1:9 height ratios
    fig, fig_axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True,
        gridspec_kw={'height_ratios': [1, 9]})
    axr0c0 = fig_axes[0]
    axr1c0 = fig_axes[1]

    # Create an array with the colors you want to use
    #colors_pts = ["#0BFF0B", "#FF0B0B"]  # GREEN AND RED FOR PASS/FAIL
    colors_pts = ["#000000"]  # BLACK

    # Set your custom color palette
    #sns.set_palette(sns.color_palette(colors))

    # Plot ALL kdeplot + boxplot/stripplot
    #sns.histplot(probe_data['angle'], kde=True, color='grey', ax=ax1)
    #sns.boxplot(y='passed', x='angle', data=probe_data, hue='passed', orient="h", fliersize=2,
    #            order = ['PASS', 'FAIL'], ax=ax1)
    #ax1.legend_.remove()
    sns.kdeplot( x='angle', data=probe_data, color='#000000', fill=True, ax=axr0c0)
    axr0c0.tick_params(axis='both', which='major', labelsize=5)
    # round up to nearest hundred from maximum xy surface error for histoloy
    max_distance = int(1.1 * math.ceil( max(probe_data['angle'])))
    axr0c0.set_xlim(0, max_distance)
    axr0c0.set_ylabel('density', fontsize=6)
    axr0c0.set_xlabel(None)
    axr0c0.set(xticklabels=[])
    axr0c0.tick_params(bottom=False)

    # compute order metrics
    dec_med = probe_data.groupby(by=["institute"])['angle'].mean().sort_values().index
    asc_med = probe_data.groupby(by=["institute"])['angle'].mean().sort_values().iloc[::-1].index
    asc_passed = probe_data['institute'][probe_data['passed'] == "PASS"].value_counts().sort_values().index

    # order by descending number of passed recordings
    order_by = asc_med

    # and get the correct color order for institutions
    order_colors=[]
    for ob in order_by:
        order_colors.append(institution_colors[ob])

    # plot boxplot
    #sns.boxplot(y='institute', x='angle', data=probe_data, orient="h", showfliers = False, 
    #            order=order_by, ax=axr1c0)
    sns.boxplot(y='institute', x='angle', data=probe_data, orient="h", showfliers = False, 
                palette=order_colors, linewidth = 0.5, order=order_by, ax=axr1c0)

    # overlay points
    axx = sns.stripplot(y='institute', x='angle', data=probe_data, size=1.5,  
                        orient="h", color="#000000", order=order_by, ax=axr1c0)

    # plot overall mean angle (mean of labs)
    mean_error_angle = probe_data['angle'].mean()
    axr1c0.axvline(x=mean_error_angle, linestyle='--', linewidth=0.5, color='gray')

    axr1c0.tick_params(axis='both', which='major', labelsize=5)
    axr1c0.set_ylabel(None)
    axr1c0.set_xlim(0, max_distance)
    axr1c0.set_xlabel(None)
    axr1c0.xaxis.set_major_locator(plt.MaxNLocator(5))
    #axr1c0.tick_params(axis='x', labelrotation=90)
    #axr1c0.get_legend().remove()

    fig.suptitle('Histology-to-planned angle', fontsize=7)
    #fig.supxlabel('\nHistology angle (degrees)', fontsize=7)
    axr1c0.set_xlabel('Histology angle (degrees)')

    plt.tight_layout()  # tighten layout around xlabel & ylabel
    fig.set_size_inches(2.15, 2.8)

    fig_path = save_figure_path(figure='fig_hist')
    fig.savefig(fig_path.joinpath('E_probe_angle_hist_all_lab.svg'), bbox_inches="tight")

    if perform_permutation_test == True:
        # compute permutation testing - ALL DATA
        probe_data_permute = probe_data
        p_m = permut_test(probe_data_permute['angle'].values, 
                          metric=distribution_dist_approx_max,
                          labels1=probe_data_permute['lab'].values,
                          labels2=probe_data_permute['subject'].values,
                          n_cores=8, n_permut=500000)
    
        print('\nHistology probe angle')
        print("PERMUTATION TEST ALL : ", p_m)


