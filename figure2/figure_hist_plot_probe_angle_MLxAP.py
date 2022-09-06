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
from permutation_test import permut_test, distribution_dist_approx

lab_number_map, institution_map, institution_colors = labs()


def plot_probe_angle_histology_panel():
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

    # generate scatterplot in first axes
    plot_probe_angle_histology()  # saves as SVG to output

    # generate histogram/density plot of Euclidean distance at surface from
    # planned to actual for all trajectories
    # AND plot by lab
    plot_probe_angle_histology_all_lab()

    fig_path = save_figure_path(figure='figure2')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('E_probe_angle_hist_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('E_probe_angle_hist_all_lab.svg')).scale(0.35).move(0, 64)))
    fig.save(fig_path.joinpath("angle_histology_panel.svg"))


def plot_probe_angle_histology():
    """
    Plot the PLANNED probe angle at [0,0], VECTORS from planned angle to
    actual angle of histology tracks, histology track points coloured
    by lab affiliation.
    """

    probe_data = load_dataframe(df_name='traj')

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

    df = filter_recordings(min_neuron_region=0)
    # Find the pids are that are passing the inclusion criteria
    pids = df[df['include'] == 1]['pid'].unique()
    isin, _ = ismember(probe_data.pid.values, pids)
    probe_data['include'] = isin

    angle_mean_all = np.mean(probe_data['angle'].values)
    angle_std_all = np.std(probe_data['angle'].values)

    angle_mean_include = np.mean(probe_data['angle'][probe_data['include'] == 1].values)
    angle_std_include = np.std(probe_data['angle'][probe_data['include'] == 1].values)

    # set x/y axis labels
    ax1.set_xlabel('histology ML angle (degrees)', fontsize=6)
    ax1.set_ylabel('histology AP angle (degrees)', fontsize=6)
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

    fig_path = save_figure_path(figure='figure2')
    fig1.savefig(fig_path.joinpath('E_probe_angle_hist_label.svg'), bbox_inches="tight")


def plot_probe_angle_histology_all_lab(min_rec_per_lab=4):
    '''Plot the DISTANCES from planned to histology angles, histology track
    boxplot of ALL angles - to see its distribution shape.
    '''

    # Load in data
    probe_data = load_dataframe(df_name='traj')

    # use repo-ephys figure style
    figure_style()
    fig, (ax1, ax2) = plt.subplots(2, gridspec_kw={'height_ratios': [1, 2]})

    # add institution column
    probe_data['institute'] = probe_data['lab'].map(institution_map)

    # create new column to indicate if each row passes advanced query
    df = filter_recordings(min_neuron_region=0)
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

    # Create an array with the colors you want to use
    colors = ["#000000", "#FF0B04"]  # BLACK AND RED
    # Set your custom color palette
    sns.set_palette(sns.color_palette(colors))

    #sns.histplot(probe_data['angle'], kde=True, color='grey', ax=ax1)
    sns.boxplot(y='passed', x='angle', data=probe_data, hue='passed', orient="h", fliersize=2,
                order = ['PASS', 'FAIL'], ax=ax1)
    ax1.legend_.remove()
    # round up to nearest hundred from maximum xy surface error for histoloy
    max_distance = int(math.ceil( max(probe_data['angle'])))
    ax1.set_xlim(0, max_distance)
    ax1.set_ylabel('count')
    ax1.set_xlabel(None)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.tick_params(axis='x', labelrotation=90)
    ax1.set(xticklabels=[])
    ax1.tick_params(bottom=False)

    sns.stripplot(y='institute', x='angle', data=probe_data, hue='passed', size=1.5, alpha=0.8, orient="h", ax=ax2)

    # plot the mean line
    sns.boxplot(showmeans=True, meanline=True, meanprops={'color': 'gray', 'ls': '-', 'lw': 1}, medianprops={'visible': False},
                whiskerprops={'visible': False}, zorder=10, x="angle", y="institute", data=probe_data, showfliers=False,
                showbox=False, showcaps=False, ax=ax2)
    ax2.set_ylabel(None)
    ax2.set_xlim(0, 20)
    ax2.set_xlabel('Histology angle (degrees)')
    ax2.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2.tick_params(axis='x', labelrotation=90)
    ax2.get_legend().remove()

    # compute permutation testing - ALL DATA
    # For this we need to limit to labs with min_rec_per_lab
    inst_counts = probe_data['institute'].value_counts()
    remove_inst = inst_counts.index[(inst_counts < min_rec_per_lab).values]
    remove_inst = ~probe_data['institute'].isin(remove_inst).values

    probe_data_permute = probe_data[remove_inst]
    p_m = permut_test(probe_data_permute['angle'].values, metric=distribution_dist_approx,
                       labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)

    print('\nHistology probe angle')
    print("PERMUTATION TEST ALL : ", p_m)

    # permutation testing - PASS DATA ONLY
    probe_data_permute = probe_data[probe_data['permute_include'] == 1]
    pp_m = permut_test(probe_data_permute['angle'].values, metric=distribution_dist_approx,
                        labels1=probe_data_permute['lab'].values, labels2=probe_data_permute['subject'].values)

    print("PERMUTATION TEST PASS : ", pp_m)

    ax1.set_title('Histology-to-planned angle', fontsize=7)
    #ax1.set_title('Permutation Test p-value: \n    ALL : ' + str(round(p_m, 4)) + '    PASS : ' + str(round(pp_m, 4)))

    plt.tight_layout()  # tighten layout around xlabel & ylabel
    fig.set_size_inches(2.15, 2.8)

    fig_path = save_figure_path(figure='figure2')
    fig.savefig(fig_path.joinpath('E_probe_angle_hist_all_lab.svg'), bbox_inches="tight")
