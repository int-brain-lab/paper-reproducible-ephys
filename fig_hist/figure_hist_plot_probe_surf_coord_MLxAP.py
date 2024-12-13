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
import matplotlib as mpl
import seaborn as sns
import numpy as np
import svgutils.compose as sc  # layout figure in svgutils
import math

from iblutil.numerical import ismember

from reproducible_ephys_functions import figure_style, save_figure_path, LAB_MAP, filter_recordings
from fig_hist.fig_hist_load_data import load_dataframe
from permutation_test import permut_test, distribution_dist_approx_max

lab_number_map, institution_map, institution_colors = LAB_MAP()

PRINT_INFO = False

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

    fig_path = save_figure_path(figure='fig_hist')
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

    fig_path = save_figure_path(figure='fig_hist')
    fig = sc.Figure("66mm", "140mm",
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_surf_coord_hist_label.svg')).scale(0.35)),
                    sc.Panel(sc.SVG(fig_path.joinpath('D_probe_dist_hist_all_lab.svg')).scale(0.35).move(0, 64)))

    fig.save(fig_path.joinpath("surf_coord_histology_panel.svg"))


def plot_probe_surf_coord(traj='micro', min_rec_per_lab=4, ax1=None, save=True):
    '''Plot the PLANNED surface coord at [0,0], VECTORS from planned surface to
    actual surface coord of histology tracks, histology track points coloured
    by lab affiliation.
    '''

    # Load in data
    probe_data = load_dataframe(df_name='traj')

    # manually exclude (for now) unresolved PIDs + one odd?(68d) looking insertion..? 38a1 - NR_0031, micro data far off??
    pids = [
        '2ff92e61-c2af-4dbf-8862-bd50b344762b',
        '8f1d5aad-8c1f-4e81-869a-5a1ab1bf53b2',
        'b53cc868-008a-4d20-a33a-ea3101be2d34',
        'f06d6cd9-a6b8-49a4-90d1-7905d04c2f8b',
        '84fd7fa3-6c2d-4233-b265-46a427d3d68d',
        '4836a465-c691-4852-a0b1-dcd2b1ce38a1']

    for p in pids:
        probe_data = probe_data[ probe_data['pid'] != p]

    # map labs to institutions
    probe_data['institute'] = probe_data['lab'].map(institution_map)

    # remove any institutes which have N less than min_rec_per_lab
    pd_inst_counts = probe_data['institute'].value_counts()
    # keys returns the value counts names (ie. institute strings!)
    inst_ex = pd_inst_counts.keys()[pd_inst_counts < min_rec_per_lab]
    # remove each institute that has too few n
    for ie in inst_ex:
        print('\n\nexcluding institute from ALL-PROBE analysis (below min_rec_per_lab): ', ie)
        probe_data = probe_data[probe_data['institute'] != ie].reset_index()

    # for micro-manipulator EXCLUDE where micro-manipulator data was not recorded - planned == micro
    if traj == 'micro':
        # exclude data where planned xyz == micro xyz
        xPM = probe_data['planned_x'] == probe_data['micro_x']
        yPM = probe_data['planned_y'] == probe_data['micro_y']
        zPM = probe_data['planned_z'] == probe_data['micro_z']
        probe_data = probe_data[ (xPM & yPM & zPM) == False ].reset_index()

    if PRINT_INFO:
        print(f'Figure 2 {traj}')
        print(f'N_inst: {probe_data.institute.nunique()}, N_sess: {probe_data.eid.nunique()}, '
              f'N_mice: {probe_data.subject.nunique()}, N_cells: NA')

    # use repo-ephys figure style
    figure_style()
    if ax1 is None:
        fig1, ax1 = plt.subplots()
        ax_lw = 0.5
        ax_lc = "grey"
        lw = 0.2
        alp = 0.8
        ms = 0.5
        avg_alp = 0.5
        avg_ms = 3
        mean_alp = 0.7
        mean_ms = 6
        mean_me = 1
    else:
        fig1 = plt.gcf()
        ax_lw = mpl.rcParams['lines.linewidth']
        ax_lc = 'k'
        lw = 0.4
        alp = 0.5
        ms = 1
        avg_alp = 1
        avg_ms = 4
        mean_alp = 1
        mean_ms = 7
        mean_me = 1.5


    # draw 0,0 lines
    ax1.axvline(x=-2243, color=ax_lc, linestyle="--", linewidth=ax_lw)
    ax1.axhline(y=-2000, color=ax_lc, linestyle="--", linewidth=ax_lw)

    # plot each micro-manipulator position on ax1 as line from origin (planned) and dot
    for idx, row in probe_data.iterrows():

        ax1.plot([row[f'{traj}_x'], row['planned_x']], [row[f'{traj}_y'], row['planned_y']],
                 color=institution_colors[institution_map[row['lab']]], 
                 linewidth=lw, alpha=alp)

        ax1.plot(row[f'{traj}_x'], row[f'{traj}_y'], 
                 color=institution_colors[institution_map[row['lab']]],
                 marker="o", markersize=ms, alpha=alp, markeredgewidth=0.5)

    # Plot mean micro coords - institution means - as cross
    inst_mean_x = probe_data.groupby('institute')[f'{traj}_x'].mean()
    inst_mean_y = probe_data.groupby('institute')[f'{traj}_y'].mean()

    for x, y, k in zip(inst_mean_x, inst_mean_y, inst_mean_x.keys()):
        ax1.plot(x, y, 
                 color=institution_colors[k], 
                 label=k, 
                 marker="+", markersize=avg_ms, alpha=avg_alp)

    # Plot overall mean (mean of institutions) - as large cross
    mean_x = probe_data[f'{traj}_x'].mean()
    mean_y = probe_data[f'{traj}_y'].mean()

    ax1.plot(mean_x, mean_y, color='k', marker="+", markersize=mean_ms, markeredgewidth=mean_me, alpha=mean_alp, label="MEAN")

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

    if traj == 'micro':
        ax1.set_xlim((-3200, -1200))
        ax1.set_ylim((-3200, -1200))
    else:
        ax1.set_xlim((-3200, -1200))
        ax1.set_ylim((-3200, -1200))

    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(5))

    if save:
        ax1.tick_params(axis='both', which='major', labelsize=5)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(5))
        # set x/y axis labels
        if traj == 'micro':
            ax1.set_xlabel('micro-manipulator ML displacement (\u03bcm)', fontsize=7)
            ax1.set_ylabel('micro-manipulator AP displacement (\u03bcm)', fontsize=7)
            ax1.set_title('Targeting variability')
            # ax1.set_title('MICRO-MANIPULATOR: Mean (SD) distance \n    '
            #              'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
            #              'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm',
            #              fontsize=8)

            # print average values
            print('MICRO-MANIPULATOR: Mean (SD) distance \n    '
                  'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n')
            #      'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm')
            # add legend
            ax1.legend(loc='upper right', prop={'size': 3.5})
        else:
            ax1.set_xlabel('histology ML displacement (\u03bcm)', fontsize=7)
            ax1.set_ylabel('histology AP displacement (\u03bcm)', fontsize=7)
            ax1.set_title('Geometrical variability')
            # ax1.set_title('HISTOLOGY: Mean (SD) distance \n    '
            #              'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n' +
            #              'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm',
            #              fontsize=8)
            # print average values
            print('HISTOLOGY: Mean (SD) distance \n    '
                  'ALL : ' + str(np.around(top_mean_all, 1)) + ' (' + str(np.around(top_std_all, 2)) + ')' + ' µm \n')
            #      'PASS : ' + str(np.around(top_mean_include, 1)) + ' (' + str(np.around(top_std_include, 2)) + ')' + ' µm')

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

        fig_path = save_figure_path(figure='fig_hist')
        fig1.savefig(fig_path.joinpath(f'D_probe_surf_coord_{traj}_label.svg'), bbox_inches="tight")

    else:
        ax1.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax1.yaxis.set_major_locator(plt.MaxNLocator(4))

        if traj == 'micro':
            ax1.set_xlabel('Micromanipulator ML displacement (\u03bcm)')
            ax1.set_ylabel('Micromanipulator AP displacement (\u03bcm)')
            ax1.set_title('Targeting variability')
            ax1.legend(loc='lower left', prop={'size': 4})
        else:
            ax1.set_xlabel('Histology ML displacement (\u03bcm)')
            ax1.set_ylabel('Histology AP displacement (\u03bcm)')
            ax1.set_title('Geometrical variability')

        ax1.spines[['right', 'top']].set_visible(False)




def plot_probe_distance_all_lab(traj='micro', min_rec_per_lab=4, perform_permutation_test=True, axs=None, save=True):
    '''Plot the DISTANCES from planned to micro displacement, histogram plus
    density plot of ALL distances - to see its distribution shape.
    COMBINED with plot of distances, split by lab
    '''
    
    # Load data
    probe_data = load_dataframe(df_name='traj')

    # manually exclude (for now) unresolved PIDs + one odd?(68d) looking insertion..? 38a1 - NR_0031, micro data far off??
    pids = [
        '2ff92e61-c2af-4dbf-8862-bd50b344762b',
        '8f1d5aad-8c1f-4e81-869a-5a1ab1bf53b2',
        'b53cc868-008a-4d20-a33a-ea3101be2d34',
        'f06d6cd9-a6b8-49a4-90d1-7905d04c2f8b',
        '84fd7fa3-6c2d-4233-b265-46a427d3d68d',
        '4836a465-c691-4852-a0b1-dcd2b1ce38a1']

    for p in pids:
        probe_data = probe_data[ probe_data['pid'] != p]

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

    if PRINT_INFO:
        print(f'Figure 2 g/ h {traj}')
        print(f'N_inst: {probe_data.institute.nunique()}, N_sess: {probe_data.eid.nunique()}, '
              f'N_mice: {probe_data.subject.nunique()}, N_cells: NA')

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
    if axs is None:
        fig, fig_axes = plt.subplots(ncols=1, nrows=2, constrained_layout=True,
            gridspec_kw={'height_ratios': [1, 9]})
        axr0c0 = fig_axes[0]
        axr1c0 = fig_axes[1]
        lw = 0.5
    else:
        axr0c0 = axs[0]
        axr1c0 = axs[1]
        lw = mpl.rcParams['lines.linewidth']

    # Create an array with colors to use
    colors_pts = ["#000000"]  # BLACK

    # Plot ALL kdeplot + boxplot/stripplot
    sns.kdeplot( x=f'{traj}_error_surf_xy', data=probe_data, color='#000000', fill=True, ax=axr0c0)
    # round up to nearest hundred from maximum xy surface error for histoloy
    max_distance = int(math.ceil( max(probe_data[f'{traj}_error_surf_xy']) / 100.0)) * 100
    max_distance = int(math.ceil( max(probe_data['hist_error_surf_xy']) / 100.0)) * 100 # use histology error for both plots
    axr0c0.set_xlim(0, max_distance)
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
    sns.boxplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data, 
                palette=order_colors, orient="h", showfliers=False, linewidth=0.5,
                order=order_by, ax=axr1c0)

    # overlay points
    axx = sns.stripplot(y='institute', x=f'{traj}_error_surf_xy', data=probe_data, 
                        size=1.5,  orient="h", color="#000000", order=order_by, ax=axr1c0)

    # plot overall mean distance (mean of labs)
    mean_error_surf_xy = probe_data[f'{traj}_error_surf_xy'].mean()
    axr1c0.axvline(x=mean_error_surf_xy, linestyle='--', linewidth=lw, color='gray')

    axr1c0.set_ylabel(None)
    axr1c0.set_xlim(0, max_distance)
    axr1c0.set_xlabel(None)
    axr1c0.xaxis.set_major_locator(plt.MaxNLocator(5))

    if save:
        axr0c0.set_ylabel('density', fontsize=6)
        axr0c0.tick_params(axis='both', which='major', labelsize=5)
        axr1c0.tick_params(axis='both', which='major', labelsize=5)
        axr1c0.axvline(x=mean_error_surf_xy, linestyle='--', linewidth=lw, color='gray')

        if traj == 'micro':
            # add legend to micro group plot only
            # handles, labels = axr1c0.get_legend_handles_labels()
            # axr1c0.legend(handles[0:2], labels[0:2], bbox_to_anchor=(0.98, 0.05), loc=4,
            #       borderaxespad=0., prop={'size': 4}, markerscale=0.2)
            fig.suptitle('Micromanipulator to planned distance', fontsize=7)
            # fig.supxlabel('\nMicromanipulator distance (µm)', fontsize=7)
            axr1c0.set_xlabel('Micromanipulator distance (\u03bcm)')
        else:
            fig.suptitle('Histology to planned distance', fontsize=7)
            # fig.supxlabel('\nHistology distance (µm)', fontsize=7)
            axr1c0.set_xlabel('Histology distance (\u03bcm)')

        plt.tight_layout()  # tighten layout around xlabel & ylabel
        fig.set_size_inches(2.15, 2.8)

        fig_path = save_figure_path(figure='fig_hist')
        fig.savefig(fig_path.joinpath(f'D_probe_dist_{traj}_all_lab.svg'), bbox_inches="tight")
    else:
        axr1c0.axvline(x=mean_error_surf_xy, linestyle='--', linewidth=lw, color='k')
        if traj == 'micro':
            axr0c0.set_title('Micromanipulator to planned \n distance')
            axr1c0.set_xlabel('Micromanipulator distance (\u03bcm)')
            axr0c0.set_ylabel('Density')
            axr1c0.set_ylabel('Lab', labelpad=-2)
        else:
            axr0c0.set_title('Histology to planned distance')
            axr1c0.set_xlabel('Histology distance (\u03bcm)')
            axr0c0.set_ylabel('')
            axr1c0.set_ylabel('')

        axr1c0.spines[['right', 'top']].set_visible(False)
        axr0c0.spines[['right', 'top']].set_visible(False)


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

        print("\n\n\nPERMUTATION TEST ALL : ", p_m, "\n\n\n\n")


