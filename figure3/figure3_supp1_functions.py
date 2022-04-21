#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 13:41:49 2021
By: Guido Meijer
"""

from os.path import join

from ibllib.atlas import atlas, BrainRegions
from permutation_test import permut_test
from statsmodels.stats.multitest import multipletests
from figure3.figure3_functions import plot_probe, get_brain_boundaries
from reproducible_ephys_functions import labs, data_path, filter_recordings, BRAIN_REGIONS
from figure3.figure3_load_data import load_dataframe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
import seaborn as sns


def panel_a(fig, ax, incl_labs, boundary_align='DG-TH', ylim=[-2000, 2000], normalize=True, clim=(0.1, 0.9)):

    df_chns = load_dataframe(df_name='chns')

    df_filt = filter_recordings(min_neuron_region=0)
    df_filt = df_filt.sort_values(by=['include'], ascending=False).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()

    for iR, data in df_filt.iterrows():
        df = df_chns[df_chns['pid'] == data['pid']]

        la = {}
        la['id'] = df['region_id'].values
        z = df['z'].values * 1e6
        boundaries, colours, regions = get_brain_boundaries(la, z)
        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract

        # Plot
        im = plot_probe(df['lfp'].values, z, ax[iR], clim=clim, normalize=normalize,
                        cmap='viridis')

        if data['include'] == True:
            ax[iR].set_title(data.subject, rotation=45, ha='left', color='green', fontsize=4)
        else:
            ax[iR].set_title(data.subject, rotation=45, ha='left', color='red', fontsize=4)

        if iR == 0:
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1] + 1, 500),
                       yticklabels=np.arange(ylim[0], ylim[1] + 1, 500) / 1000,
                       xticks=[])
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)
    ax[-1].set_axis_off()

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="90%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    if normalize:
        cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])
    else:
        cbar.ax.set_yticklabels([f'{clim[0]}', f'{clim[1]}'])
    cbar.set_label('Power spectral density (dB)', rotation=270, labelpad=-5)


def panel_b(fig, ax, incl_labs=None, boundary_align='DG-TH', ylim=[-2000, 2000], one=None):
    br = BrainRegions()

    df_chns = load_dataframe(df_name='chns')
    df_clust = load_dataframe(df_name='clust')

    df_filt = filter_recordings(min_neuron_region=0)
    df_filt = df_filt.sort_values(by=['include'], ascending=False).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()

    for iR, data in df_filt.iterrows():
        df_ch = df_chns[df_chns['pid'] == data['pid']]
        df_clu = df_clust[df_clust['pid'] == data['pid']]

        la = {}
        la['id'] = df_ch['region_id'].values
        z = df_ch['z'].values * 1e6
        boundaries, colours, regions = get_brain_boundaries(la, z)
        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract
        else:
            z_subtract = 0

        levels = [0, 30]
        im = ax[iR].scatter(np.log10(df_clu['amps'] * 1e6), df_clu['depths_aligned'] - z_subtract, c=df_clu['fr'], s=1,
                            cmap='hot', vmin=levels[0], vmax=levels[1], zorder=2)
        ax[iR].images.append(im)
        ax[iR].set_xlim(1.3, 3)

        # First for all regions
        region_info = br.get(df_ch['region_id'].values)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
        regions = z[np.c_[boundaries[0:-1], boundaries[1:]]]
        region_colours = region_info.rgb[boundaries[1:]]

        width = ax[iR].get_xlim()[1]
        for reg, col in zip(regions, region_colours):
            height = np.abs(reg[1] - reg[0])
            color = col / 255
            ax[iR].bar(x=width / 2, height=height, width=width, color=color, bottom=reg[0],
                       edgecolor='w', alpha=0.5, zorder=0)

        # Now for rep site
        region_info = br.get(df_ch['region_id_rep'].values)
        boundaries = np.where(np.diff(region_info.id) != 0)[0]
        boundaries = np.r_[0, boundaries, region_info.id.shape[0] - 1]
        regions = z[np.c_[boundaries[0:-1], boundaries[1:]]]
        region_labels = np.c_[np.mean(regions, axis=1), region_info.acronym[boundaries[1:]]]
        region_labels[region_labels[:, 1] == 'VISa', 1] = 'PPC'
        region_colours = region_info.rgb[boundaries[1:]]
        reg_idx = np.where(np.isin(region_labels[:, 1], BRAIN_REGIONS))[0]
        #
        for i, (reg, col, lab) in enumerate(zip(regions, region_colours, region_labels)):
            height = np.abs(reg[1] - reg[0])
            color = col / 255
            if np.isin(i, reg_idx):
                alpha = 1
            else:
                alpha = 0
            ax[iR].bar(x=width / 2, height=height, width=width, color=color, bottom=reg[0],
                       edgecolor='k', alpha=alpha, zorder=1)

        if data['include'] == True:
            ax[iR].set_title(data.subject, rotation=45, ha='left', color='green', fontsize=4)
        else:
            ax[iR].set_title(data.subject, rotation=45, ha='left', color='red', fontsize=4)

        if iR == 0:
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1] + 1, 500),
                       yticklabels=np.arange(ylim[0], ylim[1] + 1, 500) / 1000,
                       xticks=[])
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)

    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels([f'{levels[0]}', f'{levels[1]}'])
    cbar.set_label('Firing rate (spks/s)', rotation=270, labelpad=0)


def panel_c(ax, metrics, regions, labels, n_permut, incl_labs):
    data, lab_colors = plots_data(incl_labs)
    data_excl = exclude_recordings(data)
    results = pd.DataFrame()
    for metric in metrics:
        for region in regions:
            this_data = data.loc[data['region'] == region, metric].values
            p_all = permut_test(
                    this_data[~np.isnan(this_data)],
                    metric=permut_dist,
                    labels1=data.loc[data['region'] == region, 'institute'].values[~np.isnan(this_data)],
                    labels2=data.loc[data['region'] == region, 'subject'].values[~np.isnan(this_data)],
                    n_permut=n_permut)
            this_data = data_excl.loc[data['region'] == region, metric].values
            p_excl = permut_test(
                    this_data[~np.isnan(this_data)],
                    metric=permut_dist,
                    labels1=data_excl.loc[data['region'] == region, 'institute'].values[~np.isnan(this_data)],
                    labels2=data_excl.loc[data['region'] == region, 'subject'].values[~np.isnan(this_data)],
                    n_permut=n_permut)
            results = results.append(pd.DataFrame(index=[results.shape[0]+1], data={
                'metric': metric, 'region': region, 'p_all': p_all, 'p_excl': p_excl}))

    for i, region in enumerate(regions):
        results.loc[results['region'] == region, 'region_number'] = i

    # Perform correction for multiple testing
    _, results['p_all'], _, _ = multipletests(results['p_all'], 0.05, method='fdr_bh')
    _, results['p_excl'], _, _ = multipletests(results['p_excl'], 0.05, method='fdr_bh')

    results_all = results.pivot(index='region_number', columns='metric', values='p_all')
    results_excl = results.pivot(index='region_number', columns='metric', values='p_excl')
    #results_plot = results_excl.copy()
    results_plot = results_excl - results_all
    results_plot = results_plot.reindex(columns=metrics)
    #results_plot = np.log10(results_plot)

    axin = inset_axes(ax, width="5%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
    #cmap = sns.color_palette('viridis_r', n_colors=20)
    #cmap[0] = [1, 0, 0]
    sns.heatmap(results_plot, cmap='coolwarm', square=True,
                cbar=True, cbar_ax=axin, center=0,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-0.2, vmax=0.2, ax=ax)
    cbar = ax.collections[0].colorbar
    #cbar.set_ticks(np.log10([0.05, 0.5, 1]))
    #cbar.set_ticklabels([0.05, 0.5, 1])
    ax.set(xlabel='', ylabel='', title='Permutation p-values')
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=30, ha='right')


def plots_data(incl_labs=None):
    data = pd.read_csv(join(data_path(), 'metrics_region_all.csv'))
    lab_number_map, institution_map, lab_colors = labs()
    data['institute'] = data.lab.map(institution_map)
    data['lab_number'] = data.lab.map(lab_number_map)
    if incl_labs is not None:
        data = data[data['institute'].isin(incl_labs)]
    data = data.sort_values(by=['lab_number', 'subject']).reset_index(drop=True)
    data['lab_position'] = np.linspace(0.18, 0.9, data.shape[0])
    data['in_recording'] = data['neuron_yield'].isnull() == False
    data['yield_per_channel'] = data['neuron_yield'] / data['n_channels']
    data.loc[data['lfp_power_high'] < -100000, 'lfp_power_high'] = np.nan
    return data, lab_colors


def permut_dist(data, labs, mice):
    lab_means = []
    for lab in np.unique(labs):
        lab_means.append(np.mean(data[labs == lab]))
    lab_means = np.array(lab_means)
    return np.sum(np.abs(lab_means - np.mean(lab_means)))