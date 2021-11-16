#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:43:23 2021
By: Guido Meijer
"""

from one.api import ONE
from os.path import join
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from ibllib.atlas import atlas, BrainRegions
from permutation_test import permut_test
from statsmodels.stats.multitest import multipletests
from features_2D import (psd_data, get_brain_boundaries, plot_probe, spike_amp_data,
                         get_brain_boundaries_interest)
from reproducible_ephys_functions import labs, data_path, exclude_recordings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
import numpy as np
import seaborn as sns


def panel_a(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000], one=None):
    one = one or ONE()
    brain_atlas = atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]
    data, lab_colors = plots_data(n_rec_per_lab)
    data = data.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = data.groupby('institute').size()
    data['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

    for iR, (subj, date, probe_label) in enumerate(zip(data['subject'], data['date'], data['probe'])):

        # Download the data and get paths to downloaded data
        eid = one.search(subject=subj, date=date)[0]
        if iR == 0:
            chn_inds = one.load_dataset(eid, dataset=['channels.rawInd.npy'],
                                        collection=f'alf/{probe_label}')
        ephys_path = one.eid2path(eid).joinpath('raw_ephys_data', probe_label)

        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
        align_key = insertion[0].get('json').get('extended_qc').get('alignment_stored', None)
        trajectory = one.alyx.rest('trajectories', 'list',
                                   provenance='Ephys aligned histology track',
                                   probe_insertion=insertion[0]['id'])
        alignments = trajectory[0]['json']
        feature = np.array(alignments[align_key][0])
        track = np.array(alignments[align_key][1])

        # Instantiate EphysAlignment object
        ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                                    feature_prev=feature,
                                    brain_atlas=brain_atlas)
        xyz_channels = ephysalign.get_channel_locations(feature, track)
        z = xyz_channels[:, 2] * 1e6
        brain_regions = ephysalign.get_brain_locations(xyz_channels)

        boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract
            boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

        # Get LFP data
        plot_data = psd_data(ephys_path, one, eid, chn_inds, freq_range=[20, 80])
        im = plot_probe(plot_data, z, ax[iR], cmap='viridis')

        ax[iR].set_title(data.loc[iR, 'recording'] + 1,
                         color=lab_colors[data.loc[iR, 'institute']])

        if iR == 0:
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1]+1, 500),
                       yticklabels=np.arange(ylim[0], ylim[1]+1, 500) / 1000,
                       xticks=[])
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels(['10th\nperc.', '90th\nperc'])
    cbar.set_label('Power spectral density', rotation=270, labelpad=-8)


def panel_b(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000], one=None):
    one = one or ONE()
    brain_atlas = atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]
    data, lab_colors = plots_data(n_rec_per_lab)
    data = data.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = data.groupby('institute').size()
    data['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

    for iR, (subj, date, probe_label) in enumerate(zip(data['subject'], data['date'], data['probe'])):

        # Download the data and get paths to downloaded data
        eid = one.search(subject=subj, date=date)[0]
        print(f'Recording {iR+1} of {data.shape[0]}')

        collections = one.list_collections(eid)
        if f'alf/{probe_label}/pykilosort' in collections:
            collection = f'alf/{probe_label}/pykilosort'
        else:
            collection = f'alf/{probe_label}'
        #collection = f'alf/{probe_label}'

        insertion = one.alyx.rest('insertions', 'list', session=eid, name=probe_label)
        xyz_picks = np.array(insertion[0].get('json').get('xyz_picks', 0)) / 1e6
        align_key = insertion[0].get('json').get('extended_qc').get('alignment_stored', None)
        trajectory = one.alyx.rest('trajectories', 'list',
                                   provenance='Ephys aligned histology track',
                                   probe_insertion=insertion[0]['id'])
        alignments = trajectory[0]['json']
        feature = np.array(alignments[align_key][0])
        track = np.array(alignments[align_key][1])

        # Instantiate EphysAlignment object
        ephysalign = EphysAlignment(xyz_picks, depths, track_prev=track,
                                    feature_prev=feature,
                                    brain_atlas=brain_atlas)
        xyz_channels = ephysalign.get_channel_locations(feature, track)
        z = xyz_channels[:, 2] * 1e6
        brain_regions = ephysalign.get_brain_locations(xyz_channels)
        boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)
        bound_reg, col_reg, reg_name = get_brain_boundaries_interest(brain_regions, z, r)

        z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
        z = z - z_subtract
        boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)
        bound_reg, col_reg, reg_name = get_brain_boundaries_interest(brain_regions, z, r)

        # Get spike amp data
        x, y, c = spike_amp_data(eid, collection, one)
        y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
        y = y - z_subtract
        levels = [0, 30]
        im = ax[iR].scatter(x, y, c=c, s=1, cmap='hot', vmin=levels[0], vmax=levels[1])
        ax[iR].images.append(im)
        ax[iR].set_xlim(1.3, 3)

        for reg, co, lab in zip(bound_reg, col_reg, reg_name):
            height = np.abs(reg[1] - reg[0])
            color = co / 255
            width = ax[iR].get_xlim()[1]
            ax[iR].bar(x=width/2, height=height, width=width, color=color, bottom=reg[0],
                       edgecolor='k', linewidth=1, alpha=0.5)

        ax[iR].set_title(data.loc[iR, 'recording'] + 1,
                         color=lab_colors[data.loc[iR, 'institute']])

        if iR == 0:
            ax[iR].set(yticks=np.arange(ylim[0], ylim[1]+1, 500),
                       yticklabels=np.arange(ylim[0], ylim[1]+1, 500) / 1000,
                       xticks=[])
            ax[iR].tick_params(axis='y')
            ax[iR].spines["right"].set_visible(False)
            ax[iR].spines["bottom"].set_visible(False)
            ax[iR].spines["top"].set_visible(False)
            ax[iR].set_ylabel('Depth relative to DG-Thalamus (mm)')
        else:
            ax[iR].set_axis_off()
        ax[iR].set(ylim=ylim)

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels([f'{levels[0]} Hz', f'{levels[1]} Hz'])


def panel_c(ax, n_rec_per_lab=4, example_region='LP', example_metric='lfp_power_high', ylim=[-190, -120]):
    data, lab_colors = plots_data(n_rec_per_lab)
    data_example = pd.DataFrame(data={
        'institute': data.loc[data['region'] == example_region, 'institute'],
        example_metric: data.loc[data['region'] == example_region, example_metric].values})
    data_example = data_example.sort_values('institute')
    cmap = []
    for i, inst in enumerate(data_example['institute'].unique()):
        cmap.append(lab_colors[inst])

    sns.stripplot(data=data_example, x='institute', y=example_metric, palette=cmap, s=3, ax=ax)
    ax_lines = sns.pointplot(x='institute', y=example_metric, data=data_example,
                             ci=0, join=False, estimator=np.mean, color='k',
                             markers="_", scale=1, ax=ax)
    #plt.setp(ax_lines.collections, zorder=100, label="")
    ax.plot(np.arange(data_example['institute'].unique().shape[0]),
             [data_example[example_metric].mean()] * data_example['institute'].unique().shape[0],
             color='r', lw=1)
    ax.set(ylabel=f'LFP power in {example_region} (dB)', xlabel='',
           xlim=[-.5, len(data['institute'].unique()) + .5], ylim=ylim)
    ax.set_xticklabels(data_example['institute'].unique(), rotation=30, ha='right')
    sns.despine(trim=True)


def panel_d(ax, metrics, regions, labels, n_permut=10000, n_rec_per_lab=4):
    data, lab_colors = plots_data(n_rec_per_lab)
    results = pd.DataFrame()
    for metric in metrics:
        for region in regions:
            this_data = data.loc[data['region'] == region, metric].values
            p = permut_test(
                    this_data[~np.isnan(this_data)],
                    metric=permut_dist,
                    labels1=data.loc[data['region'] == region, 'institute'].values[~np.isnan(this_data)],
                    labels2=data.loc[data['region'] == region, 'subject'].values[~np.isnan(this_data)],
                    n_permut=n_permut)
            results = results.append(pd.DataFrame(index=[results.shape[0]+1], data={
                'metric': metric, 'region': region, 'p_value_permut': p}))

    for i, region in enumerate(regions):
        results.loc[results['region'] == region, 'region_number'] = i

    # Perform correction for multiple testing
    _, results['p_value_permut'], _, _ = multipletests(results['p_value_permut'], 0.05, method='fdr_bh')

    results_plot = results.pivot(index='region_number', columns='metric', values='p_value_permut')
    results_plot = results_plot.reindex(columns=metrics)
    results_plot = np.log10(results_plot)

    axin = inset_axes(ax, width="5%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)
    #cmap = sns.color_palette('viridis_r', n_colors=20)
    #cmap[0] = [1, 0, 0]
    sns.heatmap(results_plot, cmap='viridis_r', square=True,
                cbar=True, cbar_ax=axin,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-1.5, vmax=np.log10(0.5), ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.05, 0.5, 1]))
    cbar.set_ticklabels([0.05, 0.5, 1])
    ax.set(xlabel='', ylabel='', title='Permutation p-values')
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=30, ha='right')


def plots_data(n_rec_per_lab=4):
    data = pd.read_csv(join(data_path(), 'metrics_region.csv'))
    data = exclude_recordings(data, destriped_rms=False)
    lab_number_map, institution_map, lab_colors = labs()
    data['institute'] = data.lab.map(institution_map)
    data = data.groupby('institute').filter(
        lambda s : s['eid'].unique().shape[0] >= n_rec_per_lab)
    data = data.sort_values(by=['institute', 'subject']).reset_index(drop=True)
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



