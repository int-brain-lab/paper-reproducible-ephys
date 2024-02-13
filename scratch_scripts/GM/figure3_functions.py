#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 08:43:23 2021
By: Guido Meijer & Noam Roth
"""

from one.api import ONE
from os.path import join
from ibllib.ephys.neuropixel import SITES_COORDINATES
from ibllib.pipes.ephys_alignment import EphysAlignment
from iblatlas import atlas
from iblatlas.regions import BrainRegions
from permutation_test import permut_test
from statsmodels.stats.multitest import multipletests
from scratch_scripts.MF.features_2D import (psd_data, get_brain_boundaries, plot_probe, spike_amp_data,
                                            get_brain_boundaries_interest)
from reproducible_ephys_functions import labs, data_path, exclude_recordings
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.sankey import Sankey
import pandas as pd
import numpy as np
import seaborn as sns


def panel_sankey(fig, ax):

    #fig, ax = plt.subplots(1, 1, figsize=(6, 3), dpi=400)
    ax.axis('off')

    #currently hardcoded to match Steven & Guido analyses;
    #todo: finalize numbers and match with above code
    num_trajectories = [92, -7, -16, -21, -7, -16, -32]

    # Sankey plot
    sankey = Sankey(ax=ax, scale=0.0015, offset=0.2, head_angle=90, shoulder=0.025, gap=0.2, radius=0.05)
    sankey.add(flows=num_trajectories,
               labels=['All sessions', 'Histology damage',
                       'Insufficient # recordings',
                       'Noise/yield',
                       'Targeting',
                       'Behavior',
                       'Data analysis'],
               trunklength=0.8,
               orientations=[0, 1, 1, -1, -1,-1, 0],
               pathlengths=[0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.4],
               facecolor = sns.color_palette('Pastel1')[1])
    diagrams = sankey.finish()

    #text font and positioning
    for text in diagrams[0].texts:
            text.set_fontsize('7')


    text = diagrams[0].texts[0]
    xy = text.get_position()
    text.set_position((xy[0] - 0.2, xy[1]))
    text.set_weight('bold')

    text = diagrams[0].texts[-1]
    xy = text.get_position()
    text.set_position((xy[0] + 0.1, xy[1]))
    text.set_weight('bold')

    text = diagrams[0].texts[1]
    xy = text.get_position()
    text.set_position((xy[0], xy[1] - 0.1))

    text = diagrams[0].texts[2]
    xy = text.get_position()
    text.set_position((xy[0] + 0.2, xy[1]-0.2))

    text = diagrams[0].texts[3]
    xy = text.get_position()
    text.set_position((xy[0] + 0.06, xy[1]+0.1))

    text = diagrams[0].texts[4]
    xy = text.get_position()
    text.set_position((xy[0], xy[1]+0.1))

    text = diagrams[0].texts[5]
    xy = text.get_position()
    text.set_position((xy[0], xy[1]+0.1))



def panel_probe_lfp(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150], one=None):
    one = one or ONE()
    brain_atlas = atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]
    data, lab_colors = plots_data(n_rec_per_lab)
    data = data.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = data.groupby('lab_number').size()
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

        # Align plots to boundary between brain regions
        brain_regions = ephysalign.get_brain_locations(xyz_channels)
        boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)
        if boundary_align is not None:
            z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
            z = z - z_subtract
            boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)

        # Get LFP data
        plot_data = psd_data(ephys_path, one, eid, chn_inds, freq_range=[20, 80])

        # Plot
        im = plot_probe(plot_data, z, ax[iR], clim=clim, normalize=normalize,
                        cmap='viridis')
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


def panel_probe_neurons(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000],
                        one=None):
    one = one or ONE()
    brain_atlas = atlas.AllenAtlas(25)
    r = BrainRegions()
    depths = SITES_COORDINATES[:, 1]
    data, lab_colors = plots_data(n_rec_per_lab)
    data = data.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = data.groupby('lab_number').size()
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

        # Center probe plots at specified brain region boundary
        brain_regions = ephysalign.get_brain_locations(xyz_channels)
        boundaries, colours, regions = get_brain_boundaries(brain_regions, z, r)
        z_subtract = boundaries[np.where(np.array(regions) == boundary_align)[0][0] + 1]
        z = z - z_subtract

        # Get brain regions and boundaries
        bound_reg, col_reg, reg_name = get_brain_boundaries_interest(brain_regions, z, r)

        # Get spike amp data
        x, y, c = spike_amp_data(eid, collection, one)
        y = ephysalign.get_channel_locations(feature, track, y / 1e6)[:, 2] * 1e6
        y = y - z_subtract
        levels = [0, 30]
        im = ax[iR].scatter(x, y, c=c, s=1, cmap='hot', vmin=levels[0], vmax=levels[1], zorder=2)
        ax[iR].images.append(im)
        ax[iR].set_xlim(1.3, 3)

        # Add rectangles for non-target brain regions
        width = ax[iR].get_xlim()[1]
        all_boundaries = np.concatenate(([0], np.where(np.diff(brain_regions['id']) > 0)[0],
                                         [brain_regions['id'].shape[0] - 1]))
        for i, bound in enumerate(all_boundaries[:-1]):
            ax[iR].bar(x=width/2, height=z[all_boundaries[i+1]] - z[bound], width=width,
                       bottom=z[all_boundaries[i+1]],
                       color=brain_regions['rgb'][i] / 255,
                       edgecolor='k', linewidth=1, alpha=0.5, zorder=0)

        # Add colored rectangles for brain regions of interest
        for reg, co in zip(bound_reg, col_reg):
            ax[iR].bar(x=width/2, height=np.abs(reg[1] - reg[0]), width=width,
                       color=co / 255, bottom=reg[0],
                       edgecolor='k', linewidth=1, alpha=0.5, zorder=1)

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

    # Add brain regions
    width = ax[-1].get_xlim()[1]
    ax[-1].set(ylim=ylim)
    ax[-1].bar(x=width/2, height=750, width=width, color=np.array([0, 159, 172]) / 255,
               bottom=1250, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width/2, height=500, width=width, color=np.array([126, 208,  75]) / 255,
               bottom=650, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width/2, height=500, width=width, color=np.array([126, 208,  75]) / 255,
               bottom=50, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width/2, height=900, width=width, color=np.array([255, 144, 159]) / 255,
               bottom=-950, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width/2, height=950, width=width, color=np.array([255, 144, 159]) / 255,
               bottom=-2000, edgecolor='k', linewidth=0)
    ax[-1].text(width/2, 1600, 'PPC', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width/2, 900, 'CA1', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width/2, 300, 'DG', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width/2, -500, 'LP', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width/2, -1500, 'PO', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].set_axis_off()

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels([f'{levels[0]}', f'{levels[1]}'])
    cbar.set_label('Firing rate (spks/s)', rotation=270, labelpad=-2)


def panel_example(ax, n_rec_per_lab=4, example_region='LP', example_metric='lfp_power_high',
                  ylim=[-200, -150]):
    data, lab_colors = plots_data(n_rec_per_lab)
    data_example = pd.DataFrame(data={
        'institute': data.loc[data['region'] == example_region, 'institute'],
        'lab_number': data.loc[data['region'] == example_region, 'lab_number'],
        example_metric: data.loc[data['region'] == example_region, example_metric].values})
    data_example = data_example.sort_values('lab_number')
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
    ax.set(ylabel=f'LFP ratio in {example_region}\n(stim/baseline)', xlabel='',
           xlim=[-.5, len(data['institute'].unique()) + .5], ylim=ylim,
           yticks=np.arange(ylim[0], ylim[1]+1, 1))
    ax.set_xticklabels(data_example['institute'].unique(), rotation=30, ha='right')
    sns.despine(trim=True)


def panel_permutation(ax, metrics, regions, labels, n_permut=10000, n_rec_per_lab=4,
                      n_rec_per_region=3):
    data, lab_colors = plots_data(n_rec_per_lab)
    results = pd.DataFrame()
    for metric in metrics:
        for region in regions:
            # Select data for this region and metrics
            this_data = data.loc[data['region'] == region, metric].values
            this_labs = data.loc[data['region'] == region, 'institute'].values
            this_subjects = data.loc[data['region'] == region, 'subject'].values
            this_labs = this_labs[~np.isnan(this_data)]
            this_subjects = this_subjects[~np.isnan(this_data)]
            this_data = this_data[~np.isnan(this_data)]

            # Exclude data from labs that do not have enough recordings
            lab_names, this_n_labs = np.unique(this_labs, return_counts=True)
            excl_labs = lab_names[this_n_labs < n_rec_per_region]
            this_data = this_data[~np.isin(this_labs, excl_labs)]
            this_subjects = this_subjects[~np.isin(this_labs, excl_labs)]
            this_labs = this_labs[~np.isin(this_labs, excl_labs)]

            # Do permutation test
            p = permut_test(this_data, metric=permut_dist, labels1=this_labs,
                            labels2=this_subjects, n_permut=n_permut)
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
    sns.heatmap(results_plot, cmap='RdYlGn', square=True,
                cbar=True, cbar_ax=axin,
                annot=False, annot_kws={"size": 5},
                linewidths=.5, fmt='.2f', vmin=-1.5, vmax=np.log10(1), ax=ax)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log10([0.05, 0.1, 0.2, 0.4, 0.8]))
    cbar.set_ticklabels([0.05, 0.1, 0.2, 0.4, 0.8])
    cbar.set_label('log p-value', rotation=270, labelpad=8)
    ax.set(xlabel='', ylabel='')
    ax.set_yticklabels(regions, va='center', rotation=0)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    return results


def plots_data(n_rec_per_lab=4):
    # Load in data
    data = pd.read_csv(join(data_path(), 'metrics_region.csv'))
    lfp = pd.read_csv(join(data_path(), 'lfp_ratio_per_region.csv'))

    # Exclude recordings
    data = exclude_recordings(data)

    # Merge LFP ratio data with the rest
    data = data.merge(lfp, on=['subject', 'region'])

    # Reformat data
    lab_number_map, institution_map, lab_colors = labs()
    data['institute'] = data.lab.map(institution_map)
    data['lab_number'] = data.lab.map(lab_number_map)
    data = data.groupby('institute').filter(
        lambda s : s['eid'].unique().shape[0] >= n_rec_per_lab)
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



