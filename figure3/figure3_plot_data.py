from reproducible_ephys_functions import filter_recordings, labs, BRAIN_REGIONS
import pandas as pd
import numpy as np
from features_2D import get_brain_boundaries, plot_probe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from ibllib.atlas.regions import  BrainRegions
from figure3.figure3_load_data import load_dataframe
import seaborn as sns
from permutation_test import permut_test, permut_dist
from statsmodels.stats.multitest import multipletests

br = BrainRegions()

lab_number_map, institution_map, lab_colors = labs()

def panel_probe_lfp(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000],
                    normalize=False, clim=[-190, -150]):

    df_chns = load_dataframe(df_name='chns')

    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['lab_number', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('lab_number').size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])

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

        ax[iR].set_title(data['recording'] + 1,
                         color=lab_colors[data['institute']])
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


def panel_probe_neurons(fig, ax, n_rec_per_lab=4, boundary_align='DG-TH', ylim=[-2000, 2000]):

    df_chns = load_dataframe(df_name='chns')
    df_clust = load_dataframe(df_name='clust')

    df_filt = filter_recordings(min_rec_lab=n_rec_per_lab)
    df_filt = df_filt[df_filt['lab_include'] == 1]
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    df_filt = df_filt.sort_values(by=['lab_number', 'subject']).reset_index(drop=True)
    df_filt = df_filt.drop_duplicates(subset='subject').reset_index()
    rec_per_lab = df_filt.groupby('lab_number').size()
    df_filt['recording'] = np.concatenate([np.arange(i) for i in rec_per_lab.values])
    fig, ax = plt.subplots(1, 32)

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
                            cmap='hot', vmin=levels[0], vmax=levels[1], zorder=1)
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
            ax[iR].bar(x=width/2, height=height, width=width, color=color, bottom=reg[0], edgecolor='w', alpha=0.5)

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
            ax[iR].bar(x=width/2, height=height, width=width, color=color, bottom=reg[0], edgecolor='k', alpha=alpha)

        ax[iR].set_title(data['recording'] + 1,
                         color=lab_colors[data['institute']])

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

    # Add brain regions
    width = ax[-1].get_xlim()[1]
    ax[-1].set(ylim=ylim)
    ax[-1].bar(x=width / 2, height=750, width=width, color=np.array([0, 159, 172]) / 255,
               bottom=1250, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=500, width=width, color=np.array([126, 208, 75]) / 255,
               bottom=650, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=500, width=width, color=np.array([126, 208, 75]) / 255,
               bottom=50, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=900, width=width, color=np.array([255, 144, 159]) / 255,
               bottom=-950, edgecolor='k', linewidth=0)
    ax[-1].bar(x=width / 2, height=950, width=width, color=np.array([255, 144, 159]) / 255,
               bottom=-2000, edgecolor='k', linewidth=0)
    ax[-1].text(width / 2, 1600, 'PPC', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2, 900, 'CA1', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2, 300, 'DG', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2, -500, 'LP', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].text(width / 2, -1500, 'PO', rotation=90, va='center', color='w', fontweight='bold', ha='center')
    ax[-1].set_axis_off()

    # Add colorbar
    axin = inset_axes(ax[-1], width="50%", height="80%", loc='lower right', borderpad=0,
                      bbox_to_anchor=(1, 0.1, 1, 1), bbox_transform=ax[-1].transAxes)
    cbar = fig.colorbar(im, cax=axin, ticks=im.get_clim())
    cbar.ax.set_yticklabels([f'{levels[0]}', f'{levels[1]}'])
    cbar.set_label('Firing rate (spks/s)', rotation=270, labelpad=-2)


def panel_example(ax, n_rec_per_lab=4, example_region='LP', example_metric='lfp_power_high',
                  ylim=[-200, -150]):

    fig, ax = plt.subplots(1, 1)
    df_ins = load_dataframe(df_name='ins')
    df_filt = filter_recordings(df_ins, min_rec_lab=n_rec_per_lab)
    df_filt['lab_number'] = df_filt['lab'].map(lab_number_map)
    data = df_filt[df_filt['permute_include'] == 1]

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

    fig, ax = plt.subplots(1, 1)
    df_ins = load_dataframe(df_name='ins')
    df_filt = filter_recordings(df_ins, min_lab_region=n_rec_per_region, min_rec_lab=n_rec_per_lab)
    data = df_filt[df_filt['permute_include'] == 1]
    data['yield_per_channel'] = data['neuron_yield'] / data['n_channels']

    results = pd.DataFrame()
    for metric in metrics:
        print(metric)
        for region in regions:
            # Select data for this region and metrics
            this_data = data.loc[data['region'] == region, metric].values
            this_labs = data.loc[data['region'] == region, 'institute'].values
            this_subjects = data.loc[data['region'] == region, 'subject'].values
            print(np.sum(np.isnan(this_data)))
            this_labs = this_labs[~np.isnan(this_data)]
            this_subjects = this_subjects[~np.isnan(this_data)]
            this_data = this_data[~np.isnan(this_data)]

            # Exclude data from labs that do not have enough recordings
            lab_names, this_n_labs = np.unique(this_labs, return_counts=True)
            excl_labs = lab_names[this_n_labs < n_rec_per_region]
            print(region)
            print(excl_labs)
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
