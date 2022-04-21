# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 15:56:43 2021
By: Guido Meijer
"""


import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
import brainbox.io.one as bbone
from scipy.optimize import curve_fit
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import (query, data_path, combine_regions, figure_style,
                                          exclude_recordings)

# Settings
CLUSTERING = 'gaussian'  # gaussian or k-means
MIN_SPIKE_AMP = 0.07
MIN_FR = 0.05
MIN_WAVEFORMS = 10
REGIONS = 'PO', 'LP', 'DG', 'CA1', 'PPC'
N_TYPES = {'PO': 2, 'LP': 2, 'CA1': 3, 'PPC': 3, 'DG': 2}
#FEATURES = ['spike_amp', 'pt_ratio', 'rp_slope', 'spike_width', 'firing_rate', 'peak_to_trough',
#            'spread', 'v_below', 'v_above']
FEATURES_1D = ['spike_amp', 'spike_width', 'pt_ratio', 'rp_slope', 'rc_slope']
FEATURES_2D = ['spike_amp', 'spike_width', 'pt_ratio', 'rp_slope', 'rc_slope',
               'spread', 'v_below', 'v_above']
DATA_DIR = data_path()
FIG_PATH = join(FIG_PATH, 'waveform_clustering')

# Load in waveform data
waveforms_df = pd.read_pickle(join(DATA_DIR, 'waveform_metrics.p'))
rec_data = pd.read_csv(join(data_path(), 'metrics_region.csv'))
rec_data = exclude_recordings(rec_data)
waveforms_df = waveforms_df[waveforms_df['eid'].isin(rec_data['eid'])]

# Apply selection criteria
waveforms_df = waveforms_df[((waveforms_df['spike_amp'] >= MIN_SPIKE_AMP)
                             & (waveforms_df['n_waveforms'] >= MIN_WAVEFORMS)
                             & (waveforms_df['firing_rate'] >= MIN_FR))]
waveforms_df = waveforms_df.reset_index(drop=True)

# Put data in large 2D array
features_1d = waveforms_df[FEATURES_1D].to_numpy()
features_2d = waveforms_df[FEATURES_2D].to_numpy()

# Do t-SNE embedding for visualization
tsne_embedded_1d = TSNE(n_components=2, perplexity=30, random_state=55).fit_transform(features_1d)
tsne_embedded_2d = TSNE(n_components=2, perplexity=30, random_state=55).fit_transform(features_2d)

# Get 2D waveforms
waveforms_df['waveform'] = np.nan
waveforms_df['waveform'] = waveforms_df['waveform'].astype(object)
for i in waveforms_df.index.values:
    waveforms_df.at[i, 'waveform'] = waveforms_df.loc[i, 'waveform_2D'].mean(axis=0)


# %% Plot all regions
colors = figure_style(return_colors=True)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
for r, region in enumerate(REGIONS):
    ax1.scatter(tsne_embedded_1d[:, 0][waveforms_df['region'] == region],
                tsne_embedded_1d[:, 1][waveforms_df['region'] == region], color=colors[region],
                label=region)
    ax2.scatter(tsne_embedded_2d[:, 0][waveforms_df['region'] == region],
                tsne_embedded_2d[:, 1][waveforms_df['region'] == region], color=colors[region],
                label=region)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.set(title='1D waveform features')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set(title='1 + 2D waveform features')
ax2.legend(frameon=False)
plt.savefig(join(FIG_PATH, 't-SNE_embedding_all_regions'))

# %% Clustering per region

time_ax = np.linspace(0, (waveforms_df['waveform_2D'][1].shape[1]/30000)*1000,
                          waveforms_df['waveform_2D'][1].shape[1])

for i, region in enumerate(REGIONS):
    print(f'Processing {region}')


    if N_TYPES[region] == 2:

        if CLUSTERING == 'k-means':
            # K-means clustering
            kmeans = KMeans(n_clusters=N_TYPES[region], random_state=42, n_init=100).fit(
                waveforms_df.loc[waveforms_df['region'] == region, FEATURES_1D].to_numpy())
            waveforms_df.loc[waveforms_df['region'] == region, 'group_label'] = kmeans.labels_
        elif CLUSTERING == 'gaussian':
            # Mixture of Gaussians clustering
            gauss_mix = GaussianMixture(n_components=N_TYPES[region], random_state=42).fit(
                waveforms_df.loc[waveforms_df['region'] == region, FEATURES_1D].to_numpy())
            waveforms_df.loc[waveforms_df['region'] == region, 'group_label'] = gauss_mix.predict(
                waveforms_df.loc[waveforms_df['region'] == region, FEATURES_1D].to_numpy())

        # Get the RS and FS labels right
        fs_label = waveforms_df[waveforms_df['region'] == region].groupby('group_label').mean()['firing_rate'].idxmax()
        waveforms_df.loc[(waveforms_df['region'] == region) & (waveforms_df['group_label'] == fs_label), 'type'] = 'FS'

        rs_label = waveforms_df[waveforms_df['region'] == region].groupby('group_label').mean()['firing_rate'].idxmin()
        waveforms_df.loc[(waveforms_df['region'] == region) & (waveforms_df['group_label'] == rs_label), 'type'] = 'RS'
        region_wfs_df = waveforms_df[waveforms_df['region'] == region]

        # Plot clustering
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3.5), dpi=300)
        ax1.plot(time_ax, region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'waveform'].to_numpy().mean(),
                 color=colors['RS'], label='RS')
        ax1.plot(time_ax, region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'waveform'].to_numpy().mean(),
                 color=colors['FS'], label='FS')
        ax1.legend(frameon=False)
        ax1.set(ylabel='mV', xlabel='Time (ms)')

        ax2.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'rp_slope'], label='RS', color=colors['RS'], s=1)
        ax2.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'rp_slope'], label='FS', color=colors['FS'], s=1)
        ax2.set(xlabel='Spike width (ms)', ylabel='Repolarization slope')

        ax3.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'firing_rate'], label='RS', color=colors['RS'], s=1)
        ax3.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'firing_rate'], label='FS', color=colors['FS'], s=1)
        ax3.set(xlabel='Spike width (ms)', ylabel='Firing rate (spks/s)')

        ax4.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'pt_ratio'], label='RS', color=colors['RS'], s=1)
        ax4.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'pt_ratio'], label='FS', color=colors['FS'], s=1)
        ax4.set(xlabel='Spike width (ms)', ylabel='Peak-to-trough ratio')

        ax5.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'rc_slope'], label='RS', color=colors['RS'], s=1)
        ax5.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'rc_slope'], label='FS', color=colors['FS'], s=1)
        ax5.set(xlabel='Spike width (ms)', ylabel='Recovery slope')

        ax6.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS', 'spike_amp'], label='RS', color=colors['RS'], s=1)
        ax6.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_amp'], label='FS', color=colors['FS'], s=1)
        ax6.set(xlabel='Spike width (ms)', ylabel='Spike amplitude (uV)')

        plt.tight_layout()
        sns.despine(trim=False)
        plt.savefig(join(FIG_PATH, f'{region}_2_types'), dpi=300)

    elif N_TYPES[region] == 3:

        if CLUSTERING == 'k-means':
            # K-means clustering
            kmeans = KMeans(n_clusters=N_TYPES[region], random_state=42, n_init=100).fit(
                waveforms_df.loc[waveforms_df['region'] == region, FEATURES_2D].to_numpy())
            waveforms_df.loc[waveforms_df['region'] == region, 'group_label'] = kmeans.labels_
        elif CLUSTERING == 'gaussian':
            # Mixture of Gaussians clustering
            gauss_mix = GaussianMixture(n_components=N_TYPES[region]).fit(
                waveforms_df.loc[waveforms_df['region'] == region, FEATURES_2D].to_numpy())
            waveforms_df.loc[waveforms_df['region'] == region, 'group_label'] = gauss_mix.predict(
                waveforms_df.loc[waveforms_df['region'] == region, FEATURES_2D].to_numpy())

        # Get the RS and FS labels right
        fs_label = waveforms_df[waveforms_df['region'] == region].groupby('group_label').mean()['firing_rate'].idxmax()
        waveforms_df.loc[(waveforms_df['region'] == region) & (waveforms_df['group_label'] == fs_label), 'type'] = 'FS'

        rs2_label = waveforms_df[waveforms_df['region'] == region].groupby('group_label').mean()['v_below'].idxmin()
        types = np.array([0, 1, 2])
        rs1_label = types[~np.isin(types, np.array([fs_label, rs2_label]))][0]
        if rs2_label == fs_label:
            rs2_label = types[~np.isin(types, np.array([fs_label, rs1_label]))][0]
        waveforms_df.loc[(waveforms_df['region'] == region) & (waveforms_df['group_label'] == rs1_label), 'type'] = 'RS1'
        waveforms_df.loc[(waveforms_df['region'] == region) & (waveforms_df['group_label'] == rs2_label), 'type'] = 'RS2'
        region_wfs_df = waveforms_df[waveforms_df['region'] == region]

        # Plot clustering
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(6, 3.5), dpi=300)
        ax1.plot(time_ax, region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'waveform'].to_numpy().mean(),
                 color=colors['RS1'], label='RS1')
        ax1.plot(time_ax, region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'waveform'].to_numpy().mean(),
                 color=colors['RS2'], label='RS2')
        ax1.plot(time_ax, region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'waveform'].to_numpy().mean(),
                 color=colors['FS'], label='FS')
        ax1.legend(frameon=False)
        ax1.set(ylabel='mV', xlabel='Time (ms)')

        ax2.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'rp_slope'], label='RS1', color=colors['RS1'], s=1)
        ax2.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'rp_slope'], label='RS2', color=colors['RS2'], s=1)
        ax2.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'rp_slope'], label='FS', color=colors['FS'], s=1)
        ax2.set(xlabel='Spike width (ms)', ylabel='Repolarization slope')

        ax3.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'firing_rate'], label='RS1', color=colors['RS1'], s=1)
        ax3.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'firing_rate'], label='RS2', color=colors['RS2'], s=1)
        ax3.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'firing_rate'], label='FS', color=colors['FS'], s=1)
        ax3.set(xlabel='Spike width (ms)', ylabel='Firing rate (spks/s)')

        ax4.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'pt_ratio'], label='RS1', color=colors['RS1'], s=1)
        ax4.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'pt_ratio'], label='RS2', color=colors['RS2'], s=1)
        ax4.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'pt_ratio'], label='FS', color=colors['FS'], s=1)
        ax4.set(xlabel='Spike width (ms)', ylabel='Peak-to-trough ratio')

        ax5.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'rc_slope'], label='RS1', color=colors['RS1'], s=1)
        ax5.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'rc_slope'], label='RS2', color=colors['RS2'], s=1)
        ax5.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'rc_slope'], label='FS', color=colors['FS'], s=1)
        ax5.set(xlabel='Spike width (ms)', ylabel='Recovery slope')

        ax6.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS1', 'spike_amp'], label='RS1', color=colors['RS1'], s=1)
        ax6.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'RS2', 'spike_amp'], label='RS2', color=colors['RS2'], s=1)
        ax6.scatter(region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_width'],
                    region_wfs_df.loc[region_wfs_df['type'] == 'FS', 'spike_amp'], label='FS', color=colors['FS'], s=1)
        ax6.set(xlabel='Spike width (ms)', ylabel='Spike amplitude (uV)')

        plt.tight_layout()
        sns.despine(trim=False)
        plt.savefig(join(FIG_PATH, f'{region}_2_types'), dpi=300)


        d_len = region_wfs_df['dist_soma'].apply(lambda x: len(x)).max()
        t_len = region_wfs_df['waveform_2D'].apply(lambda x: x.shape[1]).max()
        t_x = np.linspace(0, (t_len / 30000) * 1000, t_len)
        dist_soma = np.round(np.linspace(waveforms_df['dist_soma'].apply(lambda x: np.min(x)).min(),
                                         waveforms_df['dist_soma'].apply(lambda x: np.max(x)).max(),
                                         d_len), 2)

        waveforms_1, size_1 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
        prop_1 = np.full(d_len, np.nan)
        for i in region_wfs_df.loc[region_wfs_df['type'] == 'FS'].index:
            waveforms_1[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = (
                waveforms_1[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])]
                + region_wfs_df.loc[i, 'waveform_2D'])
            size_1[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = size_1[
                np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] + 1
            this_prop = np.full(d_len, np.nan)
            this_prop[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = t_x[
                region_wfs_df.loc[i, 'waveform_2D'].argmin(axis=1)]
            prop_1 = np.vstack((prop_1, this_prop))
        waveforms_1 = waveforms_1 / size_1

        waveforms_2, size_2 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
        prop_2 = np.full(d_len, np.nan)
        for i in region_wfs_df.loc[region_wfs_df['type'] == 'RS1'].index:
            waveforms_2[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = (
                waveforms_2[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])]
                + region_wfs_df.loc[i, 'waveform_2D'])
            size_2[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = size_2[
                np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] + 1
            this_prop = np.full(d_len, np.nan)
            this_prop[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = t_x[
                region_wfs_df.loc[i, 'waveform_2D'].argmin(axis=1)]
            prop_2 = np.vstack((prop_2, this_prop))
        waveforms_2 = waveforms_2 / size_2

        waveforms_3, size_3 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
        prop_3 = np.full(d_len, np.nan)
        for i in region_wfs_df.loc[region_wfs_df['type'] == 'RS2'].index:
            waveforms_3[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = (
                waveforms_3[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])]
                + region_wfs_df.loc[i, 'waveform_2D'])
            size_3[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = size_3[
                np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] + 1
            this_prop = np.full(d_len, np.nan)
            this_prop[np.in1d(dist_soma, region_wfs_df.loc[i, 'dist_soma'])] = t_x[
                region_wfs_df.loc[i, 'waveform_2D'].argmin(axis=1)]
            prop_3 = np.vstack((prop_3, this_prop))
        waveforms_3 = waveforms_3 / size_3

        figure_style()
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5), dpi=300)
        ax3.imshow(np.flipud(waveforms_3), cmap='Greys_r', aspect='auto',
                   vmin=-np.max(waveforms_1), vmax=np.max(waveforms_1))
        ax3.get_xaxis().set_visible(False)
        ax3.get_yaxis().set_visible(False)
        ax3.set(title='RS2', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))])

        ax2.imshow(np.flipud(waveforms_2), cmap='Greys_r', aspect='auto',
                   vmin=-np.max(waveforms_2), vmax=np.max(waveforms_2))
        ax2.get_xaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)
        ax2.set(title='RS1', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))])

        ax1.imshow(np.flipud(waveforms_1), cmap='Greys_r', aspect='auto',
                   vmin=-np.max(waveforms_3), vmax=np.max(waveforms_3))
        ax1.get_xaxis().set_visible(False)
        ax1.set(title='FS', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))],
                yticks=np.linspace(0, 10, 5), yticklabels=np.round(np.linspace(-.1, .1, 5), 2),
                ylabel='Distance to soma (um)')

        for i in region_wfs_df.loc[region_wfs_df['type'] == 'RS2'].index:
            ax6.plot(t_x[region_wfs_df.loc[i, 'waveform_2D'].argmin(axis=1)],
                     region_wfs_df.loc[i, 'dist_soma'], color=[.7, .7, .7], alpha=0.2)
        ax6.errorbar(np.nanmedian(prop_3, axis=0), dist_soma,
                     xerr=np.nanstd(prop_3, axis=0)/np.sqrt(np.sum(~np.isnan(prop_3), axis=0)), lw=3)
        ax6.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

        if region_wfs_df.loc[region_wfs_df['type'] == 'RS1'].shape[0] > 0:
            for i in region_wfs_df.loc[(region_wfs_df['type'] == 'RS1') & (waveforms_df['region'] == region)].index:
                ax5.plot(t_x[region_wfs_df.loc[i, 'waveform_2D'].argmin(axis=1)],
                         region_wfs_df.loc[i, 'dist_soma'], color=[.7, .7, .7], alpha=0.2)
            ax5.errorbar(np.nanmedian(prop_2, axis=0), dist_soma,
                         xerr=np.nanstd(prop_2, axis=0)/np.sqrt(np.sum(~np.isnan(prop_2), axis=0)), lw=3)
            ax5.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

        for i in region_wfs_df.loc[region_wfs_df['type'] == 'FS'].index:
            ax4.plot(t_x[region_wfs_df.loc[i, 'waveform_2D'].argmin(axis=1)],
                     region_wfs_df.loc[i, 'dist_soma'], color=[.7, .7, .7], alpha=0.2)
        ax4.errorbar(np.nanmedian(prop_1, axis=0), dist_soma,
                     xerr=np.nanstd(prop_1, axis=0)/np.sqrt(np.sum(~np.isnan(prop_1), axis=0)), lw=3)
        ax4.set(xlim=[1, 2], xlabel='Time (ms)', ylabel='Distance to soma (um)',
                yticks=np.round(np.linspace(-.1, .1, 5), 2))

        plt.savefig(join(FIG_PATH, f'{region}_2D_waveform_groups'))
