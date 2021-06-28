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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from reproducible_ephys_paths import FIG_PATH
from reproducible_ephys_functions import query, data_path, combine_regions, figure_style

# Settings
MIN_SPIKE_AMP = 0.05
MIN_FR = 0.1
MIN_WAVEFORMS = 20
REGIONS = ['PO', 'LP', 'DG', 'CA1', 'PPC']
#FEATURES = ['spike_amp', 'pt_ratio', 'rp_slope', 'spike_width', 'firing_rate', 'peak_to_trough',
#            'spread', 'v_below', 'v_above']
FEATURES_1D = ['spike_amp', 'spike_width', 'pt_ratio', 'rp_slope', 'rc_slope']
FEATURES_2D = ['spike_amp', 'spike_width', 'pt_ratio', 'rp_slope', 'rc_slope',
               'spread', 'v_below', 'v_above']
DATA_DIR = data_path()
FIG_PATH = join(FIG_PATH, 'waveform_clustering')

# Load in waveform data
waveforms_df = pd.read_pickle(join(DATA_DIR, 'waveform_metrics.p'))

# Apply selection criteria
waveforms_df = waveforms_df[((waveforms_df['spike_amp'] >= MIN_SPIKE_AMP)
                             & (waveforms_df['n_waveforms'] >= MIN_WAVEFORMS)
                             & (waveforms_df['firing_rate'] >= MIN_FR))]

# Put data in large 2D array
features_1d = waveforms_df[FEATURES_1D].to_numpy()
features_2d = waveforms_df[FEATURES_2D].to_numpy()

# Do t-SNE embedding for visualization
tsne_embedded_1d = TSNE(n_components=2, perplexity=30, random_state=55).fit_transform(features_1d)
tsne_embedded_2d = TSNE(n_components=2, perplexity=30, random_state=55).fit_transform(features_2d)


# %% Plot all regions
colors = figure_style(return_colors=True)
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
for r, region in enumerate(REGIONS):
    ax1.scatter(tsne_embedded_1d[:, 0][waveforms_df['regions'] == region],
                tsne_embedded_1d[:, 1][waveforms_df['regions'] == region], color=colors[region],
                label=region)
    ax2.scatter(tsne_embedded_2d[:, 0][waveforms_df['regions'] == region],
                tsne_embedded_2d[:, 1][waveforms_df['regions'] == region], color=colors[region],
                label=region)
ax1.get_xaxis().set_visible(False)
ax1.get_yaxis().set_visible(False)
ax1.set(title='1D waveform features')
ax2.get_xaxis().set_visible(False)
ax2.get_yaxis().set_visible(False)
ax2.set(title='1 + 2D waveform features')
ax2.legend(frameon=False)
plt.savefig(join(FIG_PATH, 't-SNE_embedding_all_regions'))

# Plot all histograms
waveforms_df.hist(bins=30, figsize=(15, 10))

# %% Clustering per region

for i, region in enumerate(REGIONS):
    print(f'Processing {region}')

    # Do k-means clustering
    features_1d = waveforms_df[waveforms_df['regions'] == region][FEATURES_1D].to_numpy()
    features_2d = waveforms_df[waveforms_df['regions'] == region][FEATURES_2D].to_numpy()
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features_1d)
    km_ids_1d = kmeans.labels_
    kmeans = KMeans(n_clusters=3, random_state=0).fit(features_2d)
    km_ids_2d = kmeans.labels_
    waveforms_df.loc[waveforms_df['regions'] == region, 'group'] = km_ids_2d

    # Do TNSE embedding
    tsne_1d = TSNE(n_components=2, perplexity=30, random_state=55).fit_transform(features_1d)
    tsne_2d = TSNE(n_components=2, perplexity=30, random_state=55).fit_transform(features_2d)

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=150)

    ax1.scatter(tsne_2d[km_ids_1d == 0, 0], tsne_1d[km_ids_1d == 0, 1])
    ax1.scatter(tsne_2d[km_ids_1d == 1, 0], tsne_1d[km_ids_1d == 1, 1])
    ax1.scatter(tsne_2d[km_ids_1d == 2, 0], tsne_1d[km_ids_1d == 2, 1])
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    ax1.set(title='1D waveform features')

    ax2.scatter(tsne_2d[km_ids_2d == 0, 0], tsne_2d[km_ids_2d == 0, 1], label='RS2')
    ax2.scatter(tsne_2d[km_ids_2d == 1, 0], tsne_2d[km_ids_2d == 1, 1], label='RS1')
    ax2.scatter(tsne_2d[km_ids_2d == 2, 0], tsne_2d[km_ids_2d == 2, 1], label='FS')
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set(title='1D + 2D waveform features')
    ax2.legend(frameon=False)

    plt.savefig(join(FIG_PATH, f'{region}_t-SNE_embedding'))

    d_len = waveforms_df['dist_soma'].apply(lambda x: len(x)).max()
    t_len = waveforms_df['waveform_2D'].apply(lambda x: x.shape[1]).max()
    t_x = np.linspace(0, (t_len / 30000) * 1000, t_len)
    dist_soma = np.round(np.linspace(waveforms_df['dist_soma'].apply(lambda x: np.min(x)).min(),
                                     waveforms_df['dist_soma'].apply(lambda x: np.max(x)).max(),
                                     d_len), 2)

    waveforms_1, size_1 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
    prop_1 = np.full(d_len, np.nan)
    for i in waveforms_df.loc[(waveforms_df['group'] == 0) & (waveforms_df['regions'] == region)].index:
        waveforms_1[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = (
            waveforms_1[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])]
            + waveforms_df.loc[i, 'waveform_2D'])
        size_1[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = size_1[
            np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] + 1
        this_prop = np.full(d_len, np.nan)
        this_prop[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = t_x[
            waveforms_df.loc[i, 'waveform_2D'].argmin(axis=1)]
        prop_1 = np.vstack((prop_1, this_prop))
    waveforms_1 = waveforms_1 / size_1

    waveforms_2, size_2 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
    prop_2 = np.full(d_len, np.nan)
    for i in waveforms_df.loc[(waveforms_df['group'] == 1) & (waveforms_df['regions'] == region)].index:
        waveforms_2[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = (
            waveforms_2[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])]
            + waveforms_df.loc[i, 'waveform_2D'])
        size_2[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = size_2[
            np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] + 1
        this_prop = np.full(d_len, np.nan)
        this_prop[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = t_x[
            waveforms_df.loc[i, 'waveform_2D'].argmin(axis=1)]
        prop_2 = np.vstack((prop_2, this_prop))
    waveforms_2 = waveforms_2 / size_2

    waveforms_3, size_3 = np.zeros((d_len, t_len)), np.zeros((d_len, t_len))
    prop_3 = np.full(d_len, np.nan)
    for i in waveforms_df.loc[(waveforms_df['group'] == 2) & (waveforms_df['regions'] == region)].index:
        waveforms_3[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = (
            waveforms_3[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])]
            + waveforms_df.loc[i, 'waveform_2D'])
        size_3[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = size_3[
            np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] + 1
        this_prop = np.full(d_len, np.nan)
        this_prop[np.in1d(dist_soma, waveforms_df.loc[i, 'dist_soma'])] = t_x[
            waveforms_df.loc[i, 'waveform_2D'].argmin(axis=1)]
        prop_3 = np.vstack((prop_3, this_prop))
    waveforms_3 = waveforms_3 / size_3

    figure_style()
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10, 5), dpi=300)
    ax3.imshow(np.flipud(waveforms_1), cmap='Greys_r', aspect='auto',
               vmin=-np.max(waveforms_1), vmax=np.max(waveforms_1))
    ax3.get_xaxis().set_visible(False)
    ax3.get_yaxis().set_visible(False)
    ax3.set(title='RS2', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))])

    ax2.imshow(np.flipud(waveforms_2), cmap='Greys_r', aspect='auto',
               vmin=-np.max(waveforms_2), vmax=np.max(waveforms_2))
    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)
    ax2.set(title='RS1', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))])

    ax1.imshow(np.flipud(waveforms_3), cmap='Greys_r', aspect='auto',
               vmin=-np.max(waveforms_3), vmax=np.max(waveforms_3))
    ax1.get_xaxis().set_visible(False)
    ax1.set(title='FS', xlim=[np.argmin(np.abs(t_x - 1)), np.argmin(np.abs(t_x - 2))],
            yticks=np.linspace(0, 10, 5), yticklabels=np.round(np.linspace(-.1, .1, 5), 2),
            ylabel='Distance to soma (um)')

    for i in waveforms_df.loc[(waveforms_df['group'] == 0) & (waveforms_df['regions'] == region)].index:
        ax6.plot(t_x[waveforms_df.loc[i, 'waveform_2D'].argmin(axis=1)],
                 waveforms_df.loc[i, 'dist_soma'], color=[.7, .7, .7], alpha=0.2)
    ax6.errorbar(np.nanmedian(prop_1, axis=0), dist_soma,
                 xerr=np.nanstd(prop_1, axis=0)/np.sqrt(np.sum(~np.isnan(prop_1), axis=0)), lw=3)
    ax6.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

    for i in waveforms_df.loc[(waveforms_df['group'] == 1) & (waveforms_df['regions'] == region)].index:
        ax5.plot(t_x[waveforms_df.loc[i, 'waveform_2D'].argmin(axis=1)],
                 waveforms_df.loc[i, 'dist_soma'], color=[.7, .7, .7], alpha=0.2)
    ax5.errorbar(np.nanmedian(prop_2, axis=0), dist_soma,
                 xerr=np.nanstd(prop_2, axis=0)/np.sqrt(np.sum(~np.isnan(prop_2), axis=0)), lw=3)
    ax5.set(xlim=[1, 2], xlabel='Time (ms)', yticks=np.round(np.linspace(-.1, .1, 5), 2))

    for i in waveforms_df.loc[(waveforms_df['group'] == 2) & (waveforms_df['regions'] == region)].index:
        ax4.plot(t_x[waveforms_df.loc[i, 'waveform_2D'].argmin(axis=1)],
                 waveforms_df.loc[i, 'dist_soma'], color=[.7, .7, .7], alpha=0.2)
    ax4.errorbar(np.nanmedian(prop_3, axis=0), dist_soma,
                 xerr=np.nanstd(prop_3, axis=0)/np.sqrt(np.sum(~np.isnan(prop_3), axis=0)), lw=3)
    ax4.set(xlim=[1, 2], xlabel='Time (ms)', ylabel='Distance to soma (um)',
            yticks=np.round(np.linspace(-.1, .1, 5), 2))

    plt.savefig(join(FIG_PATH, f'{region}_2D_waveform_groups'))
