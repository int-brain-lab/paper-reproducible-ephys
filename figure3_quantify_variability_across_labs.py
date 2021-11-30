#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 09:48:00 2020

@author: guido
"""

from os.path import join, isdir
from os import mkdir
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from permutation_test import permut_test, example_metric
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from reproducible_ephys_functions import data_path, labs, exclude_recordings, figure_style
from reproducible_ephys_paths import FIG_PATH

# Settings
MIN_REC_LAB = 1
ANNOTATE = False
COLORBAR = True
#EXAMPLE_METRIC = 'rms_ap'
EXAMPLE_METRIC = 'lfp_power_high'
EXAMPLE_REGION = 'CA1'
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['yield_per_channel', 'median_firing_rate', 'lfp_power_high', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amp.']
lab_number_map, institution_map, lab_colors = labs()

# Load in data
data = pd.read_csv(join(data_path(), 'metrics_region.csv'))
data['institute'] = data['lab'].map(institution_map)

# Exclude recordings
data = exclude_recordings(data, destriped_rms=False)

# Exclude labs with too few recordings
rec_p_lab = data.groupby(['institute', 'eid']).size().reset_index()['institute'].value_counts()
data = data[data['institute'].isin(rec_p_lab[rec_p_lab >= MIN_REC_LAB].index)]

# Get yield per channel
data['yield_per_channel'] = data['neuron_yield'] / data['n_channels']

# Do some cleanup
data.loc[data['lfp_power_low'] < -100000, 'lfp_power_low'] = np.nan
data['in_recording'] = data['neuron_yield'].isnull() == False

# %% Run permutation test


def permut_dist(data, labs, mice):
    lab_means = []
    for lab in np.unique(labs):
        lab_means.append(np.mean(data[labs == lab]))
    lab_means = np.array(lab_means)
    return np.sum(np.abs(lab_means - np.mean(lab_means)))


results = pd.DataFrame()
for metric in METRICS:
    for region in REGIONS:
        this_data = data.loc[data['region'] == region, metric].values
        p = permut_test(
                this_data[~np.isnan(this_data)],
                metric=permut_dist,
                labels1=data.loc[data['region'] == region, 'institute'].values[~np.isnan(this_data)],
                labels2=data.loc[data['region'] == region, 'subject'].values[~np.isnan(this_data)])
        results = results.append(pd.DataFrame(index=[results.shape[0]+1], data={
            'metric': metric, 'region': region, 'p_value_permut': p}))

# Perform correction for multiple testing
# _, results['p_value_permut'], _, _ = multipletests(results['p_value_permut'], 0.05, method='fdr_tsbh')

# %% Plot results

figure_style()

for i, region in enumerate(REGIONS):
    results.loc[results['region'] == region, 'region_number'] = i

results_plot = results.pivot(index='region_number', columns='metric', values='p_value_permut')
f, ax1 = plt.subplots(1, 1, figsize=(2.5, 2), dpi=300)
sns.heatmap(results_plot, cmap='gist_stern', center=1, square=True,
            cbar=COLORBAR, annot=ANNOTATE, annot_kws={"size": 12},
            linewidths=.5, fmt='.2f', vmin=0, vmax=1, ax=ax1)
cbar = ax1.collections[0].colorbar
#cbar.ax.tick_params(labelsize=12)
#ax1.figure.axes[-1].yaxis.label.set_size(12)
ax1.set(xlabel='', ylabel='', title='Permutation p-values')
ax1.set_yticklabels(REGIONS, va='center', rotation=0)
ax1.set_xticklabels(LABELS, rotation=30, ha='right')

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'permutation_results.png'))
plt.savefig(join(FIG_PATH, 'permutation_results.pdf'))

#%%
data_example = pd.DataFrame(data={
    'institute': data.loc[data['region'] == EXAMPLE_REGION, 'institute'],
    EXAMPLE_METRIC: data.loc[data['region'] == EXAMPLE_REGION, EXAMPLE_METRIC].values})
data_example = data_example.sort_values('institute')
cmap = []
for i, inst in enumerate(data_example['institute'].unique()):
    cmap.append(lab_colors[inst])

f, ax1 = plt.subplots(1, 1, figsize=(1.75, 2), dpi=250)
sns.stripplot(data=data_example, x='institute', y=EXAMPLE_METRIC, palette=cmap, s=3, ax=ax1)
ax_lines = sns.pointplot(x='institute', y=EXAMPLE_METRIC, data=data_example,
                         ci=0, join=False, estimator=np.nanmean, color='k',
                         markers="_", scale=1, ax=ax1)
#plt.setp(ax_lines.collections, zorder=100, label="")
plt.plot(np.arange(data_example['institute'].unique().shape[0]),
         [data_example[EXAMPLE_METRIC].mean()] * data_example['institute'].unique().shape[0],
         color='r', lw=1)
ax1.set(ylabel=u'LFP power in CA1 (dB)', xlabel='',
        xlim=[-.5, data_example['institute'].unique().shape[0] + .5])
ax1.set_xticklabels(data_example['institute'].unique(), rotation=30, ha='right')
#ax1.figure.axes[-1].yaxis.label.set_size(12)

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'permutation_example'))



