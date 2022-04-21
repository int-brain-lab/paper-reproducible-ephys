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
from permutation_test import permut_test, example_metric
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from reproducible_ephys_functions import data_path, labs, exclude_recordings, figure_style
from reproducible_ephys_paths import FIG_PATH

# Settings
MAX_AP_RMS = 50  # max ap band rms to be included
MIN_REGIONS = 3  # min amount of regions to be included
MIN_CHANNELS = 10  # min amount of channels in a region to say it was targeted
ANNOTATE = False
COLORBAR = True
EXAMPLE_METRIC = 'lfp_power_high'
EXAMPLE_REGION = 'LP'
REGIONS = ['PPC', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['neuron_yield', 'median_firing_rate', 'lfp_power_high', 'rms_ap', 'spike_amp_mean']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS', 'Spike amplitude']
lab_number_map, institution_map, lab_colors = labs()

# Load in data
data = pd.read_csv(join(data_path(), 'metrics_region.csv'))
data['institute'] = data['lab'].map(institution_map)

# Exclude recordings
data = exclude_recordings(data)

# Do some cleanup
data.loc[data['lfp_power_low'] < -100000, 'lfp_power_low'] = np.nan
data['in_recording'] = data['neuron_yield'].isnull() == False

# %% Run permutation test

def permut_dist(data, regions, mice):
    means = []
    for region in np.unique(regions):
        means.append(np.mean(data[regions == region]))
    means = np.array(means)
    return np.sum(np.abs(means - np.mean(means)))


results = pd.DataFrame()
for metric in METRICS:
    p = permut_test(
            data[metric][~np.isnan(data[metric])],
            metric=permut_dist,
            labels1=data['region'].values[~np.isnan(data[metric])],
            labels2=data['subject'].values[~np.isnan(data[metric])])
    results = results.append(pd.DataFrame(index=[results.shape[0] + 1],
                                          data={'metric': metric, 'p_value': p}))

#%%

colors = figure_style(return_colors=True)
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6, 2), dpi=300)

sns.stripplot(data=data, x='region', y='neuron_yield', palette=colors, s=2, zorder=0, ax=ax1)
ax_lines = sns.pointplot(x='region', y='neuron_yield', data=data,
                         ci=0, join=False, estimator=np.mean, color='k',
                         markers="_", scale=1, zorder=1, ax=ax1)
ax1.plot(np.arange(data['region'].unique().shape[0]),
         [data['neuron_yield'].mean()] * data['region'].unique().shape[0],
         color='r', lw=1)
ax1.set(ylabel='Neuron yield', xlabel='', xlim=[-.5, 4.5])
ax1.set_xticklabels(data['region'].unique(), rotation=30, ha='right')

sns.stripplot(data=data, x='region', y='mean_firing_rate', palette=colors, s=2, zorder=0, ax=ax2)
ax_lines = sns.pointplot(x='region', y='mean_firing_rate', data=data,
                         ci=0, join=False, estimator=np.mean, color='k',
                         markers="_", scale=1, zorder=1, ax=ax2)
ax2.plot(np.arange(data['region'].unique().shape[0]),
         [data['mean_firing_rate'].mean()] * data['region'].unique().shape[0],
         color='r', lw=1)
ax2.set(ylabel='Median firing rate (spks/s)', xlabel='', xlim=[-.5, 4.5])
ax2.set_xticklabels(data['region'].unique(), rotation=30, ha='right')

sns.barplot(x=results['metric'], y='p_value', data=results, color='b', ax=ax3)
ax3.set_xticklabels(LABELS, rotation=30, ha='right')
ax3.set(ylabel='Permutation p-value', ylim=[0, 0.5], xlabel='')

sns.despine(trim=True)
plt.tight_layout()
plt.savefig(join(FIG_PATH, 'permutation_regions_example'))



