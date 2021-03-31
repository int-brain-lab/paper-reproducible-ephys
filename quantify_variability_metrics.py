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
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
from reproducible_ephys_functions import data_path, labs, exclude_recordings
from reproducible_ephys_paths import FIG_PATH

# Settings
MIN_REC_ANOVA = 0
MIN_REC_KW = 0
MAX_AP_RMS = 50  # max ap band rms to be included
MIN_REGIONS = 3  # min amount of regions to be included
MIN_CHANNELS = 10  # min amount of channels in a region to say it was targeted
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']
METRICS = ['neuron_yield', 'firing_rate', 'lfp_power_low', 'rms_ap']
LABELS = ['Neuron yield', 'Firing rate', 'LFP power', 'AP band RMS']
lab_number_map, institution_map, lab_colors = labs()

# Load in data
data = pd.read_csv(join(data_path(), 'figure3_brain_regions.csv'))
data['institute'] = data['lab'].map(institution_map)

# Exclude recordings
data = exclude_recordings(data)

# Do some cleanup
data.loc[data['lfp_power_low'] < -100000, 'lfp_power_low'] = np.nan
data['in_recording'] = data['neuron_yield'].isnull() == False


# %% Run ANOVAs

# Drop areas with too few recordings
data_av = data.groupby(['institute', 'region']).filter(lambda s: s.in_recording.sum()>=MIN_REC_ANOVA)

results = pd.DataFrame()
for metric in METRICS:
    for region in REGIONS:
        mod = ols('%s ~ institute' % metric, data=data_av[data_av['region'] == region]).fit()
        aov_table = sm.stats.anova_lm(mod, typ=2)
        results.loc[len(results) + 1, 'metric'] = metric
        results.loc[len(results), 'region'] = region
        results.loc[len(results), 'f_value_av'] = aov_table.loc['institute', 'F']
        results.loc[len(results), 'p_value_av'] = aov_table.loc['institute', 'PR(>F)']


# %% Run Kruskal-Wallis

# Drop areas with too few recordings
data_kw = data.groupby(['institute', 'region']).filter(lambda s: s.in_recording.sum()>=MIN_REC_KW)

for metric in METRICS:
    for region in REGIONS:
        groups = [group[metric].values for name, group in data_kw[
            (~data_kw[metric].isnull()) & (data_kw['region'] == region)].groupby('institute')]
        if len(groups) > 0:
            h, p = stats.kruskal(*[group[metric].values for name, group
                                   in data_kw[(~data_kw[metric].isnull())
                                              & (data_kw['region'] == region)].groupby('institute')])
        else:
            h = np.nan
            p = np.nan
        results.loc[(results['metric'] == metric) & (results['region'] == region), 'h_value_kw'] = h
        results.loc[(results['metric'] == metric) & (results['region'] == region), 'p_value_kw'] = p


# %% Plot results
for i, region in enumerate(REGIONS):
    results.loc[results['region'] == region, 'region_number'] = i

results_plot = results.pivot(index='region_number', columns='metric', values='f_value_av')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4), dpi=150)
sns.heatmap(results_plot, cmap='twilight_shifted', center=0, square=True,
            cbar=False, annot=True, annot_kws={"size": 12},
            linewidths=.5, fmt='.1f', ax=ax1)
ax1.figure.axes[-1].yaxis.label.set_size(12)
ax1.set(xlabel='', ylabel='', title='ANOVA F-values')
ax1.set_yticklabels(REGIONS, va='center', fontsize=12, rotation=0)
ax1.set_xticklabels(LABELS, rotation=30, fontsize=10, ha='right')

results_plot = results.pivot(index='region_number', columns='metric', values='p_value_av')
hax = sns.heatmap(results_plot, cmap='twilight_shifted_r', center=1, square=True,
                  cbar=False, annot=True, annot_kws={"size": 12},
                  linewidths=.5, fmt='.2f', ax=ax2)
# cbar = hax.collections[0].colorbar
# cbar.set_ticks([np.log10(0.5), np.log10(0.1), np.log10(0.05), np.log10(0.005)])
# cbar.set_ticklabels(['0.5', '0.05', '0.005'])
ax2.figure.axes[-1].yaxis.label.set_size(12)
ax2.set(xlabel='', ylabel='', title='ANOVA p-values')
ax2.set_yticklabels(REGIONS, va='center', fontsize=12, rotation=0)
ax2.set_xticklabels(LABELS, rotation=30, fontsize=10, ha='right')

# Get number of recordings
n_recordings = data.groupby(['institute', 'region']).sum()['in_recording']
n_recordings = n_recordings.unstack(level=0)

# n_recordings.columns = n_recordings.columns.map(institution_map)
sns.heatmap(n_recordings, cmap='twilight_shifted', center=0, square=True,
            annot=True, annot_kws={"size": 12},
            linewidths=.5, fmt='.0f', cbar=False, ax=ax3)
#ax3.xaxis.tick_top()
ax3.set(xlabel='', ylabel='', title='Number of recordings')
ax3.set_xticklabels(n_recordings.columns.values, rotation=45)
ax3.set_yticklabels(REGIONS, va='center', fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'ANOVA_results'))

results_plot = results.pivot(index='region_number', columns='metric', values='h_value_kw')
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4), dpi=150)
sns.heatmap(results_plot, cmap='twilight_shifted', center=0, square=True,
            cbar=False, annot=True, annot_kws={"size": 12},
            linewidths=.5, fmt='.1f', ax=ax1)
ax1.figure.axes[-1].yaxis.label.set_size(12)
ax1.set(xlabel='', ylabel='', title='Kruskal-Wallis H-values')
ax1.set_yticklabels(REGIONS, va='center', fontsize=12, rotation=0)
ax1.set_xticklabels(LABELS, rotation=30, fontsize=10, ha='right')

results_plot = results.pivot(index='region_number', columns='metric', values='p_value_kw')
hax = sns.heatmap(results_plot, cmap='twilight_shifted_r', center=1, square=True,
                  cbar=False, annot=True, annot_kws={"size": 12},
                  linewidths=.5, fmt='.2f', ax=ax2)
# cbar = hax.collections[0].colorbar
# cbar.set_ticks([np.log10(0.5), np.log10(0.1), np.log10(0.05), np.log10(0.005)])
# cbar.set_ticklabels(['0.5', '0.05', '0.005'])
ax2.figure.axes[-1].yaxis.label.set_size(12)
ax2.set(xlabel='', ylabel='', title='Kruskal-Wallis p-values')
ax2.set_yticklabels(REGIONS, va='center', fontsize=12, rotation=0)
ax2.set_xticklabels(LABELS, rotation=30, fontsize=10, ha='right')

# Get number of recordings
n_recordings = data_kw.groupby(['institute', 'region']).sum()['in_recording']
n_recordings = n_recordings.unstack(level=0)

# n_recordings.columns = n_recordings.columns.map(institution_map)
sns.heatmap(n_recordings, cmap='twilight_shifted', center=0, square=True,
            annot=True, annot_kws={"size": 12},
            linewidths=.5, fmt='.0f', cbar=False, ax=ax3)
#ax3.xaxis.tick_top()
ax3.set(xlabel='', ylabel='', title='Number of recordings')
ax3.set_xticklabels(n_recordings.columns.values, rotation=45)
ax3.set_yticklabels(n_recordings.index.values, va='center', fontsize=12, rotation=0)

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'KruskalWallis_results'))



