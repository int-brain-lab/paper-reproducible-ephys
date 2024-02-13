#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:58:51 2022
By: Guido Meijer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, to_rgb
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from permutation_test import permut_test, distribution_dist_approx_max
from reproducible_ephys_functions import labs
lab_number_map, institution_map, lab_colors = labs()

regions = ['PPC', 'CA1', 'DG', 'LP', 'PO']
metrics = ['yield_per_channel', 'median_firing_rate', 'rms_lf_db',
           'rms_ap', 'spike_amp_mean']
labels = ['Neuron yield', 'Firing rate', 'LFP power',
          'AP band RMS', 'Spike amp.']
example_region = 'CA1'
example_metric = 'rms_lf_db'
n_rec_per_region = 3
n_permut = 50000

# Load in data
data = pd.read_parquet('/home/guido/Data/df_merged.pqt')
data.loc[data['region'] == 'VISa', 'region'] = 'PPC'

results = pd.DataFrame()
for i, metric in enumerate(metrics):
    print(f'Running permutation tests for metric {metric} ({i+1} of {len(metrics)})')
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
        p = permut_test(this_data, metric=distribution_dist_approx_max, labels1=this_labs,
                        labels2=this_subjects, n_permut=n_permut, plot=False, n_cores=4)
        results = pd.concat((results, pd.DataFrame(index=[results.shape[0] + 1], data={
            'metric': metric, 'region': region, 'p_value_permut': p})))

for i, region in enumerate(regions):
    results.loc[results['region'] == region, 'region_number'] = i

results_plot = results.pivot(index='region_number', columns='metric', values='p_value_permut')
results_plot = results_plot.reindex(columns=metrics)
results_plot = np.log10(results_plot)

# %%

f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=300)

axin = inset_axes(ax, width="5%", height="80%", loc='lower right', borderpad=0,
                  bbox_to_anchor=(0.1, 0.1, 1, 1), bbox_transform=ax.transAxes)

# Create colormap
RdYlGn = cm.get_cmap('RdYlGn', 256)(np.linspace(0, 1, 800))


color_array = np.vstack([np.tile(np.concatenate((to_rgb('darkviolet'), [1])), (200, 1)), RdYlGn])
newcmp = ListedColormap(color_array)

sns.heatmap(results_plot, cmap=newcmp, square=True,
            cbar=True, cbar_ax=axin,
            annot=False, annot_kws={"size": 5},
            linewidths=.5, fmt='.2f', vmin=-2.5, vmax=np.log10(1), ax=ax)
cbar = ax.collections[0].colorbar
cbar.set_ticks(np.log10([0.01, 0.1, 1]))
cbar.set_ticklabels([0.01, 0.1, 1])
cbar.set_label('p-value (log scale)', rotation=270, labelpad=8)
ax.set(xlabel='', ylabel='', xticks=np.arange(len(labels)) + 0.5, yticks=np.arange(len(regions)) + 0.5)
ax.set_yticklabels(regions, va='center', rotation=0)
ax.set_xticklabels(labels, rotation=45, ha='right')
plt.tight_layout()

# %%


f, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=300)

data_example = pd.DataFrame(data={
    'institute': data.loc[data['region'] == example_region, 'institute'],
    'lab_number': data.loc[data['region'] == example_region, 'lab_number'],
    example_metric: data.loc[data['region'] == example_region, example_metric].values})
data_example = data_example[~data_example[example_metric].isnull()]

data_example = data_example.sort_values('institute')
cmap = []
for i, inst in enumerate(data_example['institute'].unique()):
    cmap.append(lab_colors[inst])

sns.swarmplot(data=data_example, x='institute', y=example_metric, palette=cmap, s=3, ax=ax)

"""
# Plot lab means and overal mean
ax_lines = sns.pointplot(x='institute', y=example_metric, data=data_example,
                         ci=0, join=False, estimator=np.mean, color='k',
                         markers="_", scale=1, ax=ax)
plt.setp(ax_lines.collections, zorder=100, label="")
ax.plot(np.arange(data_example['institute'].unique().shape[0]),
         [data_example[example_metric].mean()] * data_example['institute'].unique().shape[0],
         color='r', lw=1)
"""

ax.set(ylabel='LFP power in CA1 (db)', xlabel='', xlim=[-.5, len(data_example['institute'].unique())])
ax.set_xticklabels(data_example['institute'].unique(), rotation=90, ha='center')
#ax.plot([-.5, len(data['institute'].unique()) + .5], [-165, -165], lw=0.5, color='k')
#ax.plot([-0.5, -0.5], ax.get_ylim(),  lw=0.5, color='k')
#sns.despine(trim=True)
plt.tight_layout()