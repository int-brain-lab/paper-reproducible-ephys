#%%

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
import matplotlib.pyplot as plt
from reproducible_ephys_functions import data_path, labs, exclude_recordings
from reproducible_ephys_paths import FIG_PATH

# Settings
REGIONS = ['VISa', 'CA1', 'DG', 'LP', 'PO']
NICKNAMES = True  # Whether to plot the animal nicknames instead of numbers
MIN_CHANNELS = 5

# Load in data>
metrics = pd.read_csv(join(data_path(), 'figure3_brain_regions.csv'))

# Get lab info
lab_number_map, institution_map, lab_colors = labs()

# Exclude recordings
metrics = exclude_recordings(metrics)

# Reformat data
metrics.loc[metrics['n_channels'] < MIN_CHANNELS, 'neuron_yield'] = np.nan
metrics.loc[metrics['lfp_power_low'] < -100000, 'lfp_power_low'] = np.nan
metrics.loc[metrics['lfp_power_high'] < -100000, 'lfp_power_high'] = np.nan
metrics['institution'] = metrics.lab.map(institution_map)
for i, region in enumerate(REGIONS):
    metrics.loc[metrics['region'] == region, 'region_number'] = i
metrics = metrics.sort_values(['institution', 'subject', 'region_number']).reset_index(drop=True)
n_rec = np.concatenate([np.arange(i) + 1 for i in (metrics.groupby('institution').size() 
                                                   / len(REGIONS))]).astype(int)
metrics['yield_per_channel'] = metrics['neuron_yield'] / metrics['n_channels']
for i, eid in enumerate(metrics['eid'].unique()):
    metrics.loc[metrics['eid'] == eid, 'recording_number'] = i

# Set figure path
if not isdir(join(FIG_PATH, 'brain_regions')):
    mkdir(join(FIG_PATH, 'brain_regions'))
FIG_PATH = join(FIG_PATH, 'brain_regions')

# %% Plots

sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='yield_per_channel').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0,
            cbar_kws={'label': 'Neurons per channel', 'shrink': 0.7}, annot=True,
            annot_kws={"size": 12}, linewidths=.5, fmt='.2f')
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'center'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_yield-per-channel'))

# %% Plot firing rate
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='firing_rate').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0,
            cbar_kws={'label': 'Firing rate (spks/s)', 'shrink': 0.7}, annot=True,
            annot_kws={"size": 12}, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_firing-rate'))

# %% Plot spike amplitude
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='spike_amp_mean').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'Spike amplitude (mV)', 'shrink': 0.7}, annot=True,
            annot_kws={"size": 12}, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_spike-amplitude'))

# %% Plot peak-to-trough ratio
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='pt_ratio').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.2f',
            cbar_kws={'label': 'Peak-to-trough ratio', 'shrink': 0.7}, annot=True,
            annot_kws={"size": 12}, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_pt-ratio'))

# %% Plot repolarization slope
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='rp_slope').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'Repolarization slope', 'shrink': 0.7}, annot=True,
            annot_kws={"size": 12}, linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_rp-slope'))

# %% Plot LFP power
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='lfp_power_high').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=metrics['lfp_power_high'].min(),
            cbar_kws={'label': 'LFP power (dB)', 'shrink': 0.7}, annot=True, annot_kws={"size": 12},
            linewidths=.5, fmt='.0f')
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_lfp-power'))

#%%  Plot rms AP
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='rms_ap').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'AP band (rms)', 'shrink': 0.7}, annot=True, annot_kws={"size": 12},
            linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_rms-ap'))


#%%  Plot neuron count
sns.set(style='ticks', context='paper', font_scale=1.8)
f, ax1 = plt.subplots(1, 1, figsize=(15, 5), dpi=150)
metrics_plot = metrics.pivot(index='region_number', columns='recording_number',
                             values='neuron_yield').sort_values('region_number')
if NICKNAMES:
    metrics_plot = metrics_plot.rename(columns=dict(zip(metrics_plot.columns.values,
                                                        metrics['subject'].unique())))
sns.heatmap(metrics_plot, square=True, cmap='twilight_shifted', center=0, fmt='.0f',
            cbar_kws={'label': 'Neuron count', 'shrink': 0.7}, annot=True, annot_kws={"size": 12},
            linewidths=.5)
ax1.xaxis.tick_top()
ax1.set(xlabel='', ylabel='', xticklabels=metrics_plot.columns.values)
ax1.set_yticklabels(REGIONS, va='center')
if NICKNAMES:
    ax1.set_xticklabels(metrics_plot.columns.values, rotation=30, fontsize=12, ha='left')
    lab_title_y = -1.6
    lab_title_ha = 'left'
else:
    ax1.set(xticklabels=n_rec)
    lab_title_y = -0.8
    lab_title_ha = 'center'
rec_per_lab = metrics.groupby('institution').size() / len(REGIONS)
offset = 0
for i, inst in enumerate(rec_per_lab.index.values):
    ax1.text((rec_per_lab[inst] / 2) + offset, lab_title_y, inst, ha=lab_title_ha,
             color=lab_colors[inst])
    for j in range(int(offset), int(offset + rec_per_lab[inst])):
        plt.gca().get_xticklabels()[j].set_color(lab_colors[inst])
    offset += rec_per_lab[inst]

plt.tight_layout()
plt.savefig(join(FIG_PATH, 'figure3_regions_neuron-count'))


